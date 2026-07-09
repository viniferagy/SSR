#!/usr/bin/env python3
"""Generate and score path-diversity compression samples.

The exact mismatch metric compares diversity under answer-conditioned RCG
against answer-blind reasoning:

    DivCompress = Diversity(R_i ~ P(R|Q,A)) / Diversity(R*_i ~ P(R|Q))

This script keeps generation and scoring separate so expensive vLLM generation
can be resumed from raw JSONL, while embedding scoring can be rerun cheaply.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import inspect
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "scripts" / "rcot_generation"
if str(PROMPT_DIR) not in sys.path:
    sys.path.insert(0, str(PROMPT_DIR))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rcot_prompt import (  # noqa: E402
    AUGSUP_PROMPT,
    CV_SUP_PROMPT,
    DL_SUP_PROMPT,
    FS_SUP_PROMPT,
    NEU_PROMPT,
    PG_SUP_PROMPT,
    QA_SUP_PROMPT,
    SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_PROMPT,
    SSR_PLUS_PROMPT,
    SSR_PLUS_STRUCT_C1_STEPS_PROMPT,
    SSR_PLUS_STRUCT_C2_COMPACT_PROMPT,
    SSR_PLUS_STRUCT_C3_EXACT2_PROMPT,
    SSR_PLUS_STRUCT_C4_BLANKLINE_PROMPT,
    SSR_PLUS_STRUCT_C5_SEPARATE_RULE_PROMPT,
    SSR_PLUS_STRUCT_C6_LONG_STEPS_PROMPT,
    SSR_PLUS_STRUCT_C7_DEVELOPED_PROMPT,
    SSR_PLUS_STRUCT_C8_LONG_REASON_PROMPT,
    SSR_PLUS_STRUCT_C9_LONG_TEMPLATE_PROMPT,
    SSR_PLUS_STRUCT_COMPACT_PROMPT,
    SSR_PLUS_STRUCT_LONG_PROMPT,
    SSR_PLUS_STRUCT_MID_PROMPT,
    SSR_PLUS_STRUCT_PROMPT,
    SSR_PLUS_STRUCT_STEPS_ONLY_PROMPT,
    SSR_PROMPT,
    SUP_PROMPT,
)
from baseline_tier1 import (  # noqa: E402
    FDB_P1_SYSTEM,
    FDB_P2_SYSTEM,
    FDB_P2_USER,
    GIST_P1_SYSTEM,
    GIST_P2_SYSTEM,
    GIST_P2_USER,
    banned_ngram_strings,
    gamma_at,
    parse_bridge as tier_parse_bridge,
    parse_gist as tier_parse_gist,
    parse_reason as tier_parse_reason,
    sample_top_p,
)
from anchoring_measure import reviewer_protocol as reviewer_metrics  # noqa: E402
from two_step_ssr_vllm import (  # noqa: E402
    REASON_SYSTEM as TWO_STEP_REASON_SYSTEM,
    REASON_SYSTEM_SSR_ALIGNED,
    SKELETON_SYSTEM,
    parse_reason as two_step_parse_reason,
    parse_skeleton as two_step_parse_skeleton,
)


PROMPTS = {
    "NEU": NEU_PROMPT,
    "SUP": SUP_PROMPT,
    "AUG-SUP": AUGSUP_PROMPT,
    "CV-SUP": CV_SUP_PROMPT,
    "DL-SUP": DL_SUP_PROMPT,
    "FS-SUP": FS_SUP_PROMPT,
    "QA-SUP": QA_SUP_PROMPT,
    "PG-SUP": PG_SUP_PROMPT,
    # Paper-facing SSR now denotes the final derivational structured variant.
    "SSR": SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R_PROMPT,
    "SSR_PLUS": SSR_PLUS_PROMPT,
    "SSR_PLUS_STRUCT": SSR_PLUS_STRUCT_PROMPT,
    "SSR_PLUS_STRUCT_COMPACT": SSR_PLUS_STRUCT_COMPACT_PROMPT,
    "SSR_PLUS_STRUCT_C1_STEPS": SSR_PLUS_STRUCT_C1_STEPS_PROMPT,
    "SSR_PLUS_STRUCT_C2_COMPACT": SSR_PLUS_STRUCT_C2_COMPACT_PROMPT,
    "SSR_PLUS_STRUCT_C3_EXACT2": SSR_PLUS_STRUCT_C3_EXACT2_PROMPT,
    "SSR_PLUS_STRUCT_C4_BLANKLINE": SSR_PLUS_STRUCT_C4_BLANKLINE_PROMPT,
    "SSR_PLUS_STRUCT_C5_SEPARATE_RULE": SSR_PLUS_STRUCT_C5_SEPARATE_RULE_PROMPT,
    "SSR_PLUS_STRUCT_C6_LONG_STEPS": SSR_PLUS_STRUCT_C6_LONG_STEPS_PROMPT,
    "SSR_PLUS_STRUCT_C7_DEVELOPED": SSR_PLUS_STRUCT_C7_DEVELOPED_PROMPT,
    "SSR_PLUS_STRUCT_C8_LONG_REASON": SSR_PLUS_STRUCT_C8_LONG_REASON_PROMPT,
    "SSR_PLUS_STRUCT_C9_LONG_TEMPLATE": SSR_PLUS_STRUCT_C9_LONG_TEMPLATE_PROMPT,
    "SSR_PLUS_STRUCT_LONG": SSR_PLUS_STRUCT_LONG_PROMPT,
    "SSR_PLUS_STRUCT_MID": SSR_PLUS_STRUCT_MID_PROMPT,
    "SSR_PLUS_STRUCT_STEPS_ONLY": SSR_PLUS_STRUCT_STEPS_ONLY_PROMPT,
}

BLIND_SYSTEM = (
    "You are an expert AI assistant. You will be given a user query only, "
    "without any final answer. Generate a concise but substantive step-by-step "
    "reasoning process that could help solve the query from first principles. "
    "Do not assume or reveal any pre-existing final answer. Do not include the "
    "final answer itself. Output only the reasoning enclosed within <reason> "
    "and </reason> tags."
)

REASON_RE = re.compile(r"<reason>(.*?)</reason>", flags=re.IGNORECASE | re.DOTALL)
SKELETON_RE = re.compile(r"<skeleton>(.*?)</skeleton>", flags=re.IGNORECASE | re.DOTALL)
EXPLANATION_RE = re.compile(r"<\|begin_of_explanation\|>(.*?)<\|end_of_explanation\|>", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Request:
    row_idx: int
    sample_id: Any
    method: str
    condition: str
    prompt: str
    n: int
    bad_words: List[str] | None = None
    candidate_idx: int | None = None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return n


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def apply_chat_template(tokenizer: Any, system_prompt: str, user_content: str) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def fallback_context(row: Dict[str, Any], method: str) -> str:
    question = ensure_text(row.get("questions", {}).get(method))
    answer = ensure_text(row.get("answers", {}).get(method))
    return f"User: {question}\n\nAssistant: {answer}".strip() if answer else question


def prompt_context(row: Dict[str, Any], method: str) -> str:
    context = ensure_text(row.get("contexts", {}).get(method))
    return context if context.strip() else fallback_context(row, method)


def row_question(row: Dict[str, Any], method: str) -> str:
    return ensure_text(row.get("questions", {}).get(method))


def parse_reasoning(raw_output: str) -> Tuple[str, str]:
    raw = ensure_text(raw_output).strip()
    match = REASON_RE.search(raw)
    if match:
        return match.group(1).strip(), "tagged"
    match = EXPLANATION_RE.search(raw)
    if match:
        return match.group(1).strip(), "explanation_tagged"
    match = SKELETON_RE.search(raw)
    if match and len(match.group(1).strip()) > 40:
        return match.group(1).strip(), "skeleton_only"
    lower = raw.lower()
    if "<reason>" in lower:
        start = lower.index("<reason>") + len("<reason>")
        return raw[start:].strip(), "open_reason_tag"
    return raw, "raw"


def load_completed(path: Path) -> Dict[Tuple[int, str, str], List[Dict[str, Any]]]:
    out: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (int(row["row_idx"]), str(row["method"]), str(row["condition"]))
            out[key].append(row)
    return out


TWO_STEP_METHODS = {
    "SSR_QSKEL": {"context_mode": "question", "reason_style": "expand"},
    "SSR_2STEP_QA": {"context_mode": "conversation", "reason_style": "expand"},
}
TIER1_METHODS = {"A1-FDB", "A6-Gist", "B11-NGramBlock"}
B13_METHOD = "B13-BoN"
B12_METHODS = {"B12-Contrastive-g0.5"}


def build_requests(
    rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    methods: Sequence[str],
    k: int,
    include_blind: bool = True,
    include_answer_conditioned: bool = True,
    blind_shard_count: int = 1,
    blind_shard_index: int = 0,
    row_shard_count: int = 1,
    row_shard_index: int = 0,
) -> Tuple[List[Request], List[Dict[str, Any]]]:
    requests: List[Request] = []
    unsupported: List[Dict[str, Any]] = []
    blind_needed_rows: set[int] = set()
    for row_idx, row in enumerate(rows):
        for method in methods:
            q = row_question(row, method)
            if not q.strip():
                unsupported.append({"row_idx": row_idx, "method": method, "reason": "missing_question"})
                continue
            blind_needed_rows.add(row_idx)
            if row_shard_count > 1 and row_idx % row_shard_count != row_shard_index:
                continue
            if not include_answer_conditioned:
                continue
            if method in TWO_STEP_METHODS or method in TIER1_METHODS or method == B13_METHOD or method in B12_METHODS:
                continue
            if method not in PROMPTS:
                unsupported.append({"row_idx": row_idx, "method": method, "reason": "unsupported_exact_generator"})
                continue
            context = prompt_context(row, method)
            if not context.strip():
                unsupported.append({"row_idx": row_idx, "method": method, "reason": "missing_context"})
                continue
            requests.append(Request(row_idx, row.get("id", row_idx), method, "answer_conditioned", apply_chat_template(tokenizer, PROMPTS[method], context), k))
    if include_blind:
        for row_idx in sorted(blind_needed_rows):
            if blind_shard_count > 1 and row_idx % blind_shard_count != blind_shard_index:
                continue
            row = rows[row_idx]
            method = next((m for m in methods if row_question(row, m).strip()), "")
            q = row_question(row, method)
            if q.strip():
                requests.append(Request(row_idx, row.get("id", row_idx), "__BLIND__", "blind", apply_chat_template(tokenizer, BLIND_SYSTEM, q), k))
    return requests, unsupported


def sampling_params(cls: Any, args: argparse.Namespace, n: int, bad_words: List[str] | None = None) -> Any:
    kwargs: Dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": n,
    }
    if bad_words:
        kwargs["bad_words"] = bad_words
    sig = inspect.signature(cls)
    return cls(**{k: v for k, v in kwargs.items() if k in sig.parameters and v is not None})


def generate_request_batch(llm: Any, sampling_cls: Any, requests: Sequence[Request], raw_output: Path, completed: Dict[Tuple[int, str, str], List[Dict[str, Any]]], args: argparse.Namespace) -> None:
    jobs: List[Tuple[Request, int]] = []
    for req in requests:
        key = (req.row_idx, req.method, req.condition)
        have = len(completed.get(key, []))
        if have < req.n:
            jobs.append((req, req.n - have))
    print(json.dumps({"requests": len(requests), "pending": len(jobs)}, indent=2), flush=True)
    for start in range(0, len(jobs), args.chunk_size):
        chunk = jobs[start : start + args.chunk_size]
        prompts = [req.prompt for req, _remain in chunk]
        sps = [sampling_params(sampling_cls, args, remain, req.bad_words) for req, remain in chunk]
        outputs = llm.generate(prompts, sps)
        raw_rows: List[Dict[str, Any]] = []
        for (req, _remain), out in zip(chunk, outputs):
            key = (req.row_idx, req.method, req.condition)
            existing = len(completed.get(key, []))
            for local_idx, cand in enumerate(out.outputs or []):
                raw_rows.append(
                    {
                        "row_idx": req.row_idx,
                        "id": req.sample_id,
                        "method": req.method,
                        "condition": req.condition,
                        "candidate_idx": req.candidate_idx if req.candidate_idx is not None else existing + local_idx,
                        "raw_output": cand.text,
                        "finish_reason": getattr(cand, "finish_reason", None),
                        "output_tokens": len(getattr(cand, "token_ids", []) or []),
                    }
                )
        append_jsonl(raw_output, raw_rows)
        for row in raw_rows:
            completed[(int(row["row_idx"]), str(row["method"]), str(row["condition"]))].append(row)
        print(json.dumps({"chunk_start": start, "completed_candidates": sum(len(v) for v in completed.values())}), flush=True)


def pending_units(requests: Sequence[Request], completed: Dict[Tuple[int, str, str], List[Dict[str, Any]]]) -> int:
    return sum(max(0, req.n - len(completed.get((req.row_idx, req.method, req.condition), []))) for req in requests)


def completed_candidate_indices(completed: Dict[Tuple[int, str, str], List[Dict[str, Any]]], row_idx: int, method: str, condition: str) -> set[int]:
    out: set[int] = set()
    for row in completed.get((row_idx, method, condition), []):
        try:
            out.add(int(row.get("candidate_idx", 0)))
        except Exception:
            continue
    return out


def build_two_step_skeleton_requests(
    rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    methods: Sequence[str],
    k: int,
    row_shard_count: int = 1,
    row_shard_index: int = 0,
) -> List[Request]:
    out: List[Request] = []
    for row_idx, row in enumerate(rows):
        if row_shard_count > 1 and row_idx % row_shard_count != row_shard_index:
            continue
        for method in methods:
            if method not in TWO_STEP_METHODS:
                continue
            context = fallback_context(row, method)
            if not context.strip():
                continue
            user_content = f"Conversation:\n{context}\n\nCreate the abstract skeleton now."
            out.append(Request(row_idx, row.get("id", row_idx), method, "two_step_skeleton", apply_chat_template(tokenizer, SKELETON_SYSTEM, user_content), k))
    return out


def build_two_step_reason_requests(
    rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    methods: Sequence[str],
    completed: Dict[Tuple[int, str, str], List[Dict[str, Any]]],
    row_shard_count: int = 1,
    row_shard_index: int = 0,
) -> List[Request]:
    out: List[Request] = []
    for row_idx, row in enumerate(rows):
        if row_shard_count > 1 and row_idx % row_shard_count != row_shard_index:
            continue
        for method in methods:
            cfg = TWO_STEP_METHODS.get(method)
            if not cfg:
                continue
            skeletons = sorted(completed.get((row_idx, method, "two_step_skeleton"), []), key=lambda x: int(x.get("candidate_idx", 0)))
            for skel_raw in skeletons:
                cand_idx = int(skel_raw.get("candidate_idx", 0))
                if completed.get((row_idx, method, f"two_step_reason_{cand_idx}")):
                    continue
                skeleton, _mode = two_step_parse_skeleton(ensure_text(skel_raw.get("raw_output")))
                if not skeleton.strip():
                    continue
                context = row_question(row, method) if cfg["context_mode"] == "question" else prompt_context(row, method)
                system = REASON_SYSTEM_SSR_ALIGNED if cfg["reason_style"] == "ssr" else TWO_STEP_REASON_SYSTEM
                user_content = (
                    f"Input:\n{context}\n\n"
                    f"Fixed skeleton:\n<skeleton>\n{skeleton}\n</skeleton>\n\n"
                    "Expand the reasoning now."
                )
                out.append(Request(row_idx, row.get("id", row_idx), method, f"two_step_reason_{cand_idx}", apply_chat_template(tokenizer, system, user_content), 1, candidate_idx=cand_idx))
    return out


def build_tier1_stage_requests(
    rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    methods: Sequence[str],
    completed: Dict[Tuple[int, str, str], List[Dict[str, Any]]],
    k: int,
    row_shard_count: int = 1,
    row_shard_index: int = 0,
) -> List[Request]:
    out: List[Request] = []
    for row_idx, row in enumerate(rows):
        if row_shard_count > 1 and row_idx % row_shard_count != row_shard_index:
            continue
        base_method = next((m for m in ("NEU", "SSR", "AUG-SUP", "SUP") if row_question(row, m).strip()), "")
        for method in methods:
            if method == "A1-FDB":
                q = row_question(row, method) or row_question(row, base_method)
                if q.strip():
                    out.append(Request(row_idx, row.get("id", row_idx), method, "a1_p1", apply_chat_template(tokenizer, FDB_P1_SYSTEM, q), k))
            elif method == "A6-Gist":
                q = row_question(row, method) or row_question(row, base_method)
                a = ensure_text(row.get("answers", {}).get(method)) or ensure_text(row.get("answers", {}).get(base_method))
                if q.strip() and a.strip():
                    user = f"[QUESTION]\n{q}\n\n[REFERENCE ANSWER]\n{a}"
                    out.append(Request(row_idx, row.get("id", row_idx), method, "a6_p1", apply_chat_template(tokenizer, GIST_P1_SYSTEM, user), k))
            elif method == "B11-NGramBlock":
                context = prompt_context(row, method)
                answer = ensure_text(row.get("answers", {}).get(method))
                if context.strip() and answer.strip():
                    out.append(Request(row_idx, row.get("id", row_idx), method, "answer_conditioned", apply_chat_template(tokenizer, PROMPTS["NEU"], context), k, bad_words=banned_ngram_strings(tokenizer, answer)))
    return out


def build_b13_bon_requests(
    rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    methods: Sequence[str],
    completed: Dict[Tuple[int, str, str], List[Dict[str, Any]]],
    k: int,
    bon_n: int,
    row_shard_count: int = 1,
    row_shard_index: int = 0,
) -> List[Request]:
    if B13_METHOD not in methods:
        return []
    out: List[Request] = []
    for row_idx, row in enumerate(rows):
        if row_shard_count > 1 and row_idx % row_shard_count != row_shard_index:
            continue
        q = row_question(row, B13_METHOD)
        context = prompt_context(row, B13_METHOD)
        if not (q.strip() and context.strip()):
            continue
        final_done = completed_candidate_indices(completed, row_idx, B13_METHOD, "answer_conditioned")
        for rep in range(k):
            if rep in final_done:
                continue
            condition = f"b13_bon_{rep}"
            have = len(completed.get((row_idx, B13_METHOD, condition), []))
            if have < bon_n:
                out.append(Request(row_idx, row.get("id", row_idx), B13_METHOD, condition, apply_chat_template(tokenizer, PROMPTS["NEU"], context), bon_n))
    return out


def build_tier1_second_stage_requests(
    rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    methods: Sequence[str],
    completed: Dict[Tuple[int, str, str], List[Dict[str, Any]]],
    row_shard_count: int = 1,
    row_shard_index: int = 0,
) -> List[Request]:
    out: List[Request] = []
    for row_idx, row in enumerate(rows):
        if row_shard_count > 1 and row_idx % row_shard_count != row_shard_index:
            continue
        if "A1-FDB" in methods:
            q = row_question(row, "A1-FDB")
            a = ensure_text(row.get("answers", {}).get("A1-FDB"))
            for p1 in sorted(completed.get((row_idx, "A1-FDB", "a1_p1"), []), key=lambda x: int(x.get("candidate_idx", 0))):
                cand_idx = int(p1.get("candidate_idx", 0))
                if completed.get((row_idx, "A1-FDB", f"a1_p2_{cand_idx}")):
                    continue
                draft = tier_parse_reason(ensure_text(p1.get("raw_output")))
                user = FDB_P2_USER.format(question=q, answer=a, draft=draft)
                out.append(Request(row_idx, row.get("id", row_idx), "A1-FDB", f"a1_p2_{cand_idx}", apply_chat_template(tokenizer, FDB_P2_SYSTEM, user), 1, candidate_idx=cand_idx))
        if "A6-Gist" in methods:
            q = row_question(row, "A6-Gist")
            a = ensure_text(row.get("answers", {}).get("A6-Gist"))
            for p1 in sorted(completed.get((row_idx, "A6-Gist", "a6_p1"), []), key=lambda x: int(x.get("candidate_idx", 0))):
                cand_idx = int(p1.get("candidate_idx", 0))
                if completed.get((row_idx, "A6-Gist", f"a6_p2_{cand_idx}")):
                    continue
                gists = tier_parse_gist(ensure_text(p1.get("raw_output")))
                user = GIST_P2_USER.format(question=q, answer=a, gists=gists)
                out.append(Request(row_idx, row.get("id", row_idx), "A6-Gist", f"a6_p2_{cand_idx}", apply_chat_template(tokenizer, GIST_P2_SYSTEM, user), 1, candidate_idx=cand_idx))
    return out


def free_llm(llm: Any) -> None:
    if llm is None:
        return
    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_hf_model_and_tokenizer(model_path: Path, device: Any, device_map: str = "") -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map=device_map if device_map else None,
    )
    if not device_map:
        model.to(device)
    model.eval()
    return model, tokenizer


@reviewer_metrics.torch.no_grad()
def score_b13_candidate(model: Any, tokenizer: Any, row: Dict[str, Any], reasoning: str, device: Any, args: argparse.Namespace) -> Dict[str, float]:
    q = row_question(row, B13_METHOD)
    a = ensure_text(row.get("answers", {}).get(B13_METHOD))
    if not (q.strip() and a.strip() and reasoning.strip()):
        raise ValueError("missing Q/A/R for B13 candidate")
    sep_ids = reviewer_metrics.tok_1d(tokenizer, reviewer_metrics.SEQ_SEP, device)
    think_close_ids = reviewer_metrics.tok_1d(tokenizer, "</think>\n\n", device)
    ans_ids = reviewer_metrics.tok_1d(tokenizer, a, device)
    n_steps, flat_reason, end_pos = reviewer_metrics.tokenize_reasoning(tokenizer, reasoning, sep_ids, device)
    if int(ans_ids.numel()) == 0 or int(flat_reason.numel()) == 0:
        raise ValueError("empty answer or reasoning ids")

    max_ctx = getattr(model.config, "max_position_embeddings", None)
    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))
    prefix_q = reviewer_metrics.tok_1d(tokenizer, reviewer_metrics.apply_chat_prefix(tokenizer, q), device)
    base_prompt = reviewer_metrics.torch.cat([prefix_q, think_close_ids], dim=0)
    step_prompts = [base_prompt]
    for s in reviewer_metrics.step_idx_generator(n_steps)[1:]:
        r_end = int(end_pos[s - 1]) if s - 1 < len(end_pos) else int(end_pos[-1])
        step_prompts.append(reviewer_metrics.torch.cat([prefix_q, flat_reason[:r_end], think_close_ids], dim=0))
    logps = reviewer_metrics.batch_answer_logprob(model, step_prompts, ans_ids, max_ctx, reviewer_metrics.STEP_MICROBATCH)
    base_log2 = float(logps[0] * reviewer_metrics.LOGE_TO_BITS / int(ans_ids.numel()))
    final_log2 = float(logps[-1] * reviewer_metrics.LOGE_TO_BITS / int(ans_ids.numel()))
    token_h = reviewer_metrics.reasoning_entropies(model, flat_reason, prefix_q, reviewer_metrics.ENTROPY_BATCH)
    aent = reviewer_metrics.entropy_anchoring(token_h, end_pos, vocab_size, args.b13_tau_g)
    return {"B": final_log2 - base_log2, "Aent_raw": aent, "n_steps": float(n_steps), "reasoning_tokens": float(flat_reason.numel())}


def choose_b13_candidate(scored: List[Tuple[int, Dict[str, float], str]], rule: str) -> Tuple[int, Dict[str, float], str]:
    if rule == "prob":
        return min(scored, key=lambda x: float(x[1]["B"]))
    probs = [float(x[1]["B"]) for x in scored]
    ents = [float(x[1]["Aent_raw"]) for x in scored]
    p_lo, p_hi = min(probs), max(probs)
    e_lo, e_hi = min(ents), max(ents)

    def mm(v: float, lo: float, hi: float) -> float:
        return 0.5 if math.isclose(lo, hi) else (v - lo) / (hi - lo)

    return min(scored, key=lambda x: 0.5 * mm(float(x[1]["B"]), p_lo, p_hi) + 0.5 * mm(float(x[1]["Aent_raw"]), e_lo, e_hi))


def select_b13_bon_samples(
    rows: Sequence[Dict[str, Any]],
    completed: Dict[Tuple[int, str, str], List[Dict[str, Any]]],
    raw_output: Path,
    args: argparse.Namespace,
) -> None:
    if B13_METHOD not in [m for m in args.methods.split(",") if m]:
        return
    jobs: List[Tuple[int, int]] = []
    for row_idx, row in enumerate(rows):
        if row_idx % max(1, args.row_shard_count) != args.row_shard_index:
            continue
        if not row_question(row, B13_METHOD).strip():
            continue
        final_done = completed_candidate_indices(completed, row_idx, B13_METHOD, "answer_conditioned")
        for rep in range(args.k):
            if rep in final_done:
                continue
            if len(completed.get((row_idx, B13_METHOD, f"b13_bon_{rep}"), [])) >= args.b13_bon_n:
                jobs.append((row_idx, rep))
    if not jobs:
        return

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device_map = args.b13_device_map if args.b13_device_map else None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=reviewer_metrics.TORCH_DTYPE if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=reviewer_metrics.TORCH_DTYPE if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
    if device_map is None:
        model.to(device)
    model.eval()
    device = next(model.parameters()).device

    selected_rows: List[Dict[str, Any]] = []
    for row_idx, rep in jobs:
        row = rows[row_idx]
        vals = sorted(completed.get((row_idx, B13_METHOD, f"b13_bon_{rep}"), []), key=lambda x: int(x.get("candidate_idx", 0)))[: args.b13_bon_n]
        scored: List[Tuple[int, Dict[str, float], str]] = []
        fallback: Tuple[int, str] | None = None
        for val in vals:
            cand_idx = int(val.get("candidate_idx", 0))
            reasoning, _mode = parse_reasoning(ensure_text(val.get("raw_output")))
            if not reasoning.strip():
                continue
            if fallback is None:
                fallback = (cand_idx, reasoning)
            try:
                scored.append((cand_idx, score_b13_candidate(model, tokenizer, row, reasoning, device, args), reasoning))
            except Exception:
                continue
        if scored:
            cand_idx, metrics, reasoning = choose_b13_candidate(scored, args.b13_selection_rule)
            selection = {"selected_candidate": cand_idx, "selection_rule": args.b13_selection_rule, **metrics}
        elif fallback:
            cand_idx, reasoning = fallback
            selection = {"selected_candidate": cand_idx, "selection_rule": "fallback_unscored"}
        else:
            continue
        selected_rows.append(
            {
                "row_idx": row_idx,
                "id": row.get("id", row_idx),
                "method": B13_METHOD,
                "condition": "answer_conditioned",
                "candidate_idx": rep,
                "raw_output": reasoning,
                "finish_reason": "selected_bon",
                "output_tokens": 0,
                "selection": selection,
            }
        )
    append_jsonl(raw_output, selected_rows)
    for row in selected_rows:
        completed[(int(row["row_idx"]), str(row["method"]), str(row["condition"]))].append(row)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def b12_gamma(method: str, default: float) -> float:
    match = re.search(r"-g([0-9.]+)$", method)
    return float(match.group(1)) if match else default


def hf_encode(tokenizer: Any, prompts: List[str], device: Any, max_prompt_tokens: int) -> Any:
    return tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_tokens).to(device)


def prefill_contrastive(model: Any, tokenizer: Any, prompts: List[str], device: Any, max_prompt_tokens: int) -> Tuple[Any, Any, Any, Any]:
    enc = hf_encode(tokenizer, prompts, device, max_prompt_tokens)
    pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    with reviewer_metrics.torch.inference_mode():
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, position_ids=pos, use_cache=True)
    counts = enc.attention_mask.sum(-1)
    return out.logits[:, -1, :], out.past_key_values, enc.attention_mask, counts


def generate_b12_batch(model: Any, tokenizer: Any, batch: Sequence[Tuple[int, Dict[str, Any], str, int]], args: argparse.Namespace, device: Any) -> List[Dict[str, Any]]:
    import torch

    with_prompts: List[str] = []
    without_prompts: List[str] = []
    for _row_idx, row, method, _cand_idx in batch:
        with_prompts.append(apply_chat_template(tokenizer, PROMPTS["NEU"], prompt_context(row, method)))
        without_prompts.append(apply_chat_template(tokenizer, PROMPTS["NEU"], row_question(row, method)))

    logits_w, cache_w, mask_w, count_w = prefill_contrastive(model, tokenizer, with_prompts, device, args.b12_max_prompt_tokens)
    logits_o, cache_o, mask_o, count_o = prefill_contrastive(model, tokenizer, without_prompts, device, args.b12_max_prompt_tokens)
    logits_w = logits_w.to(device)
    logits_o = logits_o.to(device)

    eos = tokenizer.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    batch_size = len(batch)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated: List[List[int]] = [[] for _ in range(batch_size)]

    with torch.inference_mode():
        for step in range(args.b12_max_new_tokens):
            gamma = b12_gamma(batch[0][2], args.b12_gamma)
            gamma = gamma_at(step, gamma, args.b12_max_new_tokens, args.b12_anneal_frac)
            logits = logits_o + gamma * (logits_w - logits_o)
            next_ids = sample_top_p(logits, args.temperature, args.top_p)
            next_ids = torch.where(done, torch.full_like(next_ids, tokenizer.pad_token_id), next_ids)
            for i in range(batch_size):
                if not bool(done[i]):
                    generated[i].append(int(next_ids[i]))
            just_done = torch.tensor([int(next_ids[i]) in eos_ids for i in range(batch_size)], device=device)
            done = done | just_done
            if bool(done.all()):
                break

            step_mask = (~done).long().unsqueeze(1)
            ids_step = next_ids.unsqueeze(1)
            mask_w = torch.cat([mask_w, step_mask], dim=1)
            mask_o = torch.cat([mask_o, step_mask], dim=1)
            out_w = model(input_ids=ids_step, attention_mask=mask_w, position_ids=count_w.unsqueeze(1), past_key_values=cache_w, use_cache=True)
            out_o = model(input_ids=ids_step, attention_mask=mask_o, position_ids=count_o.unsqueeze(1), past_key_values=cache_o, use_cache=True)
            logits_w, logits_o = out_w.logits[:, -1, :].to(device), out_o.logits[:, -1, :].to(device)
            cache_w, cache_o = out_w.past_key_values, out_o.past_key_values
            count_w = count_w + step_mask.squeeze(1)
            count_o = count_o + step_mask.squeeze(1)

    out_rows: List[Dict[str, Any]] = []
    for (row_idx, row, method, cand_idx), ids in zip(batch, generated):
        ids = [x for x in ids if x not in eos_ids and x != tokenizer.pad_token_id]
        raw_text = tokenizer.decode(ids, skip_special_tokens=True)
        out_rows.append(
            {
                "row_idx": row_idx,
                "id": row.get("id", row_idx),
                "method": method,
                "condition": "answer_conditioned",
                "candidate_idx": cand_idx,
                "raw_output": tier_parse_reason(raw_text),
                "finish_reason": "contrastive",
                "output_tokens": len(ids),
                "contrastive": {"gamma": b12_gamma(method, args.b12_gamma), "anneal_frac": args.b12_anneal_frac},
            }
        )
    return out_rows


def generate_b12_samples(
    rows: Sequence[Dict[str, Any]],
    methods: Sequence[str],
    completed: Dict[Tuple[int, str, str], List[Dict[str, Any]]],
    raw_output: Path,
    args: argparse.Namespace,
) -> None:
    b12_methods = [m for m in methods if m in B12_METHODS]
    if not b12_methods:
        return
    jobs: List[Tuple[int, Dict[str, Any], str, int]] = []
    for row_idx, row in enumerate(rows):
        if row_idx % max(1, args.row_shard_count) != args.row_shard_index:
            continue
        for method in b12_methods:
            if not (row_question(row, method).strip() and prompt_context(row, method).strip()):
                continue
            done = completed_candidate_indices(completed, row_idx, method, "answer_conditioned")
            for cand_idx in range(args.k):
                if cand_idx not in done:
                    jobs.append((row_idx, row, method, cand_idx))
    if not jobs:
        return

    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_hf_model_and_tokenizer(args.model, device, args.b12_device_map)
    written = 0
    for start in range(0, len(jobs), args.b12_batch_size):
        batch = jobs[start : start + args.b12_batch_size]
        out_rows = generate_b12_batch(model, tokenizer, batch, args, device)
        append_jsonl(raw_output, out_rows)
        for row in out_rows:
            completed[(int(row["row_idx"]), str(row["method"]), str(row["condition"]))].append(row)
        written += len(out_rows)
        print(json.dumps({"b12_completed": written, "b12_total": len(jobs)}), flush=True)
        del out_rows
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    use_vllm = args.backend == "vllm" or (args.backend == "auto" and importlib.util.find_spec("vllm") is not None)
    if args.phase in {"all", "vllm"} and not use_vllm:
        raise RuntimeError("Path diversity generation requires vLLM; no HF backend fallback is available.")
    if args.phase in {"all", "vllm"}:
        from vllm import LLM, SamplingParams
    else:
        LLM = SamplingParams = None

    rows_all = read_jsonl(args.input)
    rows = rows_all[: args.limit] if args.limit is not None else rows_all
    methods = [m for m in args.methods.split(",") if m]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    requests, unsupported = build_requests(
        rows,
        tokenizer,
        methods,
        args.k,
        include_blind=not args.skip_blind,
        include_answer_conditioned=not args.only_blind,
        blind_shard_count=max(1, args.blind_shard_count),
        blind_shard_index=args.blind_shard_index,
        row_shard_count=max(1, args.row_shard_count),
        row_shard_index=args.row_shard_index,
    )
    completed = load_completed(args.raw_output)
    if args.only_blind:
        two_step_skeleton: List[Request] = []
        tier1_stage1: List[Request] = []
        b13_bon: List[Request] = []
    else:
        two_step_skeleton = build_two_step_skeleton_requests(
            rows,
            tokenizer,
            methods,
            args.k,
            row_shard_count=max(1, args.row_shard_count),
            row_shard_index=args.row_shard_index,
        )
        tier1_stage1 = build_tier1_stage_requests(
            rows,
            tokenizer,
            methods,
            completed,
            args.k,
            row_shard_count=max(1, args.row_shard_count),
            row_shard_index=args.row_shard_index,
        )
        b13_bon = build_b13_bon_requests(
            rows,
            tokenizer,
            methods,
            completed,
            args.k,
            args.b13_bon_n,
            row_shard_count=max(1, args.row_shard_count),
            row_shard_index=args.row_shard_index,
        )
    pending_count = pending_units(requests + two_step_skeleton + tier1_stage1 + b13_bon, completed)
    print(json.dumps({"requests": len(requests), "two_step_skeleton": len(two_step_skeleton), "tier1_stage1": len(tier1_stage1), "b13_bon": len(b13_bon), "pending": pending_count, "unsupported": len(unsupported)}, ensure_ascii=False, indent=2), flush=True)

    llm = None
    if args.phase in {"all", "vllm"} and pending_count:
        llm = LLM(
            model=str(args.model),
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
            seed=args.seed,
        )
        generate_request_batch(llm, SamplingParams, requests + two_step_skeleton + tier1_stage1 + b13_bon, args.raw_output, completed, args)
    if args.phase in {"all", "vllm"}:
        two_step_reason = (
            []
            if args.only_blind
            else build_two_step_reason_requests(
                rows,
                tokenizer,
                methods,
                completed,
                row_shard_count=max(1, args.row_shard_count),
                row_shard_index=args.row_shard_index,
            )
        )
        tier1_stage2 = (
            []
            if args.only_blind
            else build_tier1_second_stage_requests(
                rows,
                tokenizer,
                methods,
                completed,
                row_shard_count=max(1, args.row_shard_count),
                row_shard_index=args.row_shard_index,
            )
        )
        pending_second = pending_units(two_step_reason + tier1_stage2, completed)
        if pending_second:
            if llm is None:
                llm = LLM(
                    model=str(args.model),
                    tensor_parallel_size=args.tensor_parallel_size,
                    trust_remote_code=True,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_num_seqs=args.max_num_seqs,
                    enforce_eager=args.enforce_eager,
                    disable_custom_all_reduce=args.disable_custom_all_reduce,
                    seed=args.seed,
                )
            generate_request_batch(llm, SamplingParams, two_step_reason + tier1_stage2, args.raw_output, completed, args)
    free_llm(llm)
    llm = None
    gc.collect()
    if args.phase in {"all", "special"} and not args.only_blind:
        completed = load_completed(args.raw_output)
        select_b13_bon_samples(rows, completed, args.raw_output, args)
        completed = load_completed(args.raw_output)
        generate_b12_samples(rows, methods, completed, args.raw_output, args)
        completed = load_completed(args.raw_output)

    if args.skip_finalize:
        return

    parsed_rows: List[Dict[str, Any]] = []
    seen_final: set[Tuple[int, str, int]] = set()
    for key, vals in sorted(completed.items()):
        row_idx, method, condition = key
        if row_idx >= len(rows):
            continue
        if condition in {"two_step_skeleton", "a1_p1", "a6_p1"} or condition.startswith("b13_bon_"):
            continue
        for val in sorted(vals, key=lambda x: int(x.get("candidate_idx", 0)))[: args.k]:
            candidate_idx = int(val.get("candidate_idx", 0))
            final_key = (row_idx, method, candidate_idx)
            if final_key in seen_final:
                continue
            raw_text = ensure_text(val.get("raw_output"))
            if condition.startswith("two_step_reason_"):
                reasoning, mode = two_step_parse_reason(raw_text)
                parse_mode = f"two_step_{mode}"
            elif condition.startswith("a1_p2_"):
                p1 = next((x for x in completed.get((row_idx, method, "a1_p1"), []) if int(x.get("candidate_idx", 0)) == candidate_idx), None)
                draft = tier_parse_reason(ensure_text(p1.get("raw_output") if p1 else ""))
                bridge = tier_parse_bridge(raw_text)
                reasoning = (draft + "\n\n" + bridge).strip()
                parse_mode = "a1_fdb"
            elif condition.startswith("a6_p2_"):
                reasoning = tier_parse_reason(raw_text)
                parse_mode = "a6_gist"
            else:
                reasoning, mode = parse_reasoning(raw_text)
                parse_mode = mode
            if reasoning.strip():
                parsed_rows.append({**val, "condition": "answer_conditioned" if condition != "blind" else "blind", "reasoning": reasoning, "parse_mode": parse_mode})
                seen_final.add(final_key)
    write_jsonl(args.output, parsed_rows)
    if args.unsupported_output:
        write_jsonl(args.unsupported_output, unsupported)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "raw_output": str(args.raw_output),
        "rows": len(rows),
        "methods": methods,
        "k": args.k,
        "parsed_candidates": len(parsed_rows),
        "unsupported": len(unsupported),
    }
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def pairwise_distance(vecs: np.ndarray) -> float:
    if len(vecs) < 2:
        return math.nan
    vals: List[float] = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            vals.append(1.0 - float(np.dot(vecs[i], vecs[j])))
    return float(np.mean(vals)) if vals else math.nan


def percentile(values: Sequence[float], pct: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return math.nan
    pos = (len(vals) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def bootstrap_ci(values: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float]:
    import random

    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return math.nan, math.nan
    rng = random.Random(seed)
    boots = [float(np.mean([rng.choice(vals) for _ in vals])) for _ in range(n_boot)]
    return percentile(boots, 0.025), percentile(boots, 0.975)


def valid_per_sample_row(row: Dict[str, Any], k: int) -> bool:
    try:
        conditioned_k = int(float(row.get("conditioned_k", 0) or 0))
        blind_k = int(float(row.get("blind_k", 0) or 0))
        div = float(row.get("DivCompress", "nan"))
    except Exception:
        return False
    return conditioned_k >= k and blind_k >= k and math.isfinite(div)


def write_score_summary(
    per_sample: Sequence[Dict[str, Any]],
    output_dir: Path,
    methods: Sequence[str],
    k: int,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    valid_rows = [row for row in per_sample if valid_per_sample_row(row, k)]
    summary_rows: List[Dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in valid_rows if str(row.get("method", "")) == method]
        vals = [float(row["DivCompress"]) for row in method_rows]
        lo, hi = bootstrap_ci(vals, n_boot, seed)
        cvals = [float(row["conditioned_diversity"]) for row in method_rows if math.isfinite(float(row["conditioned_diversity"]))]
        bvals = [float(row["blind_diversity"]) for row in method_rows if math.isfinite(float(row["blind_diversity"]))]
        summary_rows.append(
            {
                "method": method,
                "n": len(vals),
                "DivCompress_mean": float(np.mean(vals)) if vals else math.nan,
                "DivCompress_ci95_low": lo,
                "DivCompress_ci95_high": hi,
                "conditioned_diversity_mean": float(np.mean(cvals)) if cvals else math.nan,
                "blind_diversity_mean": float(np.mean(bvals)) if bvals else math.nan,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "path_diversity_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["method", "n", "DivCompress_mean", "DivCompress_ci95_low", "DivCompress_ci95_high", "conditioned_diversity_mean", "blind_diversity_mean"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    excluded = len(per_sample) - len(valid_rows)
    lines = [
        "# Path Diversity Compression",
        "",
        f"- valid rows: {len(valid_rows)}",
        f"- excluded rows: {excluded}",
        f"- required k: {k}",
        "",
        "| method | n | DivCompress | 95% CI | conditioned div | blind div |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['n']} | {float(row['DivCompress_mean']):.6f} | "
            f"[{float(row['DivCompress_ci95_low']):.6f}, {float(row['DivCompress_ci95_high']):.6f}] | "
            f"{float(row['conditioned_diversity_mean']):.6f} | {float(row['blind_diversity_mean']):.6f} |"
        )
    (output_dir / "path_diversity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"valid_per_sample": len(valid_rows), "excluded_per_sample": excluded, "methods": list(methods)}


def score(args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer

    rows = read_jsonl(args.input)
    grouped: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["row_idx"]), str(row["method"]), str(row["condition"]))].append(row)

    texts = [ensure_text(row.get("reasoning")) for row in rows]
    model_kwargs: Dict[str, Any] = {}
    if args.device:
        model_kwargs["device"] = args.device
    model = SentenceTransformer(str(args.embedding_model), **model_kwargs)
    model.max_seq_length = args.max_length
    vecs = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    for row, idx in zip(rows, range(len(rows))):
        row["_vec_idx"] = idx

    methods = sorted({m for _idx, m, cond in grouped if cond == "answer_conditioned" and m != "__BLIND__"})
    per_sample: List[Dict[str, Any]] = []
    for (row_idx, method, cond), vals in grouped.items():
        if cond != "answer_conditioned" or method == "__BLIND__":
            continue
        blind_vals = grouped.get((row_idx, "__BLIND__", "blind"), [])
        if len(vals) < 2 or len(blind_vals) < 2:
            continue
        cond_vecs = vecs[[int(v["_vec_idx"]) for v in vals]]
        blind_vecs = vecs[[int(v["_vec_idx"]) for v in blind_vals]]
        cond_div = pairwise_distance(cond_vecs)
        blind_div = pairwise_distance(blind_vecs)
        ratio = cond_div / blind_div if math.isfinite(cond_div) and math.isfinite(blind_div) and blind_div > 0 else math.nan
        per_sample.append(
            {
                "sample_idx": row_idx,
                "method": method,
                "conditioned_diversity": cond_div,
                "blind_diversity": blind_div,
                "DivCompress": ratio,
                "conditioned_k": len(vals),
                "blind_k": len(blind_vals),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample_path_diversity.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["sample_idx", "method", "conditioned_diversity", "blind_diversity", "DivCompress", "conditioned_k", "blind_k"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_sample)

    summary = write_score_summary(per_sample, args.output_dir, methods, args.k, args.bootstrap, args.seed)
    print(json.dumps({"rows": len(rows), "per_sample": len(per_sample), **summary, "output_dir": str(args.output_dir)}, indent=2), flush=True)


def summarize(args: argparse.Namespace) -> None:
    per_sample: List[Dict[str, Any]] = []
    with args.per_sample.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        per_sample.extend(dict(row) for row in reader)
    methods = sorted({str(row.get("method", "")) for row in per_sample if row.get("method")})
    summary = write_score_summary(per_sample, args.output_dir, methods, args.k, args.bootstrap, args.seed)
    print(json.dumps({"per_sample": len(per_sample), **summary, "output_dir": str(args.output_dir)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--raw-output", type=Path, required=True)
    p.add_argument("--unsupported-output", type=Path)
    p.add_argument("--summary-output", type=Path)
    p.add_argument("--model", type=Path, default=Path("/home/pengguangyue/workspace/models/Qwen/Qwen3-8B"))
    p.add_argument("--methods", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-num-seqs", type=int)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--disable-custom-all-reduce", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phase", choices=["all", "vllm", "special", "finalize"], default="all")
    p.add_argument("--backend", choices=["auto", "vllm"], default="auto")
    p.add_argument("--skip-blind", action="store_true", help="Do not generate the answer-blind baseline samples in this shard.")
    p.add_argument("--only-blind", action="store_true", help="Generate only answer-blind baseline samples for the selected rows.")
    p.add_argument("--blind-shard-count", type=int, default=1)
    p.add_argument("--blind-shard-index", type=int, default=0)
    p.add_argument("--row-shard-count", type=int, default=1, help="Shard answer-conditioned generation by row index modulo this count.")
    p.add_argument("--row-shard-index", type=int, default=0, help="Shard index for answer-conditioned row sharding.")
    p.add_argument("--skip-finalize", action="store_true", help="Only append raw generation rows; do not rewrite parsed output or summaries.")
    p.add_argument("--b13-bon-n", type=int, default=8)
    p.add_argument("--b13-selection-rule", choices=["combo", "prob"], default="combo")
    p.add_argument("--b13-tau-g", type=float, default=0.1)
    p.add_argument("--b13-device-map", default="")
    p.add_argument("--b12-gamma", type=float, default=0.5)
    p.add_argument("--b12-anneal-frac", type=float, default=0.15)
    p.add_argument("--b12-batch-size", type=int, default=1)
    p.add_argument("--b12-max-new-tokens", type=int, default=2048)
    p.add_argument("--b12-max-prompt-tokens", type=int, default=24576)
    p.add_argument("--b12-device-map", default="")
    p.set_defaults(func=generate)

    p = sub.add_parser("score")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--embedding-model", type=Path, default=Path("/home/pengguangyue/workspace/models/xlm-roberta-large"))
    p.add_argument("--device", default="")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--k", type=int, default=5)
    p.set_defaults(func=score)

    p = sub.add_parser("summarize")
    p.add_argument("--per-sample", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--k", type=int, default=5)
    p.set_defaults(func=summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
