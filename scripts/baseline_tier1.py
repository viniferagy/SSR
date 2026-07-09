#!/usr/bin/env python3
"""Generate and prepare Tier-1 reviewer baselines for the SSR pipeline.

This script uses the repository's metric-format JSONL:

{
  "id": ...,
  "questions": {"NEU": ...},
  "answers": {"NEU": ...},
  "contexts": {"NEU": ...},
  "reasonings": {"NEU": ...}
}

The generation subcommand writes metric-format rows so the outputs can be scored
by ``scripts/anchoring_measure/reviewer_protocol.py`` with the same excess
metric protocol as the main paper runs.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import string
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "scripts" / "rcot_generation"
if str(PROMPT_DIR) not in sys.path:
    sys.path.insert(0, str(PROMPT_DIR))

from rcot_prompt import NEU_PROMPT  # noqa: E402


DEFAULT_BASE_METHOD = "NEU"
DEFAULT_GENERATE_METHODS = ["A1-FDB", "A6-Gist", "B11-NGramBlock", "B13-BoN"]

REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.IGNORECASE | re.DOTALL)
REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
BRIDGE_RE = re.compile(r"<bridge>(.*?)</bridge>", re.IGNORECASE | re.DOTALL)
GIST_RE = re.compile(r"<gist>(.*?)</gist>", re.IGNORECASE | re.DOTALL)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


FDB_P1_SYSTEM = (
    "You are an expert problem solver. Solve the user's question from scratch. "
    "Write the reasoning between <reasoning> and </reasoning>, then write the "
    "draft answer between <answer> and </answer>. Do not use or assume any "
    "reference answer."
)

FDB_P2_SYSTEM = (
    "You previously wrote draft reasoning for the question. A reference answer "
    "is now provided. Write only a short closing segment that continues the "
    "draft reasoning and naturally reaches a conclusion consistent with the "
    "reference answer. Do not rewrite the draft. Do not mention that a reference "
    "answer exists. Output only <bridge>...</bridge>."
)

FDB_P2_USER = """[QUESTION]
{question}

[REFERENCE ANSWER]
{answer}

[DRAFT REASONING]
{draft}"""

GIST_P1_SYSTEM = (
    "Given a question and its reference answer, extract 3-8 methodological "
    "gists needed to solve it. Each gist should describe what to establish, "
    "compare, derive, or verify. Do not use functional tags. Do not state "
    "specific final conclusions, concrete values, or verbatim phrases from the "
    "reference answer. Output only <gist>...</gist>."
)

GIST_P2_SYSTEM = (
    "Write the complete reasoning chain for the question by developing each "
    "gist in order. The chain should read as a self-contained derivation from "
    "the question. Do not cite the gists explicitly and do not copy the "
    "reference answer. Write substantive paragraphs, not placeholders. Never "
    "output ellipses such as '...' or meta-text saying the reasoning is omitted. "
    "Output only one <reasoning> block containing the full reasoning text."
)

GIST_P2_USER = """[QUESTION]
{question}

[REFERENCE ANSWER]
{answer}

[GISTS]
{gists}"""

ENDPOINT_SYSTEM = (
    "You are a strict evaluator of abstract endpoint alignment. Given a question, "
    "a reference answer, and a candidate reasoning trace, decide whether the trace "
    "is aimed at the same abstract endpoint as the reference answer. The abstract "
    "endpoint means the same user request, response act, stance, and target "
    "deliverable or artifact type. Do not require the reasoning trace to include "
    "the reference answer's specific wording, examples, bullet points, code, "
    "numbers, entities, or detailed final-answer content; missing those details is "
    "not an error by itself. Mark inconsistent only when the trace addresses a "
    "different user request or conversation turn, takes an opposite or incompatible "
    "stance, targets a different deliverable, makes an incompatible safety/refusal "
    "decision, or is so generic that no compatible endpoint can be identified. "
    "Score 5 for clearly aligned abstract endpoint, 3 for ambiguous but plausible "
    "alignment, and 1 for a different or incompatible endpoint. Output only JSON: "
    "{\"endpoint_consistent\": true|false, \"score\": 1-5, \"reason\": \"one sentence\"}."
)

ENDPOINT_USER = """[QUESTION]
{question}

[REFERENCE ANSWER]
{answer}

[CANDIDATE REASONING]
{trace}"""

STOPWORDS = set(
    """a an the of to in on at for with by from as and or but if then than that this these those
    it its is are was were be been being do does did not no so such can could may might will would
    shall should must have has had i you he she we they them his her their our your my me him us
    what which who whom how when where why there here also into over under about above below between
    during after before all any both each few more most other some only own same very just""".split()
)
PUNCT = set(string.punctuation)


@dataclass(frozen=True)
class PromptRequest:
    row_idx: int
    key: str
    prompt: str
    sampling_kind: str = "default"
    bad_words: Optional[List[str]] = None
    n: int = 1


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
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def row_value(row: Dict[str, Any], section: str, method: str) -> str:
    return ensure_text(row.get(section, {}).get(method))


def row_question(row: Dict[str, Any], method: str) -> str:
    return row_value(row, "questions", method)


def row_answer(row: Dict[str, Any], method: str) -> str:
    return row_value(row, "answers", method)


def row_context(row: Dict[str, Any], method: str) -> str:
    context = row_value(row, "contexts", method)
    if context.strip():
        return context
    q = row_question(row, method)
    a = row_answer(row, method)
    return f"User: {q}\n\nAssistant: {a}".strip() if a else q


def method_list(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def apply_chat_template(tokenizer: Any, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_tag(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""


def parse_reason(raw: str) -> str:
    raw = ensure_text(raw).strip()
    for pattern in (REASON_RE, REASONING_RE):
        out = extract_tag(raw, pattern)
        if out:
            return out
    lower = raw.lower()
    for tag in ("<reason>", "<reasoning>"):
        if tag in lower:
            return raw[lower.index(tag) + len(tag):].strip()
    return raw


def parse_answer(raw: str) -> str:
    return extract_tag(raw, ANSWER_RE)


def parse_bridge(raw: str) -> str:
    return extract_tag(raw, BRIDGE_RE) or ensure_text(raw).strip()


def parse_gist(raw: str) -> str:
    raw = ensure_text(raw).strip()
    matches = [m.strip() for m in GIST_RE.findall(raw) if m.strip()]
    if matches:
        return "\n".join(f"{i + 1}. {m}" for i, m in enumerate(matches))
    return raw


def looks_like_placeholder(text: str) -> bool:
    stripped = ensure_text(text).strip()
    normalized = stripped.strip(".。… \n\t")
    return not normalized or stripped in {"...", "…"} or len(normalized) < 20


def load_raw(path: Path) -> Dict[Tuple[int, str], List[Dict[str, Any]]]:
    raw: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return raw
    for row in read_jsonl(path):
        raw[(int(row["row_idx"]), str(row["key"]))].append(row)
    return raw


def first_raw(raw: Dict[Tuple[int, str], List[Dict[str, Any]]], row_idx: int, key: str) -> Optional[Dict[str, Any]]:
    vals = raw.get((row_idx, key), [])
    return vals[0] if vals else None


def all_raw(raw: Dict[Tuple[int, str], List[Dict[str, Any]]], row_idx: int, key: str) -> List[Dict[str, Any]]:
    return raw.get((row_idx, key), [])


def raw_candidate_count(raw: Dict[Tuple[int, str], List[Dict[str, Any]]], row_idx: int, key: str) -> int:
    return len({int(row.get("candidate_idx", 0)) for row in all_raw(raw, row_idx, key)})


def is_content_ngram(text: str, max_stop_frac: float = 2 / 3) -> bool:
    words = [w.strip("".join(PUNCT)).lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return False
    n_stop = sum(1 for w in words if w in STOPWORDS or not any(c.isalnum() for c in w))
    return n_stop / len(words) < max_stop_frac


def banned_ngram_strings(tokenizer: Any, answer: str, ns: Sequence[int] = (3, 4), limit: int = 2048) -> List[str]:
    ids = tokenizer.encode(answer, add_special_tokens=False)
    out: List[str] = []
    seen = set()
    for n in ns:
        for i in range(0, max(0, len(ids) - n + 1)):
            text = tokenizer.decode(ids[i:i + n], skip_special_tokens=False).strip()
            if not text or text in seen or not is_content_ngram(text):
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                return out
    return out


def sampling_params(cls: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(cls)
    usable = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    return cls(**usable)


def generate_requests(
    llm: Any,
    sampling_cls: Any,
    requests: List[PromptRequest],
    raw_output: Path,
    raw: Dict[Tuple[int, str], List[Dict[str, Any]]],
    args: argparse.Namespace,
) -> None:
    pending = [r for r in requests if len(raw.get((r.row_idx, r.key), [])) < r.n]
    print(json.dumps({"requests": len(requests), "completed": len(requests) - len(pending), "pending": len(pending)}, indent=2), flush=True)
    for start in range(0, len(pending), args.chunk_size):
        chunk = pending[start:start + args.chunk_size]
        if not chunk:
            continue
        prompts = [r.prompt for r in chunk]
        if any(r.bad_words for r in chunk):
            sps = [
                sampling_params(
                    sampling_cls,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    bad_words=r.bad_words,
                    n=r.n,
                )
                for r in chunk
            ]
            outputs = llm.generate(prompts, sps)
        else:
            sampling = sampling_params(
                sampling_cls,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                n=max(r.n for r in chunk),
            )
            outputs = llm.generate(prompts, sampling)
        rows = []
        for req, out in zip(chunk, outputs):
            cand_offset = raw_candidate_count(raw, req.row_idx, req.key)
            for cand_idx, cand in enumerate(out.outputs or []):
                rows.append(
                    {
                        "row_idx": req.row_idx,
                        "key": req.key,
                        "candidate_idx": cand_offset + cand_idx,
                        "raw_output": cand.text,
                        "finish_reason": getattr(cand, "finish_reason", None),
                        "output_tokens": len(getattr(cand, "token_ids", []) or []),
                        "prompt_chars": len(req.prompt),
                        "bad_words": len(req.bad_words or []),
                    }
                )
        append_jsonl(raw_output, rows)
        for row in rows:
            raw[(int(row["row_idx"]), str(row["key"]))].append(row)
        print(json.dumps({"chunk_start": start, "chunk_size": len(chunk), "completed": sum(len(v) for v in raw.values())}, ensure_ascii=False), flush=True)


def build_a1_requests(rows: List[Dict[str, Any]], tokenizer: Any, base_method: str, raw: Dict[Tuple[int, str], List[Dict[str, Any]]]) -> Tuple[List[PromptRequest], List[PromptRequest]]:
    p1: List[PromptRequest] = []
    p2: List[PromptRequest] = []
    for i, row in enumerate(rows):
        q = row_question(row, base_method)
        if not q.strip():
            continue
        if not first_raw(raw, i, "A1-FDB:p1"):
            p1.append(PromptRequest(i, "A1-FDB:p1", apply_chat_template(tokenizer, FDB_P1_SYSTEM, q)))
        p1_raw = first_raw(raw, i, "A1-FDB:p1")
        if p1_raw and not first_raw(raw, i, "A1-FDB:p2"):
            draft = parse_reason(p1_raw["raw_output"])
            user = FDB_P2_USER.format(question=q, answer=row_answer(row, base_method), draft=draft)
            p2.append(PromptRequest(i, "A1-FDB:p2", apply_chat_template(tokenizer, FDB_P2_SYSTEM, user)))
    return p1, p2


def build_a6_requests(rows: List[Dict[str, Any]], tokenizer: Any, base_method: str, raw: Dict[Tuple[int, str], List[Dict[str, Any]]]) -> Tuple[List[PromptRequest], List[PromptRequest]]:
    p1: List[PromptRequest] = []
    p2: List[PromptRequest] = []
    for i, row in enumerate(rows):
        q = row_question(row, base_method)
        a = row_answer(row, base_method)
        if not (q.strip() and a.strip()):
            continue
        if not first_raw(raw, i, "A6-Gist:p1"):
            user = f"[QUESTION]\n{q}\n\n[REFERENCE ANSWER]\n{a}"
            p1.append(PromptRequest(i, "A6-Gist:p1", apply_chat_template(tokenizer, GIST_P1_SYSTEM, user)))
        p1_raw = first_raw(raw, i, "A6-Gist:p1")
        if p1_raw and not first_raw(raw, i, "A6-Gist:p2"):
            gists = parse_gist(p1_raw["raw_output"])
            user = GIST_P2_USER.format(question=q, answer=a, gists=gists)
            p2.append(PromptRequest(i, "A6-Gist:p2", apply_chat_template(tokenizer, GIST_P2_SYSTEM, user)))
    return p1, p2


def build_b11_requests(rows: List[Dict[str, Any]], tokenizer: Any, base_method: str, raw: Dict[Tuple[int, str], List[Dict[str, Any]]], args: argparse.Namespace) -> List[PromptRequest]:
    out: List[PromptRequest] = []
    for i, row in enumerate(rows):
        if first_raw(raw, i, "B11-NGramBlock"):
            continue
        context = row_context(row, base_method)
        answer = row_answer(row, base_method)
        if not (context.strip() and answer.strip()):
            continue
        bad_words = banned_ngram_strings(tokenizer, answer, limit=args.bad_words_limit)
        out.append(PromptRequest(i, "B11-NGramBlock", apply_chat_template(tokenizer, NEU_PROMPT, context), bad_words=bad_words))
    return out


def build_b13_requests(rows: List[Dict[str, Any]], tokenizer: Any, base_method: str, raw: Dict[Tuple[int, str], List[Dict[str, Any]]], args: argparse.Namespace) -> List[PromptRequest]:
    out: List[PromptRequest] = []
    for i, row in enumerate(rows):
        if raw_candidate_count(raw, i, "B13-BoN") >= args.bon_n:
            continue
        context = row_context(row, base_method)
        if context.strip():
            out.append(PromptRequest(i, "B13-BoN", apply_chat_template(tokenizer, NEU_PROMPT, context), n=args.bon_n))
    return out


def build_metric_rows(rows: List[Dict[str, Any]], raw: Dict[Tuple[int, str], List[Dict[str, Any]]], base_method: str, methods: Sequence[str], bon_n: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    out_rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        q = row_question(row, base_method)
        a = row_answer(row, base_method)
        ctx = row_context(row, base_method)
        reasonings: Dict[str, str] = {}
        meta: Dict[str, Any] = {}
        missing: List[str] = []
        if "A1-FDB" in methods:
            p1 = first_raw(raw, i, "A1-FDB:p1")
            p2 = first_raw(raw, i, "A1-FDB:p2")
            if p1 and p2:
                draft = parse_reason(p1["raw_output"])
                bridge = parse_bridge(p2["raw_output"])
                reasonings["A1-FDB"] = (draft + "\n\n" + bridge).strip()
                meta["A1-FDB"] = {"draft_answer": parse_answer(p1["raw_output"]), "draft_chars": len(draft), "bridge_chars": len(bridge)}
            else:
                missing.append("A1-FDB")
        if "A6-Gist" in methods:
            p1 = first_raw(raw, i, "A6-Gist:p1")
            p2 = first_raw(raw, i, "A6-Gist:p2")
            if p1 and p2:
                reasoning = parse_reason(p2["raw_output"])
                if looks_like_placeholder(reasoning):
                    missing.append("A6-Gist")
                else:
                    reasonings["A6-Gist"] = reasoning
                meta["A6-Gist"] = {"gist": parse_gist(p1["raw_output"])}
            else:
                missing.append("A6-Gist")
        if "B11-NGramBlock" in methods:
            r = first_raw(raw, i, "B11-NGramBlock")
            if r:
                reasonings["B11-NGramBlock"] = parse_reason(r["raw_output"])
                meta["B11-NGramBlock"] = {"bad_words": r.get("bad_words")}
            else:
                missing.append("B11-NGramBlock")
        if "B13-BoN" in methods:
            cands = all_raw(raw, i, "B13-BoN")
            unique_cands = {}
            for cand in sorted(cands, key=lambda x: int(x.get("candidate_idx", 0))):
                unique_cands.setdefault(int(cand.get("candidate_idx", 0)), cand)
            if len(unique_cands) >= bon_n:
                for cand_idx, cand in list(unique_cands.items())[:bon_n]:
                    name = f"B13-BoN-c{cand_idx}"
                    reasonings[name] = parse_reason(cand["raw_output"])
            else:
                missing.append("B13-BoN")
        if missing or any(not v.strip() for v in reasonings.values()):
            rejected.append({"row_idx": i, "id": row.get("id", i), "missing": missing})
            continue
        output_methods = list(reasonings)
        out_rows.append(
            {
                "id": row.get("id", i),
                "questions": {m: q for m in output_methods},
                "answers": {m: a for m in output_methods},
                "contexts": {m: ctx for m in output_methods},
                "reasonings": reasonings,
                "baseline_metadata": meta,
            }
        )
    return out_rows, rejected


def cmd_generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows_all = read_jsonl(args.input)
    rows = rows_all[:args.limit] if args.limit else rows_all
    methods = method_list(args.methods) or DEFAULT_GENERATE_METHODS
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    raw = load_raw(args.raw_output)
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        seed=args.seed,
    )

    if "A1-FDB" in methods:
        p1, _p2 = build_a1_requests(rows, tokenizer, args.base_method, raw)
        generate_requests(llm, SamplingParams, p1, args.raw_output, raw, args)
        _p1, p2 = build_a1_requests(rows, tokenizer, args.base_method, raw)
        generate_requests(llm, SamplingParams, p2, args.raw_output, raw, args)
    if "A6-Gist" in methods:
        p1, _p2 = build_a6_requests(rows, tokenizer, args.base_method, raw)
        old_max = args.max_tokens
        args.max_tokens = min(args.max_tokens, args.gist_max_tokens)
        generate_requests(llm, SamplingParams, p1, args.raw_output, raw, args)
        args.max_tokens = old_max
        _p1, p2 = build_a6_requests(rows, tokenizer, args.base_method, raw)
        generate_requests(llm, SamplingParams, p2, args.raw_output, raw, args)
    if "B11-NGramBlock" in methods:
        generate_requests(llm, SamplingParams, build_b11_requests(rows, tokenizer, args.base_method, raw, args), args.raw_output, raw, args)
    if "B13-BoN" in methods:
        generate_requests(llm, SamplingParams, build_b13_requests(rows, tokenizer, args.base_method, raw, args), args.raw_output, raw, args)

    final_raw = load_raw(args.raw_output)
    metric_rows, rejected = build_metric_rows(rows, final_raw, args.base_method, methods, args.bon_n)
    write_jsonl(args.output, metric_rows)
    rejected_output = args.rejected_output or args.output.with_suffix(".rejected.jsonl")
    summary_output = args.summary_output or args.output.with_suffix(".summary.json")
    write_jsonl(rejected_output, rejected)
    summary = {
        "mode": "tier1-generate",
        "input": str(args.input),
        "output": str(args.output),
        "raw_output": str(args.raw_output),
        "methods": methods,
        "input_rows": len(rows),
        "complete_rows": len(metric_rows),
        "rejected_rows": len(rejected),
        "bon_n": args.bon_n,
    }
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def read_metric_records(path_or_dir: Path) -> List[Dict[str, Any]]:
    files = [path_or_dir] if path_or_dir.is_file() else sorted(path_or_dir.glob("metrics_rank*.jsonl"))
    out: List[Dict[str, Any]] = []
    for path in files:
        out.extend(read_jsonl(path))
    return out


def cmd_select_b13(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    records = read_metric_records(args.metrics)
    by_sample: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        method = str(rec.get("method", ""))
        if method.startswith("B13-BoN-c"):
            by_sample[int(rec["sample_idx"])][method] = rec
    selected_rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for sample_idx, row in enumerate(rows):
        cand_metrics = by_sample.get(sample_idx, {})
        candidates = []
        candidate_methods = sorted(
            [m for m in row.get("reasonings", {}) if str(m).startswith("B13-BoN-c")],
            key=lambda m: int(str(m).rsplit("c", 1)[1]),
        )
        for method in candidate_methods:
            rec = cand_metrics.get(method)
            if rec is None:
                continue
            reasoning = ensure_text(row.get("reasonings", {}).get(method))
            if not reasoning.strip():
                continue
            candidates.append((method, rec, reasoning))
        if not candidates:
            fallback = next((m for m in candidate_methods if ensure_text(row.get("reasonings", {}).get(m)).strip()), "")
            if not fallback:
                rejected.append({"sample_idx": sample_idx, "reason": "missing_candidate_reasonings"})
                continue
            method = fallback
            rec = None
            reasoning = ensure_text(row.get("reasonings", {}).get(method))
            selection_meta = {"selected_candidate": method, "selection_rule": "fallback_unscored"}
        elif args.rule == "prob":
            method, rec, reasoning = min(candidates, key=lambda x: float(x[1]["B"]))
            selection_meta = {"selected_candidate": method, "selection_rule": args.rule}
        else:
            probs = [float(x[1]["B"]) for x in candidates]
            ents = [float(x[1]["Aent_raw"]) for x in candidates]
            p_lo, p_hi = min(probs), max(probs)
            e_lo, e_hi = min(ents), max(ents)

            def mm(v: float, lo: float, hi: float) -> float:
                return 0.5 if math.isclose(lo, hi) else (v - lo) / (hi - lo)

            method, rec, reasoning = min(
                candidates,
                key=lambda x: 0.5 * mm(float(x[1]["B"]), p_lo, p_hi) + 0.5 * mm(float(x[1]["Aent_raw"]), e_lo, e_hi),
            )
            selection_meta = {"selected_candidate": method, "selection_rule": args.rule}
        q = ensure_text(row.get("questions", {}).get(method))
        a = ensure_text(row.get("answers", {}).get(method))
        c = ensure_text(row.get("contexts", {}).get(method))
        selected_rows.append(
            {
                "id": row.get("id", sample_idx),
                "questions": {"B13-BoN": q},
                "answers": {"B13-BoN": a},
                "contexts": {"B13-BoN": c},
                "reasonings": {"B13-BoN": reasoning},
                "baseline_metadata": {"B13-BoN": selection_meta},
            }
        )
    write_jsonl(args.output, selected_rows)
    rejected_output = args.rejected_output or args.output.with_suffix(".rejected.jsonl")
    write_jsonl(rejected_output, rejected)
    print(json.dumps({"selected": len(selected_rows), "rejected": len(rejected), "output": str(args.output)}, indent=2), flush=True)


def cmd_filter_methods(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    methods = method_list(args.methods)
    out: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        present = [m for m in methods if ensure_text(row.get("reasonings", {}).get(m)).strip()]
        missing = [m for m in methods if m not in present]
        if missing and args.strict:
            rejected.append({"row_idx": idx, "id": row.get("id", idx), "missing": missing})
            continue
        item = {"id": row.get("id", idx), "questions": {}, "answers": {}, "contexts": {}, "reasonings": {}}
        if "baseline_metadata" in row:
            item["baseline_metadata"] = {}
        for method in present:
            for key in ("questions", "answers", "contexts", "reasonings"):
                item[key][method] = ensure_text(row.get(key, {}).get(method))
            if "baseline_metadata" in row and method in row.get("baseline_metadata", {}):
                item["baseline_metadata"][method] = row["baseline_metadata"][method]
        out.append(item)
    write_jsonl(args.output, out)
    rejected_output = args.rejected_output or args.output.with_suffix(".rejected.jsonl")
    write_jsonl(rejected_output, rejected)
    print(json.dumps({"rows": len(out), "rejected": len(rejected), "methods": methods, "output": str(args.output)}, indent=2), flush=True)


def cmd_merge(args: argparse.Namespace) -> None:
    inputs = [Path(p) for p in args.inputs]
    row_sets = [read_jsonl(path) for path in inputs]
    if not row_sets:
        raise SystemExit("no inputs")
    n = len(row_sets[0])
    if any(len(rows) != n for rows in row_sets):
        raise SystemExit("all inputs must have the same row count and order")
    out = []
    rejected = []
    for i in range(n):
        merged = {"id": row_sets[0][i].get("id", i), "questions": {}, "answers": {}, "contexts": {}, "reasonings": {}, "baseline_metadata": {}}
        ok = True
        for rows in row_sets:
            row = rows[i]
            if row.get("id") != merged["id"]:
                ok = False
                break
            for key in ("questions", "answers", "contexts", "reasonings"):
                merged[key].update(row.get(key, {}))
            merged["baseline_metadata"].update(row.get("baseline_metadata", {}))
        if ok:
            out.append(merged)
        else:
            rejected.append({"row_idx": i, "reason": "id_mismatch"})
    write_jsonl(args.output, out)
    rejected_output = args.rejected_output or args.output.with_suffix(".rejected.jsonl")
    write_jsonl(rejected_output, rejected)
    print(json.dumps({"rows": len(out), "rejected": len(rejected), "output": str(args.output)}, indent=2), flush=True)


def parse_judge_json(text: str) -> Optional[Dict[str, Any]]:
    match = JSON_RE.search(text or "")
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def build_endpoint_prompts(
    rows: List[Dict[str, Any]],
    methods: Sequence[str],
    tokenizer: Any,
    args: argparse.Namespace,
) -> List[Tuple[int, str, str]]:
    prompts: List[Tuple[int, str, str]] = []
    for i, row in enumerate(rows[:args.limit] if args.limit else rows):
        for method in methods:
            q = ensure_text(row.get("questions", {}).get(method))
            a = ensure_text(row.get("answers", {}).get(method))
            r = ensure_text(row.get("reasonings", {}).get(method))
            if q.strip() and a.strip() and r.strip():
                user = ENDPOINT_USER.format(question=q, answer=a, trace=r)
                prompts.append((i, method, apply_chat_template(tokenizer, ENDPOINT_SYSTEM, user)))
    return prompts


def write_endpoint_outputs(args: argparse.Namespace, methods: Sequence[str], out_rows: List[Dict[str, Any]]) -> None:
    write_jsonl(args.output, out_rows)
    summary = []
    for method in methods:
        vals = [r for r in out_rows if r["method"] == method]
        valid = [r for r in vals if isinstance(r.get("endpoint_consistent"), bool)]
        summary.append(
            {
                "method": method,
                "n": len(vals),
                "valid": len(valid),
                "consistent_rate": (sum(1 for r in valid if r["endpoint_consistent"]) / len(valid)) if valid else None,
                "mean_score": (sum(float(r.get("score") or 0) for r in valid) / len(valid)) if valid else None,
            }
        )
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def cmd_endpoint_vllm(args: argparse.Namespace, methods: Sequence[str], prompts: List[Tuple[int, str, str]]) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.judge_model),
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        seed=args.seed,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
    out_rows: List[Dict[str, Any]] = []
    for start in range(0, len(prompts), args.chunk_size):
        chunk = prompts[start:start + args.chunk_size]
        outs = llm.generate([p for _, _, p in chunk], sampling)
        for (sample_idx, method, _prompt), out in zip(chunk, outs):
            text = out.outputs[0].text if out.outputs else ""
            parsed = parse_judge_json(text) or {}
            out_rows.append(
                {
                    "sample_idx": sample_idx,
                    "method": method,
                    "endpoint_consistent": parsed.get("endpoint_consistent"),
                    "score": parsed.get("score"),
                    "reason": parsed.get("reason"),
                    "raw_output": text,
                }
            )
        print(json.dumps({"chunk_start": start, "completed": len(out_rows), "total": len(prompts)}, indent=2), flush=True)
    write_endpoint_outputs(args, methods, out_rows)


def cmd_endpoint_hf(args: argparse.Namespace, methods: Sequence[str], prompts: List[Tuple[int, str, str]], tokenizer: Any) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map = args.device_map if args.device_map else None
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    ).eval()
    if device_map is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = next(iter(model.parameters())).device
    out_rows: List[Dict[str, Any]] = []
    for start in range(0, len(prompts), args.chunk_size):
        chunk = prompts[start:start + args.chunk_size]
        texts = [p for _, _, p in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_model_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=args.max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = gen[:, enc["input_ids"].shape[1]:]
        outputs = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        for (sample_idx, method, _prompt), text in zip(chunk, outputs):
            parsed = parse_judge_json(text) or {}
            out_rows.append(
                {
                    "sample_idx": sample_idx,
                    "method": method,
                    "endpoint_consistent": parsed.get("endpoint_consistent"),
                    "score": parsed.get("score"),
                    "reason": parsed.get("reason"),
                    "raw_output": text,
                }
            )
        print(json.dumps({"chunk_start": start, "completed": len(out_rows), "total": len(prompts)}, indent=2), flush=True)
    write_endpoint_outputs(args, methods, out_rows)


def cmd_endpoint(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    rows = read_jsonl(args.input)
    methods = method_list(args.methods)
    if not methods and rows:
        methods = list(rows[0].get("reasonings", {}).keys())
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=args.trust_remote_code)
    prompts = build_endpoint_prompts(rows, methods, tokenizer, args)
    if args.endpoint_engine == "hf":
        cmd_endpoint_hf(args, methods, prompts, tokenizer)
    else:
        cmd_endpoint_vllm(args, methods, prompts)


def add_vllm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, default=Path("/home/pengguangyue/workspace/models/Qwen/Qwen3-8B"))
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--raw-output", type=Path, required=True)
    p.add_argument("--rejected-output", type=Path, default=None)
    p.add_argument("--summary-output", type=Path, default=None)
    p.add_argument("--methods", default=",".join(DEFAULT_GENERATE_METHODS))
    p.add_argument("--base-method", default=DEFAULT_BASE_METHOD)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--bon-n", type=int, default=8)
    p.add_argument("--bad-words-limit", type=int, default=2048)
    p.add_argument("--gist-max-tokens", type=int, default=512)
    add_vllm_args(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("select-b13")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rejected-output", type=Path, default=None)
    p.add_argument("--rule", choices=["combo", "prob"], default="combo")
    p.set_defaults(func=cmd_select_b13)

    p = sub.add_parser("filter-methods")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rejected-output", type=Path, default=None)
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=cmd_filter_methods)

    p = sub.add_parser("merge")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rejected-output", type=Path, default=None)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("endpoint")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary-output", type=Path, default=None)
    p.add_argument("--methods", default="")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--judge-model", type=Path, default=Path("/home/pengguangyue/workspace/models/Qwen/Qwen3-8B"))
    p.add_argument("--endpoint-engine", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--device-map", default="", help="HF endpoint only: optional device_map, e.g. auto for multi-GPU.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--disable-custom-all-reduce", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.set_defaults(func=cmd_endpoint)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
