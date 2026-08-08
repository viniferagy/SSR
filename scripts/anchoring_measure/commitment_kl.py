#!/usr/bin/env python3
"""Measure answer-conditioned continuation sensitivity along reasoning prefixes.

For a fixed observed reasoning prefix R[:t], this script compares the next-token
distribution with and without the final answer A in the prompt:

    KL_t = KL(P(. | Q, A, R[:t]) || P(. | Q, R[:t]))
    Gap_t = H(P(. | Q, R[:t])) - H(P(. | Q, A, R[:t]))

Large values mean that adding A still changes or sharpens the model's
continuation distribution at that prefix. The final A_traj metric uses the mean
entropy gap; KL and Jensen-Shannon divergence are retained as diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


METHODS = ["NEU", "SUP", "AUG-SUP", "SSR"]
FRACTIONS = [0.10, 0.25, 0.50, 0.75, 0.90]
SEQ_SEP = "\n\n"
TORCH_DTYPE = torch.bfloat16


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
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


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def split_reasoning(text: str) -> List[str]:
    parts = [p.strip() for p in text.strip().split(SEQ_SEP)]
    parts = [p for p in parts if p]
    if len(parts) > 1:
        return parts
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    numbered = [line for line in lines if re.match(r"^\s*(?:\d+[\.)]|[-*])\s+", line)]
    if len(numbered) > 1:
        return lines
    sentence_parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    sentence_parts = [p.strip() for p in sentence_parts if p.strip()]
    return sentence_parts if sentence_parts else parts


def apply_chat_prefix(tokenizer: Any, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tok_1d(tokenizer: Any, text: str, device: Optional[torch.device] = None) -> torch.LongTensor:
    return torch.tensor(tokenizer(text, add_special_tokens=False).input_ids, dtype=torch.long, device=device)


def tokenize_reasoning(tokenizer: Any, text: str, sep_ids: torch.LongTensor, device: torch.device) -> Tuple[int, torch.LongTensor, List[int]]:
    parts = split_reasoning(text)
    if not parts:
        return 0, torch.empty((0,), dtype=torch.long, device=device), []
    flat: List[int] = []
    ends: List[int] = []
    sep_list = sep_ids.tolist()
    for idx, part in enumerate(parts):
        if idx > 0:
            flat.extend(sep_list)
        flat.extend(tokenizer(part, add_special_tokens=False).input_ids)
        ends.append(len(flat))
    return len(parts), torch.tensor(flat, dtype=torch.long, device=device), ends


def prefix_end_indices(n_tokens: int, fractions: Sequence[float]) -> List[int]:
    out = []
    for frac in fractions:
        frac = min(max(float(frac), 0.0), 1.0)
        out.append(min(n_tokens, max(1, int(math.ceil(n_tokens * frac)))))
    return out


def rank_info() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        return rank, world, local
    return 0, 1, 0


def load_model_and_tokenizer(model_path: Path, device: torch.device) -> Tuple[Any, Any, Optional[int]]:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
        )
    model.to(device)
    model.eval()
    return model, tokenizer, getattr(model.config, "max_position_embeddings", None)


def answer_prompt_content(question: str, answer: str) -> str:
    return f"Question:\n{question}\n\nReference final answer:\n{answer}\n\nContinue the hidden reasoning."


def no_answer_prompt_content(question: str) -> str:
    return f"Question:\n{question}\n\nContinue the hidden reasoning."


@torch.no_grad()
def next_token_logprobs(model: Any, prompt_ids: torch.LongTensor, max_ctx: Optional[int]) -> torch.Tensor:
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    if max_ctx is not None and int(prompt_ids.numel()) > max_ctx:
        prompt_ids = prompt_ids[-int(max_ctx) :]
    input_ids = prompt_ids.unsqueeze(0)
    attn = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    del out, input_ids, attn
    return logp


def entropy_from_logp(logp: torch.Tensor) -> float:
    p = logp.exp()
    h = -torch.sum(p * logp)
    return float(h.item())


def kl_jsd_and_entropy_gap(logp_with_answer: torch.Tensor, logp_no_answer: torch.Tensor) -> Tuple[float, float, float, float, float]:
    p = logp_with_answer.exp()
    q = logp_no_answer.exp()
    kl = torch.sum(p * (logp_with_answer - logp_no_answer))
    m = 0.5 * (p + q)
    logm = torch.log(m.clamp_min(torch.finfo(m.dtype).tiny))
    jsd = 0.5 * torch.sum(p * (logp_with_answer - logm)) + 0.5 * torch.sum(q * (logp_no_answer - logm))
    h_with = entropy_from_logp(logp_with_answer)
    h_no = entropy_from_logp(logp_no_answer)
    return float(kl.item()), float(jsd.item()), h_no - h_with, h_no, h_with


def score(args: argparse.Namespace) -> None:
    rank, world, local = rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    all_rows = read_jsonl(args.input)
    start_index = int(getattr(args, "start_index", 0) or 0)
    end_index = int(args.limit) if args.limit is not None else len(all_rows)
    rows = all_rows[start_index:end_index]
    methods = [m for m in args.methods.split(",") if m]
    fractions = [float(x) for x in args.fractions.split(",") if x]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(args.scoring_model, trust_remote_code=True)
        sep_ids = tok_1d(tokenizer, SEQ_SEP, torch.device("cpu"))
        dry_rows = []
        for sample_idx, row in enumerate(rows[: min(len(rows), args.dry_run_rows)], start=start_index):
            for method in methods:
                reasoning = ensure_text(row.get("reasonings", {}).get(method))
                n_steps, flat_reason, step_ends = tokenize_reasoning(tokenizer, reasoning, sep_ids, torch.device("cpu"))
                dry_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "id": row.get("id", sample_idx),
                        "method": method,
                        "n_steps": n_steps,
                        "reasoning_tokens": int(flat_reason.numel()),
                        "prefix_fractions": fractions,
                        "prefix_token_ends": prefix_end_indices(int(flat_reason.numel()), fractions),
                        "step_boundaries": step_ends[:10],
                    }
                )
        out = args.output_dir / "commitment_dry_run.jsonl"
        write_jsonl(out, dry_rows)
        print(json.dumps({"dry_run": True, "rows": len(dry_rows), "output": str(out)}, indent=2))
        return

    model, tokenizer, max_ctx = load_model_and_tokenizer(args.scoring_model, device)
    sep_ids = tok_1d(tokenizer, SEQ_SEP, device)
    chunk_tag = f"_s{start_index}_e{end_index}" if start_index or args.limit is not None else ""
    metrics_path = args.output_dir / f"commitment_metrics_rank{rank}{chunk_tag}.jsonl"
    invalid_path = args.output_dir / f"commitment_invalid_rank{rank}{chunk_tag}.jsonl"
    valid = invalid = 0
    with metrics_path.open("w", encoding="utf-8") as fw, invalid_path.open("w", encoding="utf-8") as fi:
        iterator = list(enumerate(rows, start=start_index))[rank::world]
        for sample_idx, row in tqdm(iterator, desc=f"commit-rank{rank}", disable=(rank != 0)):
            for method in methods:
                try:
                    question = ensure_text(row.get("questions", {}).get(method))
                    answer = ensure_text(row.get("answers", {}).get(method))
                    reasoning = ensure_text(row.get("reasonings", {}).get(method))
                    if not (question.strip() and answer.strip() and reasoning.strip()):
                        continue

                    n_steps, flat_reason, _ = tokenize_reasoning(tokenizer, reasoning, sep_ids, device)
                    n_tokens = int(flat_reason.numel())
                    if n_tokens == 0:
                        raise ValueError("empty reasoning ids")

                    no_a_prefix = tok_1d(tokenizer, apply_chat_prefix(tokenizer, no_answer_prompt_content(question)), device)
                    with_a_prefix = tok_1d(tokenizer, apply_chat_prefix(tokenizer, answer_prompt_content(question, answer)), device)
                    token_ends = prefix_end_indices(n_tokens, fractions)
                    kl_vals: List[float] = []
                    jsd_vals: List[float] = []
                    conf_gap_vals: List[float] = []
                    h_no_vals: List[float] = []
                    h_with_vals: List[float] = []
                    for end in token_ends:
                        r_prefix = flat_reason[:end]
                        logp_no_a = next_token_logprobs(model, torch.cat([no_a_prefix, r_prefix], dim=0), max_ctx)
                        logp_with_a = next_token_logprobs(model, torch.cat([with_a_prefix, r_prefix], dim=0), max_ctx)
                        kl, jsd, conf_gap, h_no, h_with = kl_jsd_and_entropy_gap(logp_with_a, logp_no_a)
                        kl_vals.append(kl)
                        jsd_vals.append(jsd)
                        conf_gap_vals.append(conf_gap)
                        h_no_vals.append(h_no)
                        h_with_vals.append(h_with)
                        del logp_no_a, logp_with_a

                    rec: Dict[str, Any] = {
                        "sample_idx": sample_idx,
                        "id": row.get("id", sample_idx),
                        "method": method,
                        "n_steps": n_steps,
                        "reasoning_tokens": n_tokens,
                        "prefix_fractions": fractions,
                        "prefix_token_ends": token_ends,
                        "answer_control_kl": kl_vals,
                        "answer_control_jsd": jsd_vals,
                        "confidence_gap": conf_gap_vals,
                        "entropy_without_answer": h_no_vals,
                        "entropy_with_answer": h_with_vals,
                        "KL_mean": float(np.mean(kl_vals)),
                        "JSD_mean": float(np.mean(jsd_vals)),
                        "ConfidenceGap_mean": float(np.mean(conf_gap_vals)),
                        "Entropy_without_A_mean": float(np.mean(h_no_vals)),
                        "Entropy_with_A_mean": float(np.mean(h_with_vals)),
                    }
                    for frac, kl, jsd in zip(fractions, kl_vals, jsd_vals):
                        tag = int(round(frac * 100))
                        rec[f"KL_{tag:03d}"] = kl
                        rec[f"JSD_{tag:03d}"] = jsd
                    for frac, conf_gap, h_no, h_with in zip(fractions, conf_gap_vals, h_no_vals, h_with_vals):
                        tag = int(round(frac * 100))
                        rec[f"ConfidenceGap_{tag:03d}"] = conf_gap
                        rec[f"EntropyWithoutA_{tag:03d}"] = h_no
                        rec[f"EntropyWithA_{tag:03d}"] = h_with
                    if "KL_025" in rec and "KL_090" in rec:
                        rec["KL_drop_25_to_90"] = float(rec["KL_025"]) - float(rec["KL_090"])
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fw.flush()
                    valid += 1
                except Exception as exc:
                    fi.write(json.dumps({"sample_idx": sample_idx, "id": row.get("id", sample_idx), "method": method, "error": repr(exc)}, ensure_ascii=False) + "\n")
                    fi.flush()
                    invalid += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(json.dumps({"rank": rank, "valid": valid, "invalid": invalid, "metrics": str(metrics_path)}, indent=2))


def read_metric_rows(path: Path) -> List[Dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("commitment_metrics_rank*.jsonl"))
    rows: List[Dict[str, Any]] = []
    for file in files:
        rows.extend(read_jsonl(file))
    return rows


def percentile(values: List[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    pos = (len(values) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def bootstrap_ci(values: Sequence[float], rng: random.Random, n_boot: int) -> Tuple[float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], vals[0]
    boots = [float(np.mean([rng.choice(vals) for _ in vals])) for _ in range(n_boot)]
    return percentile(boots, 0.025), percentile(boots, 0.975)


def parse_delta_pairs(text: str | None, methods: Sequence[str]) -> List[Tuple[str, str]]:
    if text:
        pairs: List[Tuple[str, str]] = []
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            if "|" in item:
                method, base = item.split("|", 1)
            elif ":" in item:
                method, base = item.split(":", 1)
            else:
                raise ValueError(f"Delta pair must be METHOD|BASE or METHOD:BASE, got: {item}")
            pairs.append((method.strip(), base.strip()))
        return pairs
    if "NEU" in methods:
        return [(method, "NEU") for method in methods if method != "NEU"]
    if len(methods) >= 2:
        base = methods[0]
        return [(method, base) for method in methods[1:]]
    return []


def summarize(args: argparse.Namespace) -> None:
    rows = read_metric_rows(args.metrics)
    if args.limit is not None:
        rows = [row for row in rows if int(row.get("sample_idx", -1)) < args.limit]
    methods = [m for m in args.methods.split(",") if m]
    delta_pairs = parse_delta_pairs(args.delta_pairs, methods)
    fields = [
        "KL_010", "KL_025", "KL_050", "KL_075", "KL_090", "KL_mean", "KL_drop_25_to_90", "JSD_mean",
        "ConfidenceGap_010", "ConfidenceGap_025", "ConfidenceGap_050", "ConfidenceGap_075", "ConfidenceGap_090",
        "ConfidenceGap_mean", "Entropy_without_A_mean", "Entropy_with_A_mean",
    ]
    fields = [field for field in fields if any(field in row for row in rows)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    means_path = args.output_dir / "commitment_means.csv"
    with means_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "method", "n", "mean"])
        writer.writeheader()
        for field in fields:
            for method in methods:
                vals = [float(row[field]) for row in rows if row.get("method") == method and field in row and math.isfinite(float(row[field]))]
                writer.writerow({"metric": field, "method": method, "n": len(vals), "mean": float(np.mean(vals)) if vals else float("nan")})

    by_sample: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_sample[int(row["sample_idx"])][str(row["method"])] = row

    rng = random.Random(args.seed)
    deltas_path = args.output_dir / "commitment_deltas.csv"
    with deltas_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "delta", "n", "mean_delta", "ci95_low", "ci95_high", "ci_excludes_zero"])
        writer.writeheader()
        for field in fields:
            for method, base in delta_pairs:
                vals = []
                for sample_methods in by_sample.values():
                    if method not in sample_methods or base not in sample_methods:
                        continue
                    a = sample_methods[method].get(field)
                    b = sample_methods[base].get(field)
                    if a is None or b is None:
                        continue
                    vals.append(float(a) - float(b))
                lo, hi = bootstrap_ci(vals, rng, args.bootstrap)
                writer.writerow(
                    {
                        "metric": field,
                        "delta": f"{method}-{base}",
                        "n": len(vals),
                        "mean_delta": float(np.mean(vals)) if vals else float("nan"),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "ci_excludes_zero": bool(vals) and (lo > 0 or hi < 0),
                    }
                )

    means = list(csv.DictReader(means_path.open(encoding="utf-8")))
    deltas = list(csv.DictReader(deltas_path.open(encoding="utf-8")))
    lines = [
        "# Commitment KL",
        "",
        "KL compares next-token distributions with vs without the final answer in context.",
        "Higher KL means the continuation remains more answer-controlled at that prefix.",
        "ConfidenceGap is H(P(.|Q,R_prefix)) - H(P(.|Q,A,R_prefix)); positive values mean the answer reduces next-token uncertainty.",
        "",
        "## Means by method",
        "",
        "| metric | " + " | ".join(methods) + " |",
        "| --- | " + " | ".join(["---:"] * len(methods)) + " |",
    ]
    for field in fields:
        vals = {}
        for method in methods:
            row = next((r for r in means if r["metric"] == field and r["method"] == method), None)
            vals[method] = float(row["mean"]) if row else float("nan")
        lines.append("| " + field + " | " + " | ".join(f"{vals.get(method, float('nan')):.6f}" for method in methods) + " |")
    lines.extend(["", "## Paired deltas", "", "| metric | delta | mean | 95% CI | excludes 0 |", "| --- | --- | ---: | --- | --- |"])
    for row in deltas:
        lines.append(f"| {row['metric']} | {row['delta']} | {float(row['mean_delta']):.6f} | [{float(row['ci95_low']):.6f}, {float(row['ci95_high']):.6f}] | {row['ci_excludes_zero']} |")
    (args.output_dir / "commitment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "means": str(means_path), "deltas": str(deltas_path), "report": str(args.output_dir / "commitment_report.md")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_score = sub.add_parser("score")
    p_score.add_argument("--input", type=Path, required=True)
    p_score.add_argument("--output-dir", type=Path, required=True)
    p_score.add_argument("--scoring-model", type=Path, required=True)
    p_score.add_argument("--methods", default=",".join(METHODS))
    p_score.add_argument("--fractions", default=",".join(str(x) for x in FRACTIONS))
    p_score.add_argument("--start-index", type=int, default=0, help="Start row offset for resumable chunk scoring; sample_idx keeps this global offset.")
    p_score.add_argument("--limit", type=int)
    p_score.add_argument("--dry-run", action="store_true")
    p_score.add_argument("--dry-run-rows", type=int, default=2)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--metrics", type=Path, required=True)
    p_sum.add_argument("--output-dir", type=Path, required=True)
    p_sum.add_argument("--methods", default=",".join(METHODS))
    p_sum.add_argument("--delta-pairs", help="Comma-separated METHOD|BASE pairs. Defaults to each method vs NEU, or vs the first method if NEU is absent.")
    p_sum.add_argument("--limit", type=int)
    p_sum.add_argument("--bootstrap", type=int, default=2000)
    p_sum.add_argument("--seed", type=int, default=13)

    args = parser.parse_args()
    if args.cmd == "score":
        score(args)
    elif args.cmd == "summarize":
        summarize(args)


if __name__ == "__main__":
    main()
