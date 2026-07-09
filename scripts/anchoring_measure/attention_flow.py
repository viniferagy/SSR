#!/usr/bin/env python3
"""Measure attention flow to answer tokens along reasoning prefixes.

This implements a small pilot from `insights/metrics_framework.md`.
For each reasoning prefix, we measure the mean attention ratio from the
generated reasoning tokens to answer tokens versus question tokens.
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


def load_model_and_tokenizer(
    model_path: Path,
    device: torch.device,
    attn_implementation: str,
) -> Tuple[Any, Any, Optional[int]]:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=cfg,
            trust_remote_code=True,
            dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=cfg,
            trust_remote_code=True,
            dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
        )
    model.to(device)
    model.eval()
    return model, tokenizer, getattr(model.config, "max_position_embeddings", None)


def q_only_prompt(question: str) -> str:
    return f"Question:\n{question}\n\nContinue the hidden reasoning."


@torch.no_grad()
def attentions_for_prompt(model: Any, prompt_ids: torch.LongTensor, max_ctx: Optional[int]) -> List[torch.Tensor]:
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    if max_ctx is not None and int(prompt_ids.numel()) > max_ctx:
        prompt_ids = prompt_ids[-int(max_ctx) :]
    inp = prompt_ids.unsqueeze(0)
    attn_mask = torch.ones_like(inp)
    out = model(input_ids=inp, attention_mask=attn_mask, output_attentions=True, use_cache=False)
    attentions = [a[0].float().mean(0) for a in out.attentions]
    return attentions


def flow_ratio(attentions: Sequence[torch.Tensor], trace_start: int, trace_end: int, q_positions: Sequence[int], a_positions: Sequence[int], layers_to_avg: int) -> float:
    if not attentions:
        return math.nan
    selected = attentions[-layers_to_avg:]
    ratios = []
    for attn in selected:
        for t in range(trace_start, trace_end):
            attn_a = float(attn[t, a_positions].sum().item()) if a_positions else 0.0
            attn_q = float(attn[t, q_positions].sum().item()) if q_positions else 0.0
            if attn_q > 1e-9:
                ratios.append(attn_a / attn_q)
    return float(np.mean(ratios)) if ratios else math.nan


def score(args: argparse.Namespace) -> None:
    rank, world, local = rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    methods = [m for m in args.methods.split(",") if m]
    fractions = [float(x) for x in args.fractions.split(",") if x]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(args.scoring_model, trust_remote_code=True)
        sep_ids = tok_1d(tokenizer, SEQ_SEP, torch.device("cpu"))
        dry_rows = []
        for sample_idx, row in enumerate(rows[: min(len(rows), args.dry_run_rows)]):
            for method in methods:
                q = ensure_text(row.get("questions", {}).get(method))
                a = ensure_text(row.get("answers", {}).get(method))
                r = ensure_text(row.get("reasonings", {}).get(method))
                n_steps, flat_reason, ends = tokenize_reasoning(tokenizer, r, sep_ids, torch.device("cpu"))
                dry_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "id": row.get("id", sample_idx),
                        "method": method,
                        "question_tokens": int(len(tokenizer(q, add_special_tokens=False).input_ids)),
                        "answer_tokens": int(len(tokenizer(a, add_special_tokens=False).input_ids)),
                        "reasoning_tokens": int(flat_reason.numel()),
                        "n_steps": n_steps,
                        "prefix_token_ends": prefix_end_indices(int(flat_reason.numel()), fractions),
                        "step_boundaries": ends[:10],
                    }
                )
        out = args.output_dir / "attention_dry_run.jsonl"
        write_jsonl(out, dry_rows)
        print(json.dumps({"dry_run": True, "rows": len(dry_rows), "output": str(out)}, indent=2))
        return

    model, tokenizer, max_ctx = load_model_and_tokenizer(args.scoring_model, device, args.attn_implementation)
    sep_ids = tok_1d(tokenizer, SEQ_SEP, device)
    metrics_path = args.output_dir / f"attention_metrics_rank{rank}.jsonl"
    invalid_path = args.output_dir / f"attention_invalid_rank{rank}.jsonl"
    valid = invalid = 0
    with metrics_path.open("w", encoding="utf-8") as fw, invalid_path.open("w", encoding="utf-8") as fi:
        iterator = list(enumerate(rows))[rank::world]
        for sample_idx, row in tqdm(iterator, desc=f"attn-rank{rank}", disable=(rank != 0)):
            for method in methods:
                try:
                    q = ensure_text(row.get("questions", {}).get(method))
                    a = ensure_text(row.get("answers", {}).get(method))
                    r = ensure_text(row.get("reasonings", {}).get(method))
                    if not (q.strip() and a.strip() and r.strip()):
                        raise ValueError("missing Q/A/R")
                    q_ids = tok_1d(tokenizer, apply_chat_prefix(tokenizer, q), device)
                    a_ids = tok_1d(tokenizer, a, device)
                    n_steps, flat_reason, ends = tokenize_reasoning(tokenizer, r, sep_ids, device)
                    if int(flat_reason.numel()) == 0 or int(a_ids.numel()) == 0:
                        raise ValueError("empty tokens")
                    prefix_q = q_ids
                    token_ends = prefix_end_indices(int(flat_reason.numel()), fractions)
                    max_full_tokens = int(prefix_q.numel() + a_ids.numel() + (max(token_ends) if token_ends else 0))
                    if args.max_full_tokens is not None and max_full_tokens > args.max_full_tokens:
                        raise ValueError(f"full token length {max_full_tokens} exceeds max_full_tokens={args.max_full_tokens}")
                    prompt_ids = torch.cat([prefix_q, a_ids], dim=0)
                    q_positions = list(range(int(prefix_q.numel())))
                    a_positions = list(range(int(prefix_q.numel()), int(prefix_q.numel() + a_ids.numel())))
                    qattn = []
                    for end in token_ends:
                        full_ids = torch.cat([prefix_q, a_ids, flat_reason[:end]], dim=0)
                        attentions = None
                        try:
                            attentions = attentions_for_prompt(model, full_ids, max_ctx)
                            trace_start = int(prefix_q.numel() + a_ids.numel())
                            trace_end = int(full_ids.numel())
                            ratio = flow_ratio(attentions, trace_start, trace_end, q_positions, a_positions, args.layers_to_avg)
                        finally:
                            del attentions
                        qattn.append(ratio)
                    rec: Dict[str, Any] = {
                        "sample_idx": sample_idx,
                        "id": row.get("id", sample_idx),
                        "method": method,
                        "n_steps": n_steps,
                        "reasoning_tokens": int(flat_reason.numel()),
                        "question_tokens": int(prefix_q.numel()),
                        "answer_tokens": int(a_ids.numel()),
                        "max_full_tokens": max_full_tokens,
                        "prefix_token_ends": token_ends,
                        "attention_ratio": qattn,
                        "attention_mean": float(np.mean(qattn)),
                    }
                    for frac, val in zip(fractions, qattn):
                        rec[f"attn_{int(round(frac * 100)):03d}"] = val
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fw.flush()
                    valid += 1
                except Exception as exc:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    fi.write(json.dumps({"sample_idx": sample_idx, "id": row.get("id", sample_idx), "method": method, "error": repr(exc)}, ensure_ascii=False) + "\n")
                    fi.flush()
                    invalid += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(json.dumps({"rank": rank, "valid": valid, "invalid": invalid, "metrics": str(metrics_path)}, indent=2))


def read_metric_rows(path: Path) -> List[Dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("attention_metrics_rank*.jsonl"))
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
    fields = ["attn_010", "attn_025", "attn_050", "attn_075", "attn_090", "attention_mean"]
    fields = [f for f in fields if any(f in row for row in rows)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    means_path = args.output_dir / "attention_means.csv"
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
    deltas_path = args.output_dir / "attention_deltas.csv"
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
        "# Attention Flow",
        "",
        "Attention ratio = mean attention from reasoning tokens to answer tokens divided by attention to question tokens.",
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
    (args.output_dir / "attention_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "means": str(means_path), "deltas": str(deltas_path), "report": str(args.output_dir / "attention_report.md")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_score = sub.add_parser("score")
    p_score.add_argument("--input", type=Path, required=True)
    p_score.add_argument("--output-dir", type=Path, required=True)
    p_score.add_argument("--scoring-model", type=Path, default=Path("/home/pengguangyue/workspace/models/Qwen/Qwen3-8B"))
    p_score.add_argument("--methods", default=",".join(METHODS))
    p_score.add_argument("--fractions", default=",".join(str(x) for x in FRACTIONS))
    p_score.add_argument("--attn-implementation", default="eager")
    p_score.add_argument("--layers-to-avg", type=int, default=4)
    p_score.add_argument("--max-full-tokens", type=int)
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
