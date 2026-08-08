#!/usr/bin/env python3
"""Compute prefix answer-information curves for RCoT traces.

For each (Q, A, R), this computes

    B_tau = log2 P(A | Q, R_prefix_tau) / |A| - log2 P(A | Q) / |A|

at token-prefix fractions such as 0.10, 0.25, 0.50, 0.75, 0.90, and 1.00.
It uses the same chat prefix and answer log-probability convention as
`reviewer_protocol.py`, so the final prefix should match the existing B metric
up to tokenization/splitting details.
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


METHOD_ORDER = ["NEU", "SUP", "AUG-SUP", "SSR"]
DEFAULT_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
LOGE_TO_BITS = 1.0 / math.log(2.0)
SEQ_SEP = "\n\n"
TORCH_DTYPE = torch.bfloat16
STEP_MICROBATCH = 4
DEFAULT_MIN_FINAL_GAIN_FOR_SHARE = 0.02


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    if len(sentence_parts) > 1:
        return sentence_parts
    return parts


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


@torch.no_grad()
def batch_answer_logprob(
    model: Any,
    prompt_ids_list: List[torch.LongTensor],
    ans_ids: torch.LongTensor,
    max_ctx: Optional[int],
    batch_size: int,
) -> List[float]:
    device = next(model.parameters()).device
    ans_ids = ans_ids.to(device)
    ans_len = int(ans_ids.numel())
    scores = []
    for i0 in range(0, len(prompt_ids_list), batch_size):
        chunk = prompt_ids_list[i0 : i0 + batch_size]
        seqs, prompt_lens = [], []
        for prompt in chunk:
            full = torch.cat([prompt.to(device), ans_ids], dim=0)
            if max_ctx is not None and int(full.numel()) > max_ctx:
                full = full[-int(max_ctx) :]
            seqs.append(full)
            prompt_lens.append(int(full.numel()) - ans_len)
        max_len = max(int(s.numel()) for s in seqs)
        input_ids = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        for b, seq in enumerate(seqs):
            length = int(seq.numel())
            input_ids[b, :length] = seq
            attn[b, :length] = 1
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logp = torch.log_softmax(out.logits, dim=-1)
        for b, prompt_len in enumerate(prompt_lens):
            pred_pos = torch.arange(prompt_len - 1, prompt_len + ans_len - 1, device=device)
            target = input_ids[b, prompt_len : prompt_len + ans_len]
            selected = logp[b, pred_pos, :].gather(1, target.unsqueeze(1)).squeeze(1)
            scores.append(float(selected.sum().item()))
        del out, logp, input_ids, attn
    return scores


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def rank_info() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank))
        return rank, world, local
    return 0, 1, 0


def prefix_end_indices(n_tokens: int, fractions: Sequence[float]) -> List[int]:
    out = []
    for frac in fractions:
        frac = min(max(float(frac), 0.0), 1.0)
        if frac <= 0:
            out.append(0)
        else:
            out.append(min(n_tokens, max(1, int(math.ceil(n_tokens * frac)))))
    return out


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


def score(args: argparse.Namespace) -> None:
    rank, world, local = rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_jsonl(args.input)
    start_index = int(getattr(args, "start_index", 0) or 0)
    end_index = int(args.limit) if args.limit is not None else len(all_rows)
    rows = all_rows[start_index:end_index]
    methods = [m for m in args.methods.split(",") if m]
    fractions = [float(x) for x in args.fractions.split(",") if x]

    if args.dry_run:
        dry_rows = []
        tokenizer = AutoTokenizer.from_pretrained(args.scoring_model, trust_remote_code=True)
        sep_ids = tok_1d(tokenizer, SEQ_SEP, torch.device("cpu"))
        for sample_idx, row in enumerate(rows[: min(len(rows), args.dry_run_rows)], start=start_index):
            for method in methods:
                r = ensure_text(row.get("reasonings", {}).get(method))
                n_steps, flat_reason, end_pos = tokenize_reasoning(tokenizer, r, sep_ids, torch.device("cpu"))
                dry_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "id": row.get("id", sample_idx),
                        "method": method,
                        "n_steps": n_steps,
                        "reasoning_tokens": int(flat_reason.numel()),
                        "prefix_fractions": fractions,
                        "prefix_token_ends": prefix_end_indices(int(flat_reason.numel()), fractions),
                        "step_boundaries": end_pos[:10],
                    }
                )
        out = args.output_dir / "prefix_dry_run.jsonl"
        write_jsonl(out, dry_rows)
        print(json.dumps({"dry_run": True, "rows": len(dry_rows), "output": str(out)}, indent=2))
        return

    model, tokenizer, max_ctx = load_model_and_tokenizer(args.scoring_model, device)
    sep_ids = tok_1d(tokenizer, SEQ_SEP, device)
    think_close_ids = tok_1d(tokenizer, "</think>\n\n", device)

    metrics_path = args.output_dir / f"prefix_metrics_rank{rank}.jsonl"
    invalid_path = args.output_dir / f"prefix_invalid_rank{rank}.jsonl"
    valid = invalid = 0
    with metrics_path.open("w", encoding="utf-8") as fw, invalid_path.open("w", encoding="utf-8") as fi:
        iterator = list(enumerate(rows, start=start_index))[rank::world]
        for sample_idx, row in tqdm(iterator, desc=f"prefix-rank{rank}", disable=(rank != 0)):
            for method in methods:
                try:
                    q = ensure_text(row.get("questions", {}).get(method))
                    a = ensure_text(row.get("answers", {}).get(method))
                    r = ensure_text(row.get("reasonings", {}).get(method))
                    if not (q.strip() and a.strip() and r.strip()):
                        continue
                    ans_ids = tok_1d(tokenizer, a, device)
                    if int(ans_ids.numel()) == 0:
                        raise ValueError("empty answer ids")
                    n_steps, flat_reason, end_pos = tokenize_reasoning(tokenizer, r, sep_ids, device)
                    n_tokens = int(flat_reason.numel())
                    if n_tokens == 0:
                        raise ValueError("empty reasoning ids")

                    prefix_q = tok_1d(tokenizer, apply_chat_prefix(tokenizer, q), device)
                    base_prompt = torch.cat([prefix_q, think_close_ids], dim=0)
                    token_ends = prefix_end_indices(n_tokens, fractions)
                    prompts = [base_prompt]
                    for end in token_ends:
                        prompts.append(torch.cat([prefix_q, flat_reason[:end], think_close_ids], dim=0))
                    logps = batch_answer_logprob(model, prompts, ans_ids, max_ctx, args.microbatch)
                    base = float(logps[0] * LOGE_TO_BITS / int(ans_ids.numel()))
                    prefix_bits = [float(x * LOGE_TO_BITS / int(ans_ids.numel())) for x in logps[1:]]
                    gains = [x - base for x in prefix_bits]
                    final_gain = gains[-1]

                    rec: Dict[str, Any] = {
                        "sample_idx": sample_idx,
                        "id": row.get("id", sample_idx),
                        "method": method,
                        "n_steps": n_steps,
                        "reasoning_tokens": n_tokens,
                        "answer_tokens": int(ans_ids.numel()),
                        "logp_base_per_token_bits": base,
                        "prefix_fractions": fractions,
                        "prefix_token_ends": token_ends,
                        "prefix_logp_per_token_bits": prefix_bits,
                        "prefix_B": gains,
                        "B_final": final_gain,
                    }
                    for frac, gain in zip(fractions, gains):
                        rec[f"B_{int(round(frac * 100)):03d}"] = gain
                    if len(gains) >= 2:
                        gain25 = rec.get("B_025", gains[min(1, len(gains) - 1)])
                        gain90 = rec.get("B_090", gains[-2])
                        rec["EarlyGain25"] = float(gain25)
                        rec["LateGain90minus25"] = float(gain90) - float(gain25)
                        if final_gain > args.min_final_gain_for_share:
                            rec["EarlyShare25"] = float(gain25) / float(final_gain)
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
    vals = list(values)
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], vals[0]
    means = []
    for _ in range(n_boot):
        means.append(float(np.mean([rng.choice(vals) for _ in vals])))
    return percentile(means, 0.025), percentile(means, 0.975)


def gini_nonnegative(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)) and float(v) >= 0.0)
    if not vals:
        return float("nan")
    total = float(sum(vals))
    if total <= 0.0:
        return 0.0
    n = len(vals)
    weighted = sum((idx + 1) * val for idx, val in enumerate(vals))
    return float((2.0 * weighted) / (n * total) - (n + 1.0) / n)


def parse_delta_pairs(text: Optional[str], methods: Sequence[str]) -> List[Tuple[str, str]]:
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


def read_prefix_metrics(path: Path) -> List[Dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("prefix_metrics_rank*.jsonl"))
    rows: List[Dict[str, Any]] = []
    for file in files:
        rows.extend(read_jsonl(file))
    return rows


def prepare_summary_rows(
    rows: List[Dict[str, Any]],
    limit: Optional[int],
    min_final_gain_for_share: float,
) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        sample_idx = int(row.get("sample_idx", -1))
        if limit is not None and sample_idx >= limit:
            continue
        row = dict(row)
        if "B_025" in row:
            row["EarlyGain25"] = float(row["B_025"])
        if "B_090" in row and "B_025" in row:
            row["LateGain90minus25"] = float(row["B_090"]) - float(row["B_025"])
        final_gain = float(row.get("B_100", row.get("B_final", float("nan"))))
        if "B_025" in row and math.isfinite(final_gain) and final_gain > min_final_gain_for_share:
            row["EarlyShare25"] = float(row["B_025"]) / final_gain
        else:
            row.pop("EarlyShare25", None)
        curve = [0.0]
        if isinstance(row.get("prefix_B"), list):
            curve.extend(float(v) for v in row["prefix_B"] if math.isfinite(float(v)))
        else:
            for field in ("B_010", "B_025", "B_050", "B_075", "B_090", "B_100"):
                if field in row and math.isfinite(float(row[field])):
                    curve.append(float(row[field]))
        if len(curve) >= 2:
            positive_increments = [max(0.0, curve[i] - curve[i - 1]) for i in range(1, len(curve))]
            row["CommitmentConcentration"] = gini_nonnegative(positive_increments)
        row["B_final"] = final_gain
        out.append(row)
    return out


def summarize(args: argparse.Namespace) -> None:
    rows = prepare_summary_rows(
        read_prefix_metrics(args.metrics),
        args.limit,
        args.min_final_gain_for_share,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = [m for m in args.methods.split(",") if m]
    delta_pairs = parse_delta_pairs(args.delta_pairs, methods)
    fields = [
        "B_010", "B_025", "B_050", "B_075", "B_090", "B_100",
        "EarlyGain25", "LateGain90minus25", "EarlyShare25",
        "CommitmentConcentration", "B_final",
    ]
    fields = [f for f in fields if any(f in row for row in rows)]

    means_path = args.output_dir / "prefix_means.csv"
    with means_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "method", "n", "mean"])
        writer.writeheader()
        for field in fields:
            for method in methods:
                vals = [float(row[field]) for row in rows if row.get("method") == method and field in row and np.isfinite(float(row[field]))]
                writer.writerow({"metric": field, "method": method, "n": len(vals), "mean": float(np.mean(vals)) if vals else float("nan")})

    by_sample: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_sample[int(row["sample_idx"])][str(row["method"])] = row
    rng = random.Random(args.seed)
    delta_path = args.output_dir / "prefix_deltas.csv"
    with delta_path.open("w", encoding="utf-8", newline="") as f:
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

    report = args.output_dir / "prefix_report.md"
    means = list(csv.DictReader(means_path.open(encoding="utf-8")))
    deltas = list(csv.DictReader(delta_path.open(encoding="utf-8")))
    lines = ["# Prefix Answer-Information Curve", "", "## Means by method", ""]
    lines.append("| metric | " + " | ".join(methods) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(methods)) + " |")
    for field in fields:
        vals = {}
        for method in methods:
            row = next((r for r in means if r["metric"] == field and r["method"] == method), None)
            vals[method] = float(row["mean"]) if row else float("nan")
        lines.append("| " + field + " | " + " | ".join(f"{vals.get(method, float('nan')):.4f}" for method in methods) + " |")
    lines.extend(["", "## Paired deltas", ""])
    lines.append("| metric | delta | mean | 95% CI | excludes 0 |")
    lines.append("| --- | --- | ---: | --- | --- |")
    for row in deltas:
        lines.append(f"| {row['metric']} | {row['delta']} | {float(row['mean_delta']):.4f} | [{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}] | {row['ci_excludes_zero']} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "means": str(means_path), "deltas": str(delta_path), "report": str(report)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_score = sub.add_parser("score")
    p_score.add_argument("--input", type=Path, required=True)
    p_score.add_argument("--output-dir", type=Path, required=True)
    p_score.add_argument("--scoring-model", type=Path, required=True)
    p_score.add_argument("--methods", default="NEU,SUP,AUG-SUP,SSR")
    p_score.add_argument("--fractions", default="0.10,0.25,0.50,0.75,0.90,1.00")
    p_score.add_argument("--start-index", type=int, default=0, help="Start row offset for resumable chunk scoring; sample_idx keeps this global offset.")
    p_score.add_argument("--limit", type=int)
    p_score.add_argument("--microbatch", type=int, default=STEP_MICROBATCH)
    p_score.add_argument("--min-final-gain-for-share", type=float, default=DEFAULT_MIN_FINAL_GAIN_FOR_SHARE)
    p_score.add_argument("--dry-run", action="store_true")
    p_score.add_argument("--dry-run-rows", type=int, default=2)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--metrics", type=Path, required=True)
    p_sum.add_argument("--output-dir", type=Path, required=True)
    p_sum.add_argument("--methods", default="NEU,SUP,AUG-SUP,SSR")
    p_sum.add_argument("--delta-pairs", help="Comma-separated METHOD|BASE pairs. Defaults to each method vs NEU, or vs the first method if NEU is absent.")
    p_sum.add_argument("--limit", type=int, help="Only summarize samples with sample_idx < limit")
    p_sum.add_argument("--min-final-gain-for-share", type=float, default=DEFAULT_MIN_FINAL_GAIN_FOR_SHARE)
    p_sum.add_argument("--bootstrap", type=int, default=2000)
    p_sum.add_argument("--seed", type=int, default=13)

    args = parser.parse_args()
    if args.cmd == "score":
        score(args)
    elif args.cmd == "summarize":
        summarize(args)


if __name__ == "__main__":
    main()
