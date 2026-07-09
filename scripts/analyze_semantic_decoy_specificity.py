#!/usr/bin/env python3
"""Semantic decoy specificity for RCoT traces.

This is a CPU/GPU-light companion to `analyze_suppression_support.py`.
It measures whether a reasoning trace is semantically closer to its true answer
than to nearby decoy answers from other samples:

    specificity = cos(embed(R), embed(A_true)) - mean_i cos(embed(R), embed(A_decoy_i))

If suppression hides lexical overlap but preserves semantic answer dependence,
SUP/AUG-SUP should have higher specificity than NEU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


METHODS = ["NEU", "SUP", "AUG-SUP", "SSR"]
DELTA_PAIRS = [("SUP", "NEU"), ("AUG-SUP", "NEU"), ("SSR", "NEU")]


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


def decoy_indices(n_rows: int, sample_idx: int, k: int) -> List[int]:
    if n_rows <= 1:
        return []
    return [(sample_idx + offset) % n_rows for offset in range(1, min(k, n_rows - 1) + 1)]


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else math.nan


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    pos = (len(values) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def bootstrap_ci(values: Sequence[float], rng: random.Random, n_boot: int) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    boots = []
    for _ in range(n_boot):
        boots.append(statistics.fmean(rng.choice(vals) for _ in vals))
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
    return DELTA_PAIRS


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def encode_texts(model: SentenceTransformer, texts: Sequence[str], batch_size: int) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def analyze(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    methods = [m for m in args.methods.split(",") if m]
    delta_pairs = parse_delta_pairs(args.delta_pairs, methods)

    model_kwargs: Dict[str, Any] = {}
    if args.device:
        model_kwargs["device"] = args.device
    model = SentenceTransformer(str(args.embedding_model), **model_kwargs)
    model.max_seq_length = args.max_length

    text_keys: Dict[Tuple[str, int, str], int] = {}
    texts: List[str] = []

    def add_text(kind: str, sample_idx: int, method: str, text: str) -> None:
        key = (kind, sample_idx, method)
        if key not in text_keys:
            text_keys[key] = len(texts)
            texts.append(text)

    for sample_idx, row in enumerate(rows):
        for method in methods:
            add_text("reasoning", sample_idx, method, ensure_text(row.get("reasonings", {}).get(method)))
            add_text("answer", sample_idx, method, ensure_text(row.get("answers", {}).get(method)))

    vecs = encode_texts(model, texts, args.batch_size)

    records: List[Dict[str, Any]] = []
    for sample_idx, row in enumerate(rows):
        decoys = decoy_indices(len(rows), sample_idx, args.decoys)
        for method in methods:
            r_idx = text_keys[("reasoning", sample_idx, method)]
            a_idx = text_keys[("answer", sample_idx, method)]
            r_vec = vecs[r_idx]
            true_cos = float(np.dot(r_vec, vecs[a_idx]))
            decoy_vals = []
            for decoy_idx in decoys:
                key = ("answer", decoy_idx, method)
                if key in text_keys:
                    decoy_vals.append(float(np.dot(r_vec, vecs[text_keys[key]])))
            decoy_mean = mean(decoy_vals)
            records.append(
                {
                    "sample_idx": sample_idx,
                    "id": row.get("id", sample_idx),
                    "method": method,
                    "semantic_true_cosine": true_cos,
                    "semantic_decoy_cosine_mean": decoy_mean,
                    "semantic_decoy_specificity": true_cos - decoy_mean if math.isfinite(decoy_mean) else math.nan,
                    "semantic_decoys": len(decoy_vals),
                }
            )

    summary_rows = []
    metrics = ["semantic_true_cosine", "semantic_decoy_cosine_mean", "semantic_decoy_specificity"]
    for metric in metrics:
        for method in methods:
            vals = [float(r[metric]) for r in records if r["method"] == method and math.isfinite(float(r.get(metric, math.nan)))]
            summary_rows.append({"metric": metric, "method": method, "n": len(vals), "mean": mean(vals)})

    by_sample: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        by_sample[int(rec["sample_idx"])][str(rec["method"])] = rec
    rng = random.Random(args.seed)
    delta_rows = []
    for metric in metrics:
        for method, base in delta_pairs:
            vals = []
            for sample_methods in by_sample.values():
                if method not in sample_methods or base not in sample_methods:
                    continue
                a = sample_methods[method].get(metric)
                b = sample_methods[base].get(metric)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(float(a)) and math.isfinite(float(b)):
                    vals.append(float(a) - float(b))
            lo, hi = bootstrap_ci(vals, rng, args.bootstrap)
            delta_rows.append(
                {
                    "metric": metric,
                    "delta": f"{method}-{base}",
                    "n": len(vals),
                    "mean_delta": mean(vals),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "ci_excludes_zero": bool(vals) and (lo > 0 or hi < 0),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_sample_semantic_decoy.csv", records, ["sample_idx", "id", "method", *metrics, "semantic_decoys"])
    write_csv(args.output_dir / "summary_by_method.csv", summary_rows, ["metric", "method", "n", "mean"])
    write_csv(args.output_dir / "paired_deltas.csv", delta_rows, ["metric", "delta", "n", "mean_delta", "ci95_low", "ci95_high", "ci_excludes_zero"])

    by_metric_method = {(row["metric"], row["method"]): row for row in summary_rows}
    lines = [f"# Semantic Decoy Specificity (n={len(rows)})", "", "## Means by method", ""]
    lines.append("| metric | " + " | ".join(methods) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(methods)) + " |")
    for metric in metrics:
        vals = []
        for method in methods:
            row = by_metric_method.get((metric, method), {})
            vals.append(float(row.get("mean", math.nan)))
        lines.append("| " + metric + " | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    lines.extend(["", "## Paired deltas", ""])
    lines.append("| metric | delta | mean | 95% CI | excludes 0 |")
    lines.append("| --- | --- | ---: | --- | --- |")
    for row in delta_rows:
        lines.append(f"| {row['metric']} | {row['delta']} | {float(row['mean_delta']):.4f} | [{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}] | {row['ci_excludes_zero']} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "records": len(records), "output_dir": str(args.output_dir)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, default=Path("/home/pengguangyue/workspace/models/xlm-roberta-large"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--delta-pairs", help="Comma-separated METHOD|BASE pairs. Defaults to each method vs NEU, or vs the first method if NEU is absent.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--decoys", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
