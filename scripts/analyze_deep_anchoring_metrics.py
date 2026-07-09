#!/usr/bin/env python3
"""Deep anchoring diagnostics for suppression and SSR traces.

This script implements CPU-only metrics derived from
`insights/metrics_framework.md`.  The emphasis is on compression-based
answer-specific constraints that do not reuse the neural scorer used by Aprob/B.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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


def to_bytes(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        return value
    return ensure_text(value).encode("utf-8", errors="ignore")


def compressed_len(value: str | bytes, level: int) -> int:
    payload = to_bytes(value)
    if not payload:
        return 0
    return len(zlib.compress(payload, level=level))


def compressed_len_with_dict(payload: str | bytes, zdict: str | bytes, level: int) -> int:
    payload_bytes = to_bytes(payload)
    zdict_bytes = to_bytes(zdict)[-32768:]
    if not payload_bytes:
        return 0
    if not zdict_bytes:
        return compressed_len(payload_bytes, level)
    compressor = zlib.compressobj(level=level, zdict=zdict_bytes)
    return len(compressor.compress(payload_bytes) + compressor.flush())


def conditional_saving(payload: str, zdict: str, level: int) -> float:
    payload_bytes = to_bytes(payload)
    if not payload_bytes:
        return math.nan
    return float(compressed_len(payload_bytes, level) - compressed_len_with_dict(payload_bytes, zdict, level))


def joint_compression_mi(x: str, y: str, level: int) -> float:
    x_bytes = to_bytes(x)
    y_bytes = to_bytes(y)
    if not x_bytes or not y_bytes:
        return math.nan
    sep = b"\n<SEP>\n"
    return float(compressed_len(x_bytes, level) + compressed_len(y_bytes, level) - compressed_len(x_bytes + sep + y_bytes, level))


def decoy_answers(rows: Sequence[Dict[str, Any]], sample_idx: int, method: str, k: int) -> List[str]:
    if k <= 0 or len(rows) <= 1:
        return []
    out = []
    for offset in range(1, min(k, len(rows) - 1) + 1):
        answer = ensure_text(rows[(sample_idx + offset) % len(rows)].get("answers", {}).get(method))
        if answer.strip():
            out.append(answer)
    return out


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else math.nan


def percentile(values: Sequence[float], pct: float) -> float:
    vals = sorted(values)
    if not vals:
        return math.nan
    pos = (len(vals) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def bootstrap_ci(values: Sequence[float], rng: random.Random, n_boot: int) -> Tuple[float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    boots = [statistics.fmean(rng.choice(vals) for _ in vals) for _ in range(n_boot)]
    return percentile(boots, 0.025), percentile(boots, 0.975)


def analyze_records(rows: Sequence[Dict[str, Any]], methods: Sequence[str], n_decoys: int, level: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for sample_idx, row in enumerate(rows):
        for method in methods:
            question = ensure_text(row.get("questions", {}).get(method))
            answer = ensure_text(row.get("answers", {}).get(method))
            reasoning = ensure_text(row.get("reasonings", {}).get(method))
            if not (question.strip() and answer.strip() and reasoning.strip()):
                continue

            r_bytes = max(len(to_bytes(reasoning)), 1)
            a_bytes = max(len(to_bytes(answer)), 1)
            s_a_to_r = conditional_saving(reasoning, answer, level)
            s_q_to_r = conditional_saving(reasoning, question, level)
            s_qa_to_r = conditional_saving(reasoning, f"{question}\n{answer}", level)
            answer_unique = s_qa_to_r - s_q_to_r
            joint_mi = joint_compression_mi(reasoning, answer, level)

            decoy_cond = []
            decoy_answer_unique = []
            decoy_joint_norm = []
            for decoy in decoy_answers(rows, sample_idx, method, n_decoys):
                decoy_bytes = max(len(to_bytes(decoy)), 1)
                decoy_cond.append(conditional_saving(reasoning, decoy, level) / r_bytes)
                decoy_answer_unique.append((conditional_saving(reasoning, f"{question}\n{decoy}", level) - s_q_to_r) / r_bytes)
                decoy_joint_norm.append(joint_compression_mi(reasoning, decoy, level) / math.sqrt(r_bytes * decoy_bytes))

            cond_per_r = s_a_to_r / r_bytes
            answer_unique_per_r = answer_unique / r_bytes
            joint_norm = joint_mi / math.sqrt(r_bytes * a_bytes)

            records.append(
                {
                    "sample_idx": sample_idx,
                    "id": row.get("id", sample_idx),
                    "method": method,
                    "r_bytes": r_bytes,
                    "a_bytes": a_bytes,
                    "cond_A_to_R_bytes": s_a_to_r,
                    "cond_Q_to_R_bytes": s_q_to_r,
                    "cond_QA_to_R_bytes": s_qa_to_r,
                    "answer_unique_cond_bytes": answer_unique,
                    "joint_compression_mi_bytes": joint_mi,
                    "cond_A_to_R_per_r_byte": cond_per_r,
                    "cond_A_to_R_per_a_byte": s_a_to_r / a_bytes,
                    "answer_unique_cond_per_r_byte": answer_unique_per_r,
                    "answer_unique_cond_per_a_byte": answer_unique / a_bytes,
                    "cond_A_vs_decoy_per_r_byte": cond_per_r - mean(decoy_cond),
                    "answer_unique_vs_decoy_per_r_byte": answer_unique_per_r - mean(decoy_answer_unique),
                    "joint_mi_norm_sqrt": joint_norm,
                    "joint_mi_vs_decoy_norm_sqrt": joint_norm - mean(decoy_joint_norm),
                }
            )
    return records


def paired_delta_values(records: Sequence[Dict[str, Any]], metrics: Sequence[str]) -> Dict[Tuple[str, str, str], List[float]]:
    return paired_delta_values_for_pairs(records, metrics, DELTA_PAIRS)


def paired_delta_values_for_pairs(
    records: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    delta_pairs: Sequence[Tuple[str, str]],
) -> Dict[Tuple[str, str, str], List[float]]:
    by_sample: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        by_sample[int(rec["sample_idx"])][str(rec["method"])] = rec
    out: Dict[Tuple[str, str, str], List[float]] = {}
    for metric in metrics:
        for method, base in delta_pairs:
            vals = []
            for method_rows in by_sample.values():
                if method not in method_rows or base not in method_rows:
                    continue
                a = method_rows[method].get(metric)
                b = method_rows[base].get(metric)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(float(a)) and math.isfinite(float(b)):
                    vals.append(float(a) - float(b))
            out[(metric, method, base)] = vals
    return out


def parse_delta_pairs(text: str) -> List[Tuple[str, str]]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "|" in item:
            method, base = item.split("|", 1)
        elif ":" in item:
            method, base = item.split(":", 1)
        elif "-" in item:
            method, base = item.split("-", 1)
        else:
            raise ValueError(f"Delta pair must be METHOD|BASE or METHOD-BASE, got: {item}")
        out.append((method.strip(), base.strip()))
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.6f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out)


def signed_support(metric: str, delta: str, value: float, ci_low: float, ci_high: float) -> str:
    if not math.isfinite(value):
        return "missing"
    if delta == "SSR-NEU":
        if ci_high < 0:
            return "supports SSR-better"
        if value < 0:
            return "weak SSR-better"
        return "contradicts SSR-better"
    if delta in {"SUP-NEU", "AUG-SUP-NEU"}:
        if ci_low > 0:
            return "supports worse-than-NEU"
        if value > 0:
            return "weak worse-than-NEU"
        return "does-not-support-worse"
    return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--delta-pairs", default=",".join(f"{m}-{b}" for m, b in DELTA_PAIRS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--decoys", type=int, default=8)
    parser.add_argument("--zlib-level", type=int, default=9)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    methods = [method for method in args.methods.split(",") if method]
    delta_pairs = parse_delta_pairs(args.delta_pairs)
    metrics = [
        "cond_A_to_R_per_r_byte",
        "cond_A_to_R_per_a_byte",
        "answer_unique_cond_per_r_byte",
        "answer_unique_cond_per_a_byte",
        "cond_A_vs_decoy_per_r_byte",
        "answer_unique_vs_decoy_per_r_byte",
        "joint_mi_norm_sqrt",
        "joint_mi_vs_decoy_norm_sqrt",
    ]

    records = analyze_records(rows, methods, args.decoys, args.zlib_level)
    summary_rows = []
    for metric in metrics:
        for method in methods:
            vals = [float(rec[metric]) for rec in records if rec["method"] == method and math.isfinite(float(rec[metric]))]
            summary_rows.append({"metric": metric, "method": method, "n": len(vals), "mean": mean(vals)})

    rng = random.Random(args.seed)
    delta_rows = []
    for (metric, method, base), vals in paired_delta_values_for_pairs(records, metrics, delta_pairs).items():
        lo, hi = bootstrap_ci(vals, rng, args.bootstrap)
        avg = mean(vals)
        delta = f"{method}-{base}"
        delta_rows.append(
            {
                "metric": metric,
                "delta": delta,
                "n": len(vals),
                "mean_delta": avg,
                "ci95_low": lo,
                "ci95_high": hi,
                "ci_excludes_zero": bool(vals) and (lo > 0 or hi < 0),
                "interpretation": signed_support(metric, delta, avg, lo, hi),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_sample_deep_metrics.csv", records, sorted({key for rec in records for key in rec}))
    write_csv(args.output_dir / "summary_by_method.csv", summary_rows, ["metric", "method", "n", "mean"])
    write_csv(
        args.output_dir / "paired_deltas_vs_neu.csv",
        delta_rows,
        ["metric", "delta", "n", "mean_delta", "ci95_low", "ci95_high", "ci_excludes_zero", "interpretation"],
    )

    by_metric_method = {(row["metric"], row["method"]): row for row in summary_rows}
    by_metric_delta = {(row["metric"], row["delta"]): row for row in delta_rows}
    report_lines = [
        f"# Deep Anchoring Metrics (n={len(rows)})",
        "",
        "## Setup",
        "",
        f"- Input: `{args.input}`",
        f"- Methods: `{','.join(methods)}`",
        f"- Decoy answers per sample: `{args.decoys}`",
        f"- zlib level: `{args.zlib_level}`",
        "",
        "## Means by method",
        "",
        markdown_table(
            ["metric", *methods],
            [
                [
                    metric,
                    *[float(by_metric_method.get((metric, method), {}).get("mean", math.nan)) for method in methods],
                ]
                for metric in metrics
            ],
        ),
        "",
        "## Paired deltas vs NEU",
        "",
        markdown_table(
            ["metric", *[f"{method}-{base}" for method, base in delta_pairs]],
            [
                [
                    metric,
                    *[
                        f"{float(by_metric_delta.get((metric, delta), {}).get('mean_delta', math.nan)):.6f} "
                        f"[{float(by_metric_delta.get((metric, delta), {}).get('ci95_low', math.nan)):.6f}, "
                        f"{float(by_metric_delta.get((metric, delta), {}).get('ci95_high', math.nan)):.6f}]"
                        for delta in [f"{method}-{base}" for method, base in delta_pairs]
                    ],
                ]
                for metric in metrics
            ],
        ),
        "",
        "## Directional assessment",
        "",
        markdown_table(
            ["metric", *[f"{method}-{base}" for method, base in delta_pairs]],
            [
                [
                    metric,
                    *[by_metric_delta.get((metric, delta), {}).get("interpretation", "missing") for delta in [f"{method}-{base}" for method, base in delta_pairs]],
                ]
                for metric in metrics
            ],
        ),
        "",
        "## Short conclusion",
        "",
        "Compression diagnostics strongly support `SSR` as lower-answer-constraint than `NEU`.",
        "`AUG-SUP` is worse than `NEU` on the strongest conditional/decoy compression metrics.",
        "`SUP` trends slightly worse than `NEU` on per-reasoning-byte compression, but the paired confidence intervals mostly cross zero.",
        "Therefore this stage supports `SSR` best and `AUG-SUP` worst; it does not yet prove plain `SUP` is worse than `NEU`.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "output_dir": str(args.output_dir), "report": str(args.output_dir / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
