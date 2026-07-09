#!/usr/bin/env python3
"""Analyze whether reasoning length correlates with final anchoring metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


FINAL_METRICS = {
    "A_lex": "A_lex_QF",
    "A_traj": "A_traj",
    "A_prob": "A_prob",
}
GROUPS = {
    "main4": ["NEU", "SUP", "AUG-SUP", "SSR"],
    "main_stress10": ["NEU", "SUP", "AUG-SUP", "SSR", "QA-SUP", "PG-SUP", "FDB", "Gist", "BoN", "NGramBlock"],
    "controls4": ["Blind CoT", "Response-as-CoT", "+Prob Anchor", "+Traj Anchor"],
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl_files(path: Path, pattern: str) -> Iterable[Dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob(pattern))
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return float("nan")
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
    sx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    sy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy)


def rankdata(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in order[i:j]:
            ranks[k] = rank
        i = j
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return float("nan")
    rx = rankdata([x for x, _ in pairs])
    ry = rankdata([y for _, y in pairs])
    return pearson(rx, ry)


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return float("nan")
    mx = mean([x for x, _ in pairs])
    my = mean([y for _, y in pairs])
    denom = sum((x - mx) ** 2 for x, _ in pairs)
    if denom <= 1e-12:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / denom


def corr_row(label: str, metric: str, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    xs = [float(r["reasoning_tokens"]) for r in records]
    ys = [float(r[metric]) for r in records]
    return {
        "scope": label,
        "metric": metric,
        "n": len(records),
        "mean_tokens": mean(xs),
        "mean_metric": mean(ys),
        "pearson_r": pearson(xs, ys),
        "spearman_rho": spearman(xs, ys),
        "slope_per_100_tokens": 100.0 * slope(xs, ys),
    }


def residualize_by_method(records: Sequence[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_method[str(rec["method"])].append(rec)
    out: List[Dict[str, Any]] = []
    for method, vals in by_method.items():
        token_mean = mean([float(r["reasoning_tokens"]) for r in vals])
        metric_mean = mean([float(r[metric]) for r in vals])
        for rec in vals:
            out.append(
                {
                    "method": method,
                    "reasoning_tokens": float(rec["reasoning_tokens"]) - token_mean,
                    metric: float(rec[metric]) - metric_mean,
                }
            )
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (float, int)):
        if not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out) + "\n"


def load_records(per_record_csv: Path, traj_metrics: Path) -> List[Dict[str, Any]]:
    traj = {}
    for row in read_jsonl_files(traj_metrics, "commitment_metrics_rank*.jsonl"):
        traj[(int(row["sample_idx"]), str(row["method"]))] = 100.0 * float(row["ConfidenceGap_mean"])

    records: List[Dict[str, Any]] = []
    for row in read_csv(per_record_csv):
        key = (int(row["sample_idx"]), str(row["method"]))
        if key not in traj:
            continue
        records.append(
            {
                "sample_idx": key[0],
                "method": key[1],
                "reasoning_tokens": float(row["reasoning_tokens"]),
                "A_lex": 100.0 * float(row["A_lex_QF"]),
                "A_traj": traj[key],
                "A_prob": 100.0 * float(row["A_prob"]),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-record-csv", type=Path, required=True)
    parser.add_argument("--traj-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.per_record_csv, args.traj_metrics)
    methods = sorted({str(r["method"]) for r in records})
    groups = dict(GROUPS)
    groups["all14"] = methods

    by_method_rows: List[Dict[str, Any]] = []
    for method in methods:
        vals = [r for r in records if r["method"] == method]
        for metric in FINAL_METRICS:
            by_method_rows.append(corr_row(method, metric, vals))

    group_rows: List[Dict[str, Any]] = []
    for group, group_methods in groups.items():
        vals = [r for r in records if r["method"] in set(group_methods)]
        for metric in FINAL_METRICS:
            raw = corr_row(group, metric, vals)
            resid = corr_row(group + "_method_residual", metric, residualize_by_method(vals, metric))
            group_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "n": len(vals),
                    "raw_pearson_r": raw["pearson_r"],
                    "raw_spearman_rho": raw["spearman_rho"],
                    "method_resid_pearson_r": resid["pearson_r"],
                    "method_resid_spearman_rho": resid["spearman_rho"],
                    "method_resid_slope_per_100_tokens": resid["slope_per_100_tokens"],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "length_correlations_by_method.csv",
        by_method_rows,
        ["scope", "metric", "n", "mean_tokens", "mean_metric", "pearson_r", "spearman_rho", "slope_per_100_tokens"],
    )
    write_csv(
        args.output_dir / "length_correlations_by_group.csv",
        group_rows,
        [
            "group",
            "metric",
            "n",
            "raw_pearson_r",
            "raw_spearman_rho",
            "method_resid_pearson_r",
            "method_resid_spearman_rho",
            "method_resid_slope_per_100_tokens",
        ],
    )

    main4 = [r for r in group_rows if r["group"] == "main4"]
    main_method_rows = [r for r in by_method_rows if r["scope"] in GROUPS["main4"]]
    lines = [
        "# Length Effect Diagnostics",
        "",
        f"Records: `{len(records)}`",
        "",
        "Final metrics: `A_lex = 100 * A_lex_QF`, `A_traj = 100 * ConfidenceGap_mean`, `A_prob = 100 * clipped normalized answer-surprisal reduction`.",
        "",
        "## Main Methods: Method-Residual Correlation",
        "",
        markdown_table(
            ["Metric", "N", "Pearson", "Spearman", "Slope / 100 tok"],
            [
                [
                    r["metric"],
                    r["n"],
                    r["method_resid_pearson_r"],
                    r["method_resid_spearman_rho"],
                    r["method_resid_slope_per_100_tokens"],
                ]
                for r in main4
            ],
        ),
        "## Main Methods: Within-Method Correlation",
        "",
        markdown_table(
            ["Method", "Metric", "N", "Mean Tok", "Pearson", "Spearman", "Slope / 100 tok"],
            [
                [
                    r["scope"],
                    r["metric"],
                    r["n"],
                    r["mean_tokens"],
                    r["pearson_r"],
                    r["spearman_rho"],
                    r["slope_per_100_tokens"],
                ]
                for r in main_method_rows
            ],
        ),
        "## Group Summary",
        "",
        markdown_table(
            ["Group", "Metric", "N", "Raw Pearson", "Raw Spearman", "Residual Pearson", "Residual Spearman"],
            [
                [
                    r["group"],
                    r["metric"],
                    r["n"],
                    r["raw_pearson_r"],
                    r["raw_spearman_rho"],
                    r["method_resid_pearson_r"],
                    r["method_resid_spearman_rho"],
                ]
                for r in group_rows
            ],
        ),
        "",
        "Raw pooled correlations mix method identity with length; use method-residual or within-method rows for the causal concern.",
    ]
    (args.output_dir / "length_effect_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
