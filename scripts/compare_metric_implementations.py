#!/usr/bin/env python3
"""Compare proxy anchoring metrics with paper-formula metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


METHOD_ORDER = ["NEU", "SUP", "AUG-SUP", "SSR"]
METRIC_KEYS = {
    "Alex": "lexical_anchoring",
    "Aent": "entropy_anchoring",
    "Aprob": "ProbabilisticAnchoring",
}
ZONE_KEYS = ["Reason", "Encode", "Cloze", "Copy"]


def read_metrics(path_or_dir: Path) -> List[Dict[str, Any]]:
    files = [path_or_dir] if path_or_dir.is_file() else sorted(path_or_dir.glob("metrics_rank*.jsonl"))
    records: List[Dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        raise FileNotFoundError(f"No metric records found under {path_or_dir}")
    return records


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def summarize(records: Iterable[Dict[str, Any]], method_order: List[str] = METHOD_ORDER) -> Dict[str, Any]:
    values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        method = row.get("method", "unknown")
        for label, key in METRIC_KEYS.items():
            if finite(row.get(key)):
                values[method][label].append(float(row[key]))
        if finite(row.get("entropy_anchoring")) and finite(row.get("ProbabilisticAnchoring")):
            values[method]["_x"].append(float(row["entropy_anchoring"]))
            values[method]["_y"].append(float(row["ProbabilisticAnchoring"]))

    out: Dict[str, Any] = {}
    ordered = [m for m in method_order if m in values] + [m for m in values if m not in method_order]
    for method in ordered:
        metrics = {
            label: 100.0 * float(np.mean(values[method][label]))
            for label in METRIC_KEYS
            if values[method].get(label)
        }
        x = np.array(values[method].get("_x", []), dtype=float)
        y = np.array(values[method].get("_y", []), dtype=float)
        zones = {}
        if len(x) and len(x) == len(y):
            zones = {
                "Reason": 100.0 * float(np.mean((x < 0.5) & (y < 0.5))),
                "Encode": 100.0 * float(np.mean((x < 0.5) & (y >= 0.5))),
                "Cloze": 100.0 * float(np.mean((x >= 0.5) & (y < 0.5))),
                "Copy": 100.0 * float(np.mean((x >= 0.5) & (y >= 0.5))),
            }
        out[method] = {"N": int(len(values[method].get("Alex", []))), "metrics": metrics, "zones": zones}
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
        return str(v)

    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out) + "\n"


def write_summary_tables(summary: Dict[str, Any], out_dir: Path, prefix: str) -> None:
    metric_rows = []
    zone_rows = []
    for method, item in summary.items():
        metric_rows.append({
            "Method": method,
            "N": item["N"],
            "Alex": item["metrics"].get("Alex", np.nan),
            "Aent": item["metrics"].get("Aent", np.nan),
            "Aprob": item["metrics"].get("Aprob", np.nan),
        })
        zone_rows.append({
            "Method": method,
            "N": item["N"],
            **{zone: item["zones"].get(zone, np.nan) for zone in ZONE_KEYS},
        })
    write_csv(out_dir / f"{prefix}_table1_metrics.csv", metric_rows, ["Method", "N", "Alex", "Aent", "Aprob"])
    write_csv(out_dir / f"{prefix}_table2_zones.csv", zone_rows, ["Method", "N", *ZONE_KEYS])
    (out_dir / f"{prefix}_table1_metrics.md").write_text(
        markdown_table(["Method", "N", "Alex", "Aent", "Aprob"], [[r["Method"], r["N"], r["Alex"], r["Aent"], r["Aprob"]] for r in metric_rows]),
        encoding="utf-8",
    )
    (out_dir / f"{prefix}_table2_zones.md").write_text(
        markdown_table(["Method", "N", *ZONE_KEYS], [[r["Method"], r["N"], *(r[z] for z in ZONE_KEYS)] for r in zone_rows]),
        encoding="utf-8",
    )


def keyed(records: Iterable[Dict[str, Any]]) -> Dict[tuple[int, str], Dict[str, Any]]:
    out = {}
    for row in records:
        if "sample_idx" in row and row.get("method") is not None:
            out[(int(row["sample_idx"]), str(row["method"]))] = row
    return out


def corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def paired_rows(proxy_records: List[Dict[str, Any]], paper_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    proxy = keyed(proxy_records)
    paper = keyed(paper_records)
    rows = []
    for key in sorted(set(proxy) & set(paper)):
        pxy = proxy[key]
        pap = paper[key]
        sample_idx, method = key
        row = {"sample_idx": sample_idx, "method": method}
        for label, metric_key in METRIC_KEYS.items():
            row[f"proxy_{label}"] = float(pxy[metric_key])
            row[f"paper_{label}"] = float(pap[metric_key])
            row[f"delta_{label}"] = row[f"paper_{label}"] - row[f"proxy_{label}"]
        rows.append(row)
    return rows


def paired_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for method in METHOD_ORDER:
        subset = [r for r in rows if r["method"] == method]
        for label in ["Alex", "Aent", "Aprob"]:
            proxy_vals = [r[f"proxy_{label}"] for r in subset]
            paper_vals = [r[f"paper_{label}"] for r in subset]
            deltas = [r[f"delta_{label}"] for r in subset]
            if not subset:
                continue
            out.append({
                "Method": method,
                "Metric": label,
                "N_common": len(subset),
                "ProxyMean": 100.0 * float(np.mean(proxy_vals)),
                "PaperMean": 100.0 * float(np.mean(paper_vals)),
                "PaperMinusProxy": 100.0 * float(np.mean(deltas)),
                "Pearson": corr(proxy_vals, paper_vals),
            })
    return out


def write_report(out_dir: Path, proxy_summary: Dict[str, Any], paper_summary: Dict[str, Any], stats: List[Dict[str, Any]]) -> None:
    lines = [
        "# Paper-Formula vs Proxy Anchoring Metrics",
        "",
        "This report compares the repository's original proxy implementation with a paper-formula implementation.",
        "",
        "Paper-formula changes:",
        "",
        "- `Aent`: uses `Gunif = 1 / (1 + Var(u) / 0.1)` and `Lnon-unif = CV(delta) / (1 + CV(delta))`.",
        "- `Aprob`: uses per-answer-token PMI, `(log2 P(A|Q,R) - log2 P(A|Q)) / |A|`.",
        "- `Alex`: unchanged token LCS recall.",
        "",
        "## Proxy Summary",
        "",
        (out_dir / "proxy_table1_metrics.md").read_text(encoding="utf-8").strip(),
        "",
        (out_dir / "proxy_table2_zones.md").read_text(encoding="utf-8").strip(),
        "",
        "## Paper-Formula Summary",
        "",
        (out_dir / "paper_table1_metrics.md").read_text(encoding="utf-8").strip(),
        "",
        (out_dir / "paper_table2_zones.md").read_text(encoding="utf-8").strip(),
        "",
        "## Paired Metric Differences",
        "",
        markdown_table(
            ["Method", "Metric", "N", "ProxyMean", "PaperMean", "Paper-Proxy", "Pearson"],
            [[r["Method"], r["Metric"], r["N_common"], r["ProxyMean"], r["PaperMean"], r["PaperMinusProxy"], r["Pearson"]] for r in stats],
        ).strip(),
        "",
        "## Interpretation",
        "",
        "The two implementations should agree exactly on Alex. Differences in Aent come from the mapping from normalized entropy variance and local CV into a [0, 1] score. Differences in Aprob are conceptual: the proxy score is dominated by final answer certainty and curve shape, while the paper-formula score is a direct bit-gain rate over the no-reasoning baseline.",
        "",
    ]
    (out_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", type=Path, required=True, help="Existing proxy metrics directory or JSONL")
    parser.add_argument("--paper", type=Path, required=True, help="Paper-formula metrics directory or JSONL")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    proxy_records = read_metrics(args.proxy)
    paper_records = read_metrics(args.paper)
    proxy_summary = summarize(proxy_records)
    paper_summary = summarize(paper_records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "proxy_summary.json").write_text(json.dumps(proxy_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "paper_summary.json").write_text(json.dumps(paper_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_tables(proxy_summary, args.out_dir, "proxy")
    write_summary_tables(paper_summary, args.out_dir, "paper")

    paired = paired_rows(proxy_records, paper_records)
    stats = paired_stats(paired)
    write_csv(
        args.out_dir / "paired_records.csv",
        paired,
        ["sample_idx", "method", *(f"{kind}_{label}" for label in ["Alex", "Aent", "Aprob"] for kind in ["proxy", "paper", "delta"])],
    )
    write_csv(
        args.out_dir / "paired_stats.csv",
        stats,
        ["Method", "Metric", "N_common", "ProxyMean", "PaperMean", "PaperMinusProxy", "Pearson"],
    )
    (args.out_dir / "paired_stats.md").write_text(
        markdown_table(
            ["Method", "Metric", "N", "ProxyMean", "PaperMean", "Paper-Proxy", "Pearson"],
            [[r["Method"], r["Metric"], r["N_common"], r["ProxyMean"], r["PaperMean"], r["PaperMinusProxy"], r["Pearson"]] for r in stats],
        ),
        encoding="utf-8",
    )
    write_report(args.out_dir, proxy_summary, paper_summary, stats)
    print(args.out_dir / "comparison_report.md")


if __name__ == "__main__":
    main()
