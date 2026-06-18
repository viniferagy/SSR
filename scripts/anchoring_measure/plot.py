#!/usr/bin/env python3
"""Plot probabilistic anchoring figures and tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["NEU", "SUP", "AUG-SUP", "SSR"]
CONTROL_ORDER = ["Real CoT", "+Prob Anchor", "+Entropy Anchor", "Response as CoT"]
ZONE_KEYS = ["Reason", "Encode", "Cloze", "Copy"]
Y_LABEL = "Probabilistic Anchoring"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("metrics_rank*.jsonl"))
    if not files and path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    rows: List[Dict[str, Any]] = []
    for item in files:
        with item.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No records in {path}")
    return rows


def scale_aprob(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        for key in ["Aprob", "entropy_anchoring", "lexical_anchoring", "method"]:
            if key not in row:
                raise KeyError(f"Metric record is missing required field: {key}")
        item = dict(row)
        y = float(row["Aprob"])
        item["Aprob"] = float(np.sqrt(np.clip(y, 0.0, 1.0)))
        out.append(item)
    return out


def zone_thresholds(records: List[Dict[str, Any]], quantile: float) -> tuple[float, float, Dict[str, float]]:
    x = np.array([float(r["entropy_anchoring"]) for r in records], dtype=float)
    y = np.array([float(r["Aprob"]) for r in records], dtype=float)
    x_thr = float(np.quantile(x[np.isfinite(x)], quantile))
    y_thr = float(np.quantile(y[np.isfinite(y)], quantile))
    return x_thr, y_thr, {
        "quantile": float(quantile),
        "x_threshold": x_thr,
        "y_threshold": y_thr,
    }


def infer_method_order(records: List[Dict[str, Any]]) -> List[str]:
    methods = {str(row["method"]) for row in records}
    if methods.issubset(set(METHOD_ORDER)):
        return [method for method in METHOD_ORDER if method in methods]
    if methods.issubset(set(CONTROL_ORDER)):
        return [method for method in CONTROL_ORDER if method in methods]
    known = METHOD_ORDER + CONTROL_ORDER
    ordered = [method for method in known if method in methods]
    ordered.extend(sorted(methods.difference(ordered)))
    return ordered


def plot_records(
    records: List[Dict[str, Any]],
    method_order: List[str],
    ylabel: str,
    output_stem: Path,
    x_thr: float,
    y_thr: float,
) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["method"]].append(row)

    fig, axes = plt.subplots(1, len(method_order), figsize=(4.3 * len(method_order), 4.3))
    if len(method_order) == 1:
        axes = [axes]
    colors = {"Reason": "green", "Encode": "orange", "Cloze": "blue", "Copy": "red"}
    cmap, norm = plt.cm.coolwarm, plt.Normalize(0.0, 1.0)
    for idx, method in enumerate(method_order):
        ax = axes[idx]
        rows = grouped.get(method, [])
        x = np.array([r["entropy_anchoring"] for r in rows], dtype=float)
        y = np.array([r["Aprob"] for r in rows], dtype=float)
        c = np.array([r["lexical_anchoring"] for r in rows], dtype=float)
        ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=60, alpha=0.7, rasterized=True)
        ax.set_title(method, fontsize=22, fontweight="bold", pad=15)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=18)
        for spine in ax.spines.values():
            spine.set_visible(False)

        bg_alpha = 0.02
        ax.fill_between([0, x_thr], 0, y_thr, color=colors["Reason"], alpha=bg_alpha)
        ax.fill_between([0, x_thr], y_thr, 1, color=colors["Encode"], alpha=bg_alpha)
        ax.fill_between([x_thr, 1], 0, y_thr, color=colors["Cloze"], alpha=bg_alpha)
        ax.fill_between([x_thr, 1], y_thr, 1, color=colors["Copy"], alpha=bg_alpha)

        lw, ls, pad = 3.5, "--", 0.08
        ax.plot([0, 0, x_thr], [y_thr, 0, 0], color=colors["Reason"], ls=ls, lw=lw, clip_on=False)
        ax.text(pad, pad, "Reason", color=colors["Reason"], ha="left", va="bottom", fontsize=20, fontweight="bold")
        ax.plot([0, 0, x_thr], [y_thr, 1, 1], color=colors["Encode"], ls=ls, lw=lw, clip_on=False)
        ax.text(pad, 1 - pad, "Encode", color=colors["Encode"], ha="left", va="top", fontsize=20, fontweight="bold")
        ax.plot([x_thr, 1, 1], [0, 0, y_thr], color=colors["Cloze"], ls=ls, lw=lw, clip_on=False)
        ax.text(1 - pad, pad, "Cloze", color=colors["Cloze"], ha="right", va="bottom", fontsize=20, fontweight="bold")
        ax.plot([x_thr, 1, 1], [1, 1, y_thr], color=colors["Copy"], ls=ls, lw=lw, clip_on=False)
        ax.text(1 - pad, 1 - pad, "Copy", color=colors["Copy"], ha="right", va="top", fontsize=20, fontweight="bold")
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=22, fontweight="bold")

    fig.text(0.5, -0.05, "Entropic Anchoring", ha="center", fontsize=24, fontweight="bold")
    plt.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(rows: List[Dict[str, Any]], method_order: List[str], x_thr: float, y_thr: float) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = row["method"]
        grouped[method]["Alex"].append(float(row["lexical_anchoring"]))
        grouped[method]["Aent"].append(float(row["entropy_anchoring"]))
        grouped[method]["Aprob"].append(float(row["Aprob"]))
    summary: Dict[str, Dict[str, Any]] = {}
    for method in method_order:
        item = grouped.get(method)
        if not item:
            continue
        x = np.array(item["Aent"], dtype=float)
        y = np.array(item["Aprob"], dtype=float)
        vals = np.array(item["Aprob"], dtype=float)
        summary[method] = {
            "N": int(vals.size),
            "metrics": {
                "Alex": 100.0 * float(np.mean(item["Alex"])),
                "Aent": 100.0 * float(np.mean(item["Aent"])),
                "Aprob": 100.0 * float(np.mean(vals)),
            },
            "zones": {
                "Reason": 100.0 * float(np.mean((x < x_thr) & (y < y_thr))),
                "Encode": 100.0 * float(np.mean((x < x_thr) & (y >= y_thr))),
                "Cloze": 100.0 * float(np.mean((x >= x_thr) & (y < y_thr))),
                "Copy": 100.0 * float(np.mean((x >= x_thr) & (y >= y_thr))),
            },
        }
    return summary


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out) + "\n"


def write_summary_tables(
    prefix: str,
    stem: str,
    summary: Dict[str, Dict[str, Any]],
    method_order: List[str],
    out_dir: Path,
) -> None:
    metric_rows = []
    zone_rows = []
    for method in method_order:
        if method not in summary:
            continue
        item = summary[method]
        metric_rows.append(
            {
                "Method": method,
                "N": item["N"],
                "Alex": item["metrics"]["Alex"],
                "Aent": item["metrics"]["Aent"],
                "Aprob": item["metrics"]["Aprob"],
            }
        )
        zone_rows.append(
            {
                "Method": method,
                "N": item["N"],
                **{k: item["zones"][k] for k in ZONE_KEYS},
            }
        )

    with (out_dir / f"{stem}_table1_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "N", "Alex", "Aent", "Aprob"])
        writer.writeheader()
        writer.writerows(metric_rows)
    with (out_dir / f"{stem}_table2_zones.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "N", *ZONE_KEYS])
        writer.writeheader()
        writer.writerows(zone_rows)

    (out_dir / f"{stem}_table1_metrics.md").write_text(
        markdown_table(
            ["Method", "N", "Alex", "Aent", "Aprob"],
            [[r["Method"], r["N"], r["Alex"], r["Aent"], r["Aprob"]] for r in metric_rows],
        ),
        encoding="utf-8",
    )
    (out_dir / f"{stem}_table2_zones.md").write_text(
        markdown_table(
            ["Method", "N", *ZONE_KEYS],
            [[r["Method"], r["N"], *(r[k] for k in ZONE_KEYS)] for r in zone_rows],
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--zone-quantile", type=float, default=0.6)
    args = parser.parse_args()

    rows = scale_aprob(read_jsonl(args.metrics))
    method_order = infer_method_order(rows)
    x_thr, y_thr, threshold_meta = zone_thresholds(rows, args.zone_quantile)
    plot_stem = args.out_dir / args.prefix
    plot_records(rows, method_order, Y_LABEL, plot_stem, x_thr, y_thr)
    summary = summarize(rows, method_order, x_thr, y_thr)
    write_summary_tables(args.prefix, args.prefix, summary, method_order, args.out_dir)

    report_path = args.out_dir / f"{args.prefix}_summary.json"
    report = {
        "source_metrics": str(args.metrics),
        "method_order": method_order,
        "zone_threshold": threshold_meta,
        "summary": summary,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
