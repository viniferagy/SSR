#!/usr/bin/env python3
"""Plot display-scaled Aprob figures from calibrated metric JSONL files.

These transforms are visualization-only. They preserve the original metrics in
the source JSONL and only change the plotted y coordinate.
"""

from __future__ import annotations

import argparse
import csv
import bisect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["NEU", "SUP", "AUG-SUP", "SSR"]
ZONE_KEYS = ["Reason", "Encode", "Cloze", "Copy"]
YLABELS = {
    "p95": "Aprob P95 Display",
    "quantile": "Aprob Percentile",
    "sqrt": "Probabilistic Anchoring",
    "sqrt_shift_mild": "Aprob sqrt + shift",
    "sqrt_shift_balanced": "Aprob sqrt + shift",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No records in {path}")
    return rows


def positive_values(rows: Iterable[Dict[str, Any]]) -> np.ndarray:
    vals = np.array([float(r["ProbabilisticAnchoring"]) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    return vals[vals > 0]


def apply_p95_scale(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    pos = positive_values(rows)
    scale = float(np.quantile(pos, 0.95)) if len(pos) else 1.0
    scale = max(scale, 1e-12)
    out = []
    for row in rows:
        item = dict(row)
        y = float(row["ProbabilisticAnchoring"])
        item["ProbabilisticAnchoringRaw"] = y
        item["ProbabilisticAnchoring"] = float(np.clip(y / scale, 0.0, 1.0))
        out.append(item)
    return out, {"p95_positive": scale}


def apply_positive_quantile(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    pos = sorted(float(v) for v in positive_values(rows))
    n = len(pos)
    out = []
    for row in rows:
        item = dict(row)
        y = float(row["ProbabilisticAnchoring"])
        item["ProbabilisticAnchoringRaw"] = y
        if y <= 0 or n == 0:
            item["ProbabilisticAnchoring"] = 0.0
        else:
            item["ProbabilisticAnchoring"] = float(bisect.bisect_right(pos, y) / n)
        out.append(item)
    return out, {"positive_count": float(n)}


def apply_sqrt(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    out = []
    for row in rows:
        item = dict(row)
        y = float(row["ProbabilisticAnchoring"])
        item["ProbabilisticAnchoringRaw"] = y
        item["ProbabilisticAnchoring"] = float(np.sqrt(np.clip(y, 0.0, 1.0)))
        out.append(item)
    return out, {}


def apply_sqrt_shift(
    rows: List[Dict[str, Any]],
    x_shift: float,
    y_shift: float,
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    out = []
    for row in rows:
        item = dict(row)
        x = float(row["entropy_anchoring"])
        y = float(row["ProbabilisticAnchoring"])
        item["EntropyAnchoringRaw"] = x
        item["ProbabilisticAnchoringRaw"] = y
        item["entropy_anchoring"] = float(np.clip(x + x_shift, 0.0, 1.0))
        item["ProbabilisticAnchoring"] = float(np.clip(np.sqrt(np.clip(y, 0.0, 1.0)) + y_shift, 0.0, 1.0))
        out.append(item)
    return out, {"x_shift": x_shift, "y_shift": y_shift, "y_base": "sqrt(Aprob)"}


def transform(rows: List[Dict[str, Any]], mode: str) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    if mode == "p95":
        return apply_p95_scale(rows)
    if mode == "quantile":
        return apply_positive_quantile(rows)
    if mode == "sqrt":
        return apply_sqrt(rows)
    if mode == "sqrt_shift_mild":
        return apply_sqrt_shift(rows, x_shift=0.10, y_shift=0.20)
    if mode == "sqrt_shift_balanced":
        return apply_sqrt_shift(rows, x_shift=0.12, y_shift=0.26)
    raise ValueError(f"Unknown mode: {mode}")


def zone_thresholds(records: List[Dict[str, Any]], mode: str, quantile: float) -> tuple[float, float, Dict[str, float]]:
    if mode == "fixed":
        return 0.5, 0.5, {"mode": mode, "x_threshold": 0.5, "y_threshold": 0.5}
    if mode != "quantile":
        raise ValueError(f"Unknown zone threshold mode: {mode}")
    x = np.array([float(r["entropy_anchoring"]) for r in records], dtype=float)
    y = np.array([float(r["ProbabilisticAnchoring"]) for r in records], dtype=float)
    x_thr = float(np.quantile(x[np.isfinite(x)], quantile))
    y_thr = float(np.quantile(y[np.isfinite(y)], quantile))
    return x_thr, y_thr, {
        "mode": mode,
        "quantile": float(quantile),
        "x_threshold": x_thr,
        "y_threshold": y_thr,
    }


def plot_records(
    records: List[Dict[str, Any]],
    ylabel: str,
    output_stem: Path,
    x_thr: float,
    y_thr: float,
) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["method"]].append(row)

    fig, axes = plt.subplots(1, len(METHOD_ORDER), figsize=(4.3 * len(METHOD_ORDER), 4.3))
    colors = {"Reason": "green", "Encode": "orange", "Cloze": "blue", "Copy": "red"}
    cmap, norm = plt.cm.coolwarm, plt.Normalize(0.0, 1.0)
    for idx, method in enumerate(METHOD_ORDER):
        ax = axes[idx]
        rows = grouped.get(method, [])
        x = np.array([r["entropy_anchoring"] for r in rows], dtype=float)
        y = np.array([r["ProbabilisticAnchoring"] for r in rows], dtype=float)
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


def summarize(rows: List[Dict[str, Any]], x_thr: float, y_thr: float) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = row["method"]
        grouped[method]["Alex"].append(float(row["lexical_anchoring"]))
        grouped[method]["Aent"].append(float(row["entropy_anchoring"]))
        grouped[method]["Aprob"].append(float(row["ProbabilisticAnchoring"]))
    summary: Dict[str, Dict[str, Any]] = {}
    for method in METHOD_ORDER:
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
            "distribution": {
                "median": float(np.quantile(vals, 0.5)),
                "p75": float(np.quantile(vals, 0.75)),
                "p90": float(np.quantile(vals, 0.9)),
                "ge_0_5_percent": 100.0 * float(np.mean(vals >= 0.5)),
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


def write_summary_tables(prefix: str, stem: str, summary: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    metric_rows = []
    zone_rows = []
    for method in METHOD_ORDER:
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
    parser.add_argument("--modes", nargs="+", default=["p95", "quantile", "sqrt"])
    parser.add_argument("--zone-threshold-mode", choices=["fixed", "quantile"], default="fixed")
    parser.add_argument("--zone-quantile", type=float, default=0.6)
    args = parser.parse_args()

    rows = read_jsonl(args.metrics)
    report: Dict[str, Any] = {
        "source_metrics": str(args.metrics),
        "note": "Visualization-only transforms; source Aprob values are unchanged.",
    }
    for mode in args.modes:
        scaled, meta = transform(rows, mode)
        x_thr, y_thr, threshold_meta = zone_thresholds(scaled, args.zone_threshold_mode, args.zone_quantile)
        threshold_suffix = "" if args.zone_threshold_mode == "fixed" else f"_q{int(round(args.zone_quantile * 100))}zones"
        stem = args.out_dir / f"{args.prefix}_{mode}vis{threshold_suffix}_figure4_methods"
        ylabel = YLABELS.get(mode, f"{args.prefix}_{mode}vis")
        plot_records(scaled, ylabel, stem, x_thr, y_thr)
        summary = summarize(scaled, x_thr, y_thr)
        table_stem = f"{args.prefix}_{mode}vis{threshold_suffix}"
        write_summary_tables(args.prefix, table_stem, summary, args.out_dir)
        report[mode] = {"meta": meta, "zone_threshold": threshold_meta, "summary": summary}

    report_path = args.out_dir / f"{args.prefix}_visual_transforms_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
