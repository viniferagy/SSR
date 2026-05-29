#!/usr/bin/env python3
"""Build calibrated Aprob variants and plot behavioral-zone figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["NEU", "SUP", "AUG-SUP", "SSR"]
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
        raise FileNotFoundError(f"No metric records under {path_or_dir}")
    return records


def clip01(x: float) -> float:
    if not math.isfinite(x):
        return float("nan")
    return max(0.0, min(1.0, x))


def ncmi(row: Dict[str, Any]) -> float:
    h0 = -float(row["PaperBaseLog2PerToken"])
    gain = float(row["PaperBitGainRate"])
    if h0 <= 1e-12:
        return float("nan")
    return gain / h0


def keyed_r0(records: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for row in records:
        out[int(row["sample_idx"])] = row
    return out


def build_calibrated_records(
    paper_records: List[Dict[str, Any]],
    r0_records: List[Dict[str, Any]] | None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    r0_by_sample = keyed_r0(r0_records or [])
    cal_records: List[Dict[str, Any]] = []
    excess_records: List[Dict[str, Any]] = []
    for row in paper_records:
        out = dict(row)
        score = clip01(ncmi(row))
        out["AprobNCMI"] = ncmi(row)
        out["Aprob_cal"] = score
        out["ProbabilisticAnchoring"] = score
        cal_records.append(out)

        if r0_by_sample:
            sample_idx = int(row["sample_idx"])
            r0 = r0_by_sample.get(sample_idx)
            if not r0:
                continue
            base = clip01(ncmi(r0))
            denom = max(1.0 - base, 1e-12)
            excess = clip01((score - base) / denom)
            ex = dict(row)
            ex["AprobNCMI"] = ncmi(row)
            ex["R0_AprobNCMI"] = ncmi(r0)
            ex["R0_Aprob_cal"] = base
            ex["Aprob_excess"] = excess
            ex["ProbabilisticAnchoring"] = excess
            excess_records.append(ex)
    return cal_records, excess_records


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        method = row["method"]
        values[method]["Alex"].append(float(row["lexical_anchoring"]))
        values[method]["Aent"].append(float(row["entropy_anchoring"]))
        values[method]["Aprob"].append(float(row["ProbabilisticAnchoring"]))
    summary: Dict[str, Any] = {}
    for method in METHOD_ORDER:
        item = values.get(method)
        if not item:
            continue
        x = np.array(item["Aent"], dtype=float)
        y = np.array(item["Aprob"], dtype=float)
        zones = {
            "Reason": 100.0 * float(np.mean((x < 0.5) & (y < 0.5))),
            "Encode": 100.0 * float(np.mean((x < 0.5) & (y >= 0.5))),
            "Cloze": 100.0 * float(np.mean((x >= 0.5) & (y < 0.5))),
            "Copy": 100.0 * float(np.mean((x >= 0.5) & (y >= 0.5))),
        }
        summary[method] = {
            "N": len(item["Alex"]),
            "metrics": {
                "Alex": 100.0 * float(np.mean(item["Alex"])),
                "Aent": 100.0 * float(np.mean(item["Aent"])),
                "Aprob": 100.0 * float(np.mean(item["Aprob"])),
            },
            "zones": zones,
        }
    return summary


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.1f}"
        return str(v)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out) + "\n"


def write_summary(prefix: str, summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_rows = []
    zone_rows = []
    for method in METHOD_ORDER:
        if method not in summary:
            continue
        item = summary[method]
        metric_rows.append({
            "Method": method,
            "N": item["N"],
            "Alex": item["metrics"]["Alex"],
            "Aent": item["metrics"]["Aent"],
            "Aprob": item["metrics"]["Aprob"],
        })
        zone_rows.append({
            "Method": method,
            "N": item["N"],
            **{k: item["zones"][k] for k in ZONE_KEYS},
        })
    with (out_dir / f"{prefix}_table1_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "N", "Alex", "Aent", "Aprob"])
        writer.writeheader()
        writer.writerows(metric_rows)
    with (out_dir / f"{prefix}_table2_zones.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "N", *ZONE_KEYS])
        writer.writeheader()
        writer.writerows(zone_rows)
    (out_dir / f"{prefix}_table1_metrics.md").write_text(
        markdown_table(["Method", "N", "Alex", "Aent", "Aprob"], [[r["Method"], r["N"], r["Alex"], r["Aent"], r["Aprob"]] for r in metric_rows]),
        encoding="utf-8",
    )
    (out_dir / f"{prefix}_table2_zones.md").write_text(
        markdown_table(["Method", "N", *ZONE_KEYS], [[r["Method"], r["N"], *(r[k] for k in ZONE_KEYS)] for r in zone_rows]),
        encoding="utf-8",
    )


def plot_records(records: List[Dict[str, Any]], title_suffix: str, output: Path) -> None:
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
        ax.fill_between([0, 0.5], 0, 0.5, color=colors["Reason"], alpha=bg_alpha)
        ax.fill_between([0, 0.5], 0.5, 1, color=colors["Encode"], alpha=bg_alpha)
        ax.fill_between([0.5, 1], 0, 0.5, color=colors["Cloze"], alpha=bg_alpha)
        ax.fill_between([0.5, 1], 0.5, 1, color=colors["Copy"], alpha=bg_alpha)
        lw, ls, pad = 3.5, "--", 0.08
        ax.plot([0, 0, 0.5], [0.5, 0, 0], color=colors["Reason"], ls=ls, lw=lw, clip_on=False)
        ax.text(pad, pad, "Reason", color=colors["Reason"], ha="left", va="bottom", fontsize=20, fontweight="bold")
        ax.plot([0, 0, 0.5], [0.5, 1, 1], color=colors["Encode"], ls=ls, lw=lw, clip_on=False)
        ax.text(pad, 1 - pad, "Encode", color=colors["Encode"], ha="left", va="top", fontsize=20, fontweight="bold")
        ax.plot([0.5, 1, 1], [0, 0, 0.5], color=colors["Cloze"], ls=ls, lw=lw, clip_on=False)
        ax.text(1 - pad, pad, "Cloze", color=colors["Cloze"], ha="right", va="bottom", fontsize=20, fontweight="bold")
        ax.plot([0.5, 1, 1], [1, 1, 0.5], color=colors["Copy"], ls=ls, lw=lw, clip_on=False)
        ax.text(1 - pad, 1 - pad, "Copy", color=colors["Copy"], ha="right", va="top", fontsize=20, fontweight="bold")
        if idx == 0:
            ax.set_ylabel(title_suffix, fontsize=22, fontweight="bold")
    fig.text(0.5, -0.05, "Entropic Anchoring", ha="center", fontsize=24, fontweight="bold")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(out_dir: Path, cal_summary: Dict[str, Any], excess_summary: Dict[str, Any] | None) -> None:
    parts = [
        "# Calibrated Aprob Comparison",
        "",
        "Definitions:",
        "",
        "- `Aprob_cal = clip(I(R;A|Q) / H(A|Q), 0, 1)`.",
        "- `Aprob_excess = clip((Aprob_cal(R) - Aprob_cal(R0)) / (1 - Aprob_cal(R0)), 0, 1)`.",
        "- `R0` is answer-blind reasoning generated from Q only.",
        "",
        "## Aprob_cal",
        "",
        (out_dir / "aprob_cal_table1_metrics.md").read_text(encoding="utf-8").strip(),
        "",
        (out_dir / "aprob_cal_table2_zones.md").read_text(encoding="utf-8").strip(),
    ]
    if excess_summary is not None:
        parts.extend([
            "",
            "## Aprob_excess",
            "",
            (out_dir / "aprob_excess_table1_metrics.md").read_text(encoding="utf-8").strip(),
            "",
            (out_dir / "aprob_excess_table2_zones.md").read_text(encoding="utf-8").strip(),
        ])
    (out_dir / "calibrated_aprob_report.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-metrics", type=Path, required=True)
    parser.add_argument("--r0-metrics", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    paper_records = read_metrics(args.paper_metrics)
    r0_records = read_metrics(args.r0_metrics) if args.r0_metrics else None
    cal_records, excess_records = build_calibrated_records(paper_records, r0_records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "aprob_cal.metrics.jsonl", cal_records)
    cal_summary = summarize(cal_records)
    write_summary("aprob_cal", cal_summary, args.out_dir)
    plot_records(cal_records, "Aprob_cal", args.out_dir / "aprob_cal_figure4_methods.pdf")

    excess_summary = None
    if excess_records:
        write_jsonl(args.out_dir / "aprob_excess.metrics.jsonl", excess_records)
        excess_summary = summarize(excess_records)
        write_summary("aprob_excess", excess_summary, args.out_dir)
        plot_records(excess_records, "Aprob_excess", args.out_dir / "aprob_excess_figure4_methods.pdf")

    write_report(args.out_dir, cal_summary, excess_summary)
    print(args.out_dir / "calibrated_aprob_report.md")


if __name__ == "__main__":
    main()
