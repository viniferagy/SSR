#!/usr/bin/env python3
"""Plot current behavioral zones with the original paper-style layout."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


METHOD_ORDER = ["NEU", "SUP", "AUG-SUP", "SSR"]
CONTROL_ORDER = ["Blind CoT", "+Prob Anchor", "+Traj Anchor", "Response-as-CoT"]
ZONE_KEYS = ["Reason", "Encode", "Monitor", "Copy"]

# Manual figure-tuning map:
# - Panel order / legend order: METHOD_ORDER and CONTROL_ORDER above.
# - Axis titles: X_LABEL and Y_LABEL below.
# - Shared canvas, font, tick, point, and axis-padding sizes: constants below.
# - Zone colors, dashed borders, zone-label placement: decorate_behavior_axis().
# - Method-panel scatter color map and point style: plot_records().
# - Controlled combined marker shapes, canvas size, summary-marker style, legend: plot_combined_records().
# - Runtime knobs for point counts, controlled marker size, legend size, jitter: argparse options in main().
X_LABEL = "Trajectory Anchoring"
Y_LABEL = "Probabilistic Anchoring"

# Shared layout and typography knobs for both method panels and controlled previews.
# Increase PANEL_SIZE for larger subplots; reduce it for a more compact exported figure.
PANEL_SIZE = 3.6

# Font-size knobs. TITLE_SIZE controls panel titles such as NEU/SUP/SSR.
# TICK_SIZE controls numeric tick labels. ZONE_LABEL_SIZE controls Reason/Encode/Monitor/Copy.
# AXIS_LABEL_SIZE controls the y-axis title in each figure. BOTTOM_LABEL_SIZE controls the x-axis title.
TITLE_SIZE = 22
TICK_SIZE = 18
ZONE_LABEL_SIZE = 20
AXIS_LABEL_SIZE = 22
BOTTOM_LABEL_SIZE = 24

# Method-panel scatter style. Controlled combined figures can override point size via CLI
# --controlled-point-size and --controlled-jitter-point-size.
POINT_SIZE = 90
POINT_ALPHA = 0.7

# Tick locations and whitespace around the 0..1 behavioral square.
# DEFAULT_AXIS_PAD is included inside the dashed zone frame.
TICKS = np.linspace(0.0, 1.0, 6)
DEFAULT_AXIS_PAD = 0.12


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
    }
)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise FileNotFoundError(f"No records in {path}")
    return rows


def mean_for_method(rows: Iterable[Dict[str, Any]], method: str, key: str) -> float:
    values = [float(row[key]) for row in rows if row["method"] == method]
    if not values:
        raise KeyError(f"Missing reference method={method!r} for {key!r}")
    return float(np.mean(values))


def reference_anchored_saturation_display(value: float, low: float, mid: float) -> float:
    """Shared display scale: low -> 0, method-middle -> 0.5, derived high -> 1."""
    half_span = max(mid - low, 1e-12)
    high = low + 2.0 * half_span
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def metric_display_reference(
    rows: Iterable[Dict[str, Any]],
    metric: str,
    mid_methods: List[str],
    mid_source: str,
) -> Dict[str, Any]:
    row_list = list(rows)
    low = mean_for_method(row_list, "Blind CoT", metric)
    method_means = {
        method: mean_for_method(row_list, method, metric)
        for method in mid_methods
        if any(row["method"] == method for row in row_list)
    }
    if not method_means:
        raise KeyError(f"No mid reference methods found for {metric!r}")
    values = list(method_means.values())
    if mid_source == "mean":
        mid = float(np.mean(values))
    elif mid_source == "median":
        mid = float(np.median(values))
    else:
        raise ValueError(f"Unsupported mid_source={mid_source!r}")
    if mid <= low:
        raise ValueError(f"Display mid must be larger than low for {metric!r}: low={low}, mid={mid}")
    return {
        "low": low,
        "mid": mid,
        "high": low + 2.0 * (mid - low),
        "mid_source": mid_source,
        "mid_methods": mid_methods,
        "method_means": method_means,
    }


def metric_display(value: float, refs: Dict[str, Any]) -> float:
    return reference_anchored_saturation_display(value, float(refs["low"]), float(refs["mid"]))


def transform_rows(
    rows: List[Dict[str, Any]],
    mid_methods: List[str],
    mid_source: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # The final paper uses question-filtered lexical anchoring. Accept an
    # already-normalized A_lex column for external inputs, but prefer A_lex_QF
    # from the public metric pipeline when both are present.
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("A_lex_QF") not in {None, ""}:
            item["A_lex"] = item["A_lex_QF"]
        normalized_rows.append(item)
    rows = normalized_rows
    refs = {
        "A_prob": metric_display_reference(rows, "A_prob", mid_methods, mid_source),
        "A_lex": metric_display_reference(rows, "A_lex", mid_methods, mid_source),
        "A_traj": metric_display_reference(rows, "A_traj", mid_methods, mid_source),
    }

    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["Aprob_display"] = metric_display(float(row["A_prob"]), refs["A_prob"])
        item["Atraj_display"] = metric_display(float(row["A_traj"]), refs["A_traj"])
        item["Alex_display"] = metric_display(float(row["A_lex"]), refs["A_lex"])
        out.append(item)

    meta = {
        "A_prob_display_reference": refs["A_prob"],
        "A_lex_display_reference": refs["A_lex"],
        "A_traj_display_reference": refs["A_traj"],
    }
    return out, meta


def method_display_centroids(
    rows: List[Dict[str, Any]], method_order: List[str], meta: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = str(row["method"])
        grouped[method]["A_traj"].append(float(row["A_traj"]))
        grouped[method]["A_prob"].append(float(row["A_prob"]))

    centroids: Dict[str, Dict[str, float]] = {}
    for method in method_order:
        item = grouped.get(method)
        if not item:
            continue
        mean_traj = float(np.mean(item["A_traj"]))
        mean_prob = float(np.mean(item["A_prob"]))
        centroids[method] = {
            "x": metric_display(mean_traj, meta["A_traj_display_reference"]),
            "y": metric_display(mean_prob, meta["A_prob_display_reference"]),
        }
    return centroids


def soft_near_centroid_sample(
    x_all: np.ndarray,
    y_all: np.ndarray,
    center: Dict[str, float],
    max_points: int,
    rng: np.random.Generator,
    softness: float,
) -> np.ndarray:
    points = np.column_stack([x_all, y_all])
    center_vec = np.array([center["x"], center["y"]], dtype=float)
    if points.shape[0] <= max_points:
        return np.arange(points.shape[0])

    cov = np.cov(points, rowvar=False)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        cov = np.eye(2)
    cov = cov + np.eye(2) * 1e-4
    try:
        inv_cov = np.linalg.pinv(cov)
        delta = points - center_vec
        dist2 = np.einsum("ij,jk,ik->i", delta, inv_cov, delta)
    except np.linalg.LinAlgError:
        dist2 = np.sum((points - center_vec) ** 2, axis=1)
    dist2 = np.maximum(dist2, 0.0)
    if softness <= 0.0:
        return np.argsort(dist2)[:max_points]

    kth = min(max_points - 1, dist2.size - 1)
    local_scale = float(np.partition(dist2, kth)[kth])
    scale = max(local_scale * max(softness, 1e-6), 1e-9)
    weights = np.exp(-0.5 * dist2 / scale)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        return np.argsort(dist2)[:max_points]
    keep = rng.choice(dist2.size, size=max_points, replace=False, p=weights / np.sum(weights))
    return np.sort(keep)


def stable_method_seed(method: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(method))


def sampled_display_arrays(
    rows: List[Dict[str, Any]],
    method: str,
    centroids: Dict[str, Dict[str, float]],
    max_points: int,
    plot_points: int | None,
    sample_seed: int,
    centroid_softness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_all = np.array([float(r["Atraj_display"]) for r in rows], dtype=float)
    y_all = np.array([float(r["Aprob_display"]) for r in rows], dtype=float)
    c_all = np.array([float(r["Alex_display"]) for r in rows], dtype=float)
    rng = np.random.default_rng(sample_seed + stable_method_seed(method))
    if max_points > 0 and len(rows) > max_points:
        center = centroids.get(method)
        if center is not None:
            keep = soft_near_centroid_sample(
                x_all, y_all, center, max_points, rng, centroid_softness
            )
        else:
            keep = np.sort(rng.choice(len(rows), size=max_points, replace=False))
        x, y, c = x_all[keep], y_all[keep], c_all[keep]
    else:
        x, y, c = x_all, y_all, c_all
    if plot_points is not None and plot_points > 0 and x.size > plot_points:
        plot_keep = np.sort(rng.choice(x.size, size=plot_points, replace=False))
        x, y, c = x[plot_keep], y[plot_keep], c[plot_keep]
    return x, y, c


def decorate_behavior_axis(
    ax: plt.Axes,
    x_thr: float,
    y_thr: float,
    axis_pad: float,
    show_ylabel: bool,
    zone_label_mode: str = "corner",
    tick_size: float = TICK_SIZE,
    zone_label_size: float = ZONE_LABEL_SIZE,
    y_label_size: float = AXIS_LABEL_SIZE,
) -> None:
    # Zone color knobs. These colors are used for dashed borders, zone labels,
    # and the very light background tint in each behavioral region.
    colors = {"Reason": "green", "Encode": "orange", "Monitor": "blue", "Copy": "red"}

    # Axis range knobs. axis_pad controls extra whitespace around the 0..1 square,
    # and that padding is still enclosed by the dashed zone frame.
    axis_min = -axis_pad
    axis_max = 1 + axis_pad
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.tick_params(axis="both", labelsize=tick_size)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Background tint opacity. Increase slightly if the four zones should read more strongly.
    bg_alpha = 0.02
    ax.fill_between([axis_min, x_thr], axis_min, y_thr, color=colors["Reason"], alpha=bg_alpha)
    ax.fill_between([axis_min, x_thr], y_thr, axis_max, color=colors["Encode"], alpha=bg_alpha)
    ax.fill_between([x_thr, axis_max], axis_min, y_thr, color=colors["Monitor"], alpha=bg_alpha)
    ax.fill_between([x_thr, axis_max], y_thr, axis_max, color=colors["Copy"], alpha=bg_alpha)

    # Dashed zone-border knobs. lw controls line thickness; ls controls dash style.
    lw, ls = 5.5, "--"

    # Corner-label placement knobs used when zone_label_mode="corner".
    # Increase label_inset to push labels farther inward from the dashed frame.
    label_inset = max(0.04, axis_pad * 0.7)
    left_label = axis_min + label_inset
    right_label = axis_max - label_inset
    bottom_label = axis_min + label_inset
    top_label = axis_max - label_inset
    ax.plot([axis_min, axis_min, x_thr], [y_thr, axis_min, axis_min], color=colors["Reason"], ls=ls, lw=lw)
    ax.plot([axis_min, axis_min, x_thr], [y_thr, axis_max, axis_max], color=colors["Encode"], ls=ls, lw=lw)
    ax.plot([x_thr, axis_max, axis_max], [axis_min, axis_min, y_thr], color=colors["Monitor"], ls=ls, lw=lw)
    ax.plot([x_thr, axis_max, axis_max], [axis_max, axis_max, y_thr], color=colors["Copy"], ls=ls, lw=lw)
    if zone_label_mode == "center":
        # Center-label placement used by the controlled combined panel.
        # Move these x/y formulas if a zone name overlaps points in the controlled figure.
        ax.text((axis_min + x_thr) / 2, (axis_min + y_thr) / 2, "Reason", color=colors["Reason"], ha="center", va="center", fontsize=zone_label_size, fontweight="bold")
        ax.text((axis_min + x_thr) / 2, (y_thr + axis_max) / 2, "Encode", color=colors["Encode"], ha="center", va="center", fontsize=zone_label_size, fontweight="bold")
        ax.text((x_thr + axis_max) / 2, (axis_min + y_thr) / 2, "Monitor", color=colors["Monitor"], ha="center", va="center", fontsize=zone_label_size, fontweight="bold")
        ax.text((x_thr + axis_max) / 2, (y_thr + axis_max) / 2, "Copy", color=colors["Copy"], ha="center", va="center", fontsize=zone_label_size, fontweight="bold")
    else:
        # Corner-label placement used by the method panels.
        ax.text(left_label, bottom_label, "Reason", color=colors["Reason"], ha="left", va="bottom", fontsize=zone_label_size, fontweight="bold")
        ax.text(left_label, top_label, "Encode", color=colors["Encode"], ha="left", va="top", fontsize=zone_label_size, fontweight="bold")
        ax.text(right_label, bottom_label, "Monitor", color=colors["Monitor"], ha="right", va="bottom", fontsize=zone_label_size, fontweight="bold")
        ax.text(right_label, top_label, "Copy", color=colors["Copy"], ha="right", va="top", fontsize=zone_label_size, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel(Y_LABEL, fontsize=y_label_size, fontweight="bold")


def plot_records(
    records: List[Dict[str, Any]],
    method_order: List[str],
    centroids: Dict[str, Dict[str, float]],
    output_stem: Path,
    x_thr: float,
    y_thr: float,
    max_points: int,
    plot_points: int | None,
    sample_seed: int,
    centroid_softness: float,
    axis_pad: float,
) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["method"])].append(row)

    fig, axes = plt.subplots(1, len(method_order), figsize=(PANEL_SIZE * len(method_order), PANEL_SIZE))
    if len(method_order) == 1:
        axes = [axes]

    # A_lex color-map knobs for method panels. Swap coolwarm for another matplotlib
    # colormap, or change Normalize bounds if the color scale should be less saturated.
    cmap, norm = plt.cm.coolwarm, plt.Normalize(0.0, 1.0)

    for idx, method in enumerate(method_order):
        ax = axes[idx]
        rows = grouped.get(method, [])
        x, y, c = sampled_display_arrays(
            rows, method, centroids, max_points, plot_points, sample_seed, centroid_softness
        )
        # Method-panel point style. POINT_SIZE and POINT_ALPHA are defined near the top.
        ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=POINT_SIZE, alpha=POINT_ALPHA, linewidth=0, rasterized=True)
        center = centroids.get(method)
        if center is not None:
            ax.scatter(
                [float(center["x"])],
                [float(center["y"])],
                marker="*",
                s=260,
                c="white",
                edgecolors="black",
                linewidth=1.2,
                zorder=5,
            )
        ax.set_title(method, fontsize=TITLE_SIZE, fontweight="bold", pad=15)
        decorate_behavior_axis(ax, x_thr, y_thr, axis_pad, show_ylabel=idx == 0)

    fig.text(0.5, -0.05, X_LABEL, ha="center", fontsize=BOTTOM_LABEL_SIZE, fontweight="bold")
    plt.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_combined_records(
    records: List[Dict[str, Any]],
    method_order: List[str],
    centroids: Dict[str, Dict[str, float]],
    output_stem: Path,
    x_thr: float,
    y_thr: float,
    max_points: int,
    plot_points: int | None,
    sample_seed: int,
    centroid_softness: float,
    axis_pad: float,
    point_size: float,
    legend_size: float,
    summary_stat: str,
    jitter: float,
    jitter_point_size: float,
) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["method"])].append(row)

    # Controlled combined canvas size. Reduce these multipliers for a tighter figure;
    # increase them if the legend or axis labels need more breathing room.
    fig, ax = plt.subplots(1, 1, figsize=(PANEL_SIZE * 1.25, PANEL_SIZE * 1.1))

    # A_lex color-map knobs for the controlled combined panel.
    cmap, norm = plt.cm.coolwarm, plt.Normalize(0.0, 1.0)

    # Controlled marker-shape knobs. These are also reflected in the legend handles.
    markers = {
        "Blind CoT": "o",
        "+Prob Anchor": "s",
        "+Traj Anchor": "^",
        "Response-as-CoT": "D",
    }

    for method in method_order:
        rows = grouped.get(method, [])
        if not rows:
            continue
        x, y, c = sampled_display_arrays(
            rows, method, centroids, max_points, plot_points, sample_seed, centroid_softness
        )
        if jitter > 0.0 and x.size:
            # Jitter cloud knobs. jitter controls spread around the sampled points;
            # jitter_point_size and alpha control how visible the background cloud is.
            rng = np.random.default_rng(sample_seed + 100_000 + stable_method_seed(method))
            axis_min = -axis_pad
            axis_max = 1.0 + axis_pad
            x_jitter = np.clip(x + rng.normal(0.0, jitter, size=x.size), axis_min, axis_max)
            y_jitter = np.clip(y + rng.normal(0.0, jitter, size=y.size), axis_min, axis_max)
            ax.scatter(
                x_jitter,
                y_jitter,
                c=c,
                cmap=cmap,
                norm=norm,
                marker=markers.get(method, "o"),
                s=jitter_point_size,
                alpha=0.22,
                linewidth=0,
                rasterized=True,
            )
        alex_values = np.array([float(row["Alex_display"]) for row in rows], dtype=float)
        if summary_stat == "mean":
            summary_color = float(np.mean(alex_values))
        elif summary_stat == "median":
            summary_color = float(np.median(alex_values))
        else:
            raise ValueError(f"Unsupported summary_stat={summary_stat!r}")
        center = centroids.get(method, {"x": float(np.mean(x)), "y": float(np.mean(y))})
        # Controlled summary-marker knobs. point_size comes from --controlled-point-size;
        # edgecolors/linewidth control the black outline around each large marker.
        ax.scatter(
            [float(center["x"])],
            [float(center["y"])],
            c=[summary_color],
            cmap=cmap,
            norm=norm,
            marker=markers.get(method, "o"),
            s=point_size,
            alpha=0.95,
            edgecolors="black",
            linewidth=0.7,
            rasterized=True,
        )

    decorate_behavior_axis(
        ax,
        x_thr,
        y_thr,
        axis_pad,
        show_ylabel=True,
        zone_label_mode="center",
        tick_size=15,
        zone_label_size=17,
        y_label_size=20,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker=markers.get(method, "o"),
            color="none",
            markerfacecolor="0.35",
            markeredgecolor="none",
            markersize=12,
            label=method,
        )
        for method in method_order
        if grouped.get(method)
    ]
    # Controlled legend knobs. bbox_to_anchor moves the whole legend; ncol changes
    # the number of legend columns; legend_size comes from --controlled-legend-size.
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 1.005),
        ncol=2,
        frameon=False,
        fontsize=legend_size,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    # Controlled x-axis title knob. This is separate from BOTTOM_LABEL_SIZE so the
    # compact controlled figure can have a smaller title than the method panels.
    fig.text(0.54, -0.04, X_LABEL, ha="center", fontsize=22, fontweight="bold")
    plt.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(
    rows: List[Dict[str, Any]],
    method_order: List[str],
    centroids: Dict[str, Dict[str, float]],
    x_thr: float,
    y_thr: float,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = str(row["method"])
        grouped[method]["Alex"].append(float(row["A_lex"]))
        grouped[method]["AlexDisplay"].append(float(row["Alex_display"]))
        grouped[method]["Atraj"].append(float(row["A_traj"]))
        grouped[method]["Aprob"].append(float(row["A_prob"]))
        grouped[method]["x"].append(float(row["Atraj_display"]))
        grouped[method]["y"].append(float(row["Aprob_display"]))

    summary: Dict[str, Dict[str, Any]] = {}
    for method in method_order:
        item = grouped.get(method)
        if not item:
            continue
        x = np.array(item["x"], dtype=float)
        y = np.array(item["y"], dtype=float)
        centroid = centroids.get(method, {"x": float(np.mean(x)), "y": float(np.mean(y))})
        summary[method] = {
            "N": int(x.size),
            "metrics": {
                "A_lex": float(np.mean(item["Alex"])),
                "A_lex_display": 100.0 * float(np.mean(item["AlexDisplay"])),
                "A_traj": float(np.mean(item["Atraj"])),
                "A_prob": float(np.mean(item["Aprob"])),
                "Aprob_display": 100.0 * float(centroid["y"]),
                "Atraj_display": 100.0 * float(centroid["x"]),
            },
            "zones": {
                "Reason": 100.0 * float(np.mean((x < x_thr) & (y < y_thr))),
                "Encode": 100.0 * float(np.mean((x < x_thr) & (y >= y_thr))),
                "Monitor": 100.0 * float(np.mean((x >= x_thr) & (y < y_thr))),
                "Copy": 100.0 * float(np.mean((x >= x_thr) & (y >= y_thr))),
            },
        }
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(prefix: str, summary: Dict[str, Dict[str, Any]], method_order: List[str], out_dir: Path) -> None:
    metric_rows = []
    zone_rows = []
    for method in method_order:
        if method not in summary:
            continue
        item = summary[method]
        metric_rows.append({"Method": method, "N": item["N"], **item["metrics"]})
        zone_rows.append({"Method": method, "N": item["N"], **item["zones"]})

    write_csv(
        out_dir / f"{prefix}_display_metrics.csv",
        metric_rows,
        ["Method", "N", "A_lex", "A_lex_display", "A_traj", "A_prob", "Aprob_display", "Atraj_display"],
    )
    write_csv(
        out_dir / f"{prefix}_zones.csv",
        zone_rows,
        ["Method", "N", *ZONE_KEYS],
    )


def filter_methods(rows: List[Dict[str, Any]], order: List[str]) -> List[Dict[str, Any]]:
    wanted = set(order)
    return [row for row in rows if row["method"] in wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="paper_style")
    parser.add_argument("--aprob-k", type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument("--alex-k", type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--atraj-k",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--display-mid-source", choices=["median", "mean"], default="median")
    parser.add_argument(
        "--display-mid-methods",
        default=",".join(METHOD_ORDER),
        help="Comma-separated methods whose metric means define the display middle.",
    )
    parser.add_argument("--x-threshold", type=float, default=0.5)
    parser.add_argument("--y-threshold", type=float, default=0.5)

    # Point-count knobs. max-points first selects near-centroid examples; plot-points
    # then randomly subsamples from that selected set for display.
    parser.add_argument("--max-points-per-method", type=int, default=300)
    parser.add_argument(
        "--plot-points-per-method",
        type=int,
        default=100,
        help="If set, randomly plot this many points after the centroid-near selection.",
    )
    # Controlled-only point-count knob. Use this to keep the combined controlled panel
    # sparse, while leaving method panels at --plot-points-per-method.
    parser.add_argument(
        "--controlled-plot-points-per-method",
        type=int,
        default=None,
        help="Override plotted points per method for the combined controlled panel.",
    )
    parser.add_argument(
        "--combine-controlled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot all controlled baselines in one shared behavioral-zone panel.",
    )

    # Controlled combined visual knobs. These tune only the controlled single-panel figure.
    parser.add_argument("--controlled-point-size", type=float, default=225.0)
    parser.add_argument("--controlled-legend-size", type=float, default=14.0)
    parser.add_argument("--controlled-summary-stat", choices=["median", "mean"], default="median")
    parser.add_argument("--controlled-jitter", type=float, default=0.05)
    parser.add_argument("--controlled-jitter-point-size", type=float, default=68.0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--centroid-softness",
        type=float,
        default=0.0,
        help="Larger values sample a looser cloud around the method-level centroid.",
    )
    parser.add_argument("--axis-pad", type=float, default=DEFAULT_AXIS_PAD)
    args = parser.parse_args()

    mid_methods = [item.strip() for item in args.display_mid_methods.split(",") if item.strip()]
    rows, meta = transform_rows(read_csv(args.csv), mid_methods, args.display_mid_source)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for suffix, order in [("controlled", CONTROL_ORDER), ("methods", METHOD_ORDER)]:
        selected = filter_methods(rows, order)
        centroids = method_display_centroids(selected, order, meta)
        if suffix == "controlled" and args.combine_controlled:
            controlled_plot_points = (
                args.controlled_plot_points_per_method
                if args.controlled_plot_points_per_method is not None
                else args.plot_points_per_method
            )
            plot_combined_records(
                selected,
                order,
                centroids,
                args.out_dir / f"{args.prefix}_{suffix}",
                args.x_threshold,
                args.y_threshold,
                args.max_points_per_method,
                controlled_plot_points,
                args.sample_seed,
                args.centroid_softness,
                args.axis_pad,
                args.controlled_point_size,
                args.controlled_legend_size,
                args.controlled_summary_stat,
                args.controlled_jitter,
                args.controlled_jitter_point_size,
            )
        else:
            plot_records(
                selected,
                order,
                centroids,
                args.out_dir / f"{args.prefix}_{suffix}",
                args.x_threshold,
                args.y_threshold,
                args.max_points_per_method,
                args.plot_points_per_method,
                args.sample_seed,
                args.centroid_softness,
                args.axis_pad,
            )
        write_summary(
            f"{args.prefix}_{suffix}",
            summarize(selected, order, centroids, args.x_threshold, args.y_threshold),
            order,
            args.out_dir,
        )

    report = {
        "source_csv": str(args.csv),
        "x_axis": X_LABEL,
        "y_axis": Y_LABEL,
        "zone_thresholds": {"x": args.x_threshold, "y": args.y_threshold},
        "display_reference": meta,
        "centroid_definition": (
            "x = reference_anchored_saturation_display(mean(A_traj)); "
            "y = reference_anchored_saturation_display(mean(A_prob)); "
            "color = reference_anchored_saturation_display(A_lex)"
        ),
        "point_selection": "deterministic soft near-centroid sample using covariance-shaped distance",
        "centroid_softness": args.centroid_softness,
        "axis_padding": args.axis_pad,
        "outputs": {
            "controlled": str(args.out_dir / f"{args.prefix}_controlled.pdf"),
            "methods": str(args.out_dir / f"{args.prefix}_methods.pdf"),
        },
    }
    (args.out_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.out_dir / f"{args.prefix}_controlled.pdf")
    print(args.out_dir / f"{args.prefix}_methods.pdf")


if __name__ == "__main__":
    main()
