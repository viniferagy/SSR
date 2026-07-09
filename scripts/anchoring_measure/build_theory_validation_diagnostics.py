#!/usr/bin/env python3
"""Build lightweight diagnostics for the SSR theory appendix.

The outputs connect the skeleton-channel bound to observable paper diagnostics:
- lexical leakage in generated SSR skeleton summaries;
- answer-swap sensitivity for the final trace;
- A_prob clipping-boundary rates from raw probability shards.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


STOPWORDS = set(
    """
    a an the and or but if then else while of in on at to for from by with
    without about into through during before after above below up down out over
    under again further here there when where why how all any both each few more
    most other some such no nor not only own same so than too very can will just
    don should now is are was were be been being have has had do does did as it
    its this that these those i you he she they we my your his her their our me
    him them us what which who whom whose
    """.split()
)
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def toks(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOPWORDS and len(t) > 1]


def skeleton_content(text: str) -> str:
    lines: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+\.\s*", "", line)
        line = re.sub(r"^\[[A-Z-]+\]\s*", "", line)
        line = re.sub(r"^\[(HIGH|LOW)\]\s*", "", line)
        line = re.sub(r"\[(HIGH|LOW)\]", "", line)
        lines.append(line.strip())
    return "\n".join(lines)


def content_recall(trace: str, answer: str, question: str | None, idf: dict[str, float]) -> float:
    trace_counts = Counter(toks(trace))
    answer_tokens = toks(answer)
    if question:
        remaining_question = Counter(toks(question))
        filtered = []
        for token in answer_tokens:
            if remaining_question[token] > 0:
                remaining_question[token] -= 1
            else:
                filtered.append(token)
        answer_tokens = filtered

    answer_counts = Counter(answer_tokens)
    denom = 0.0
    numer = 0.0
    for token, count in answer_counts.items():
        weight = idf.get(token, 1.0)
        denom += weight * count
        numer += weight * min(trace_counts.get(token, 0), count)
    return 0.0 if denom <= 0 else numer / denom


def build_idf(rows: list[dict[str, Any]], method: str) -> dict[str, float]:
    df: Counter[str] = Counter()
    n_docs = 0
    for row in rows:
        answer = str(row.get("answers", {}).get(method, ""))
        doc_tokens = set(toks(answer))
        if not doc_tokens:
            continue
        n_docs += 1
        for token in doc_tokens:
            df[token] += 1
    return {token: math.log((1 + n_docs) / (1 + count)) + 1 for token, count in df.items()}


def skeleton_leakage(input_path: Path, method: str) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    idf = build_idf(rows, method)
    records: list[dict[str, float]] = []
    for idx, row in enumerate(rows):
        skeleton = str(row.get("generated_skeletons", {}).get(method, ""))
        if not skeleton.strip():
            continue
        question = str(row.get("questions", {}).get(method, ""))
        answer = str(row.get("answers", {}).get(method, ""))
        reasoning = str(row.get("reasonings", {}).get(method, ""))
        content = skeleton_content(skeleton)
        lines = [line for line in skeleton.splitlines() if line.strip()]
        records.append(
            {
                "sample_idx": float(idx),
                "skeleton_lines": float(len(lines)),
                "high_lines": float(sum(1 for line in lines if "[HIGH]" in line)),
                "skeleton_tokens": float(len(toks(content))),
                "skeleton_Alex_QF": content_recall(content, answer, question, idf),
                "reason_Alex_QF": content_recall(reasoning, answer, question, idf),
                "skeleton_Alex": content_recall(content, answer, None, idf),
                "reason_Alex": content_recall(reasoning, answer, None, idf),
            }
        )
    return {
        "N": len(records),
        "Avg_skeleton_lines": mean(r["skeleton_lines"] for r in records),
        "Avg_HIGH_lines": mean(r["high_lines"] for r in records),
        "Avg_skeleton_tokens": mean(r["skeleton_tokens"] for r in records),
        "Skeleton_Alex_QF_pct": 100.0 * mean(r["skeleton_Alex_QF"] for r in records),
        "Skeleton_Alex_QF_median_pct": 100.0 * median(r["skeleton_Alex_QF"] for r in records),
        "SSR_reason_Alex_QF_pct_same_input": 100.0 * mean(r["reason_Alex_QF"] for r in records),
        "Skeleton_Alex_pct": 100.0 * mean(r["skeleton_Alex"] for r in records),
        "SSR_reason_Alex_pct_same_input": 100.0 * mean(r["reason_Alex"] for r in records),
    }


def prob_boundaries(prob_dir: Path) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in prob_dir.glob("prob_metrics_rank*.jsonl"):
        for row in read_jsonl(path):
            method = str(row.get("method", ""))
            if method in {"NEU", "SUP", "AUG-SUP", "SSR"}:
                by_method[method].append(row)

    rows: list[dict[str, Any]] = []
    for method in ["NEU", "SUP", "AUG-SUP", "SSR"]:
        vals = by_method[method]
        b100 = [float(row["B_100"]) for row in vals]
        aprob = [float(row["Aprob"]) for row in vals]
        rows.append(
            {
                "Method": method,
                "N": len(vals),
                "Aprob_clipped_mean_pct": 100.0 * mean(aprob),
                "B100_mean": mean(b100),
                "B100_median": median(b100),
                "Negative_B100_rate": mean(x < 0 for x in b100),
                "Aprob_saturated_rate": mean(x >= 0.999999 for x in aprob),
            }
        )
    return rows


def answer_swap(summary_path: Path, deltas_path: Path) -> list[dict[str, Any]]:
    keep_metrics = ["word_rouge_l_f1_sensitivity", "content_jaccard_sensitivity", "embedding_sensitivity"]
    means: dict[tuple[str, str], float] = {}
    with summary_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["metric"] in keep_metrics and row["method"] in {"NEU", "SUP", "AUG-SUP", "SSR"}:
                means[(row["metric"], row["method"])] = float(row["mean"])

    deltas: dict[str, dict[str, str]] = {}
    with deltas_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["metric"] in keep_metrics and row["delta"] == "SSR-NEU":
                deltas[row["metric"]] = row

    rows: list[dict[str, Any]] = []
    for metric in keep_metrics:
        delta = deltas.get(metric, {})
        rows.append(
            {
                "metric": metric,
                "NEU": means[(metric, "NEU")],
                "SUP": means[(metric, "SUP")],
                "AUG-SUP": means[(metric, "AUG-SUP")],
                "SSR": means[(metric, "SSR")],
                "SSR_minus_NEU": float(delta.get("mean_delta", "nan")),
                "ci95_low": float(delta.get("ci95_low", "nan")),
                "ci95_high": float(delta.get("ci95_high", "nan")),
                "ci_excludes_zero": delta.get("ci_excludes_zero", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]] | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_list = [rows] if isinstance(rows, dict) else rows
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_list[0]))
        writer.writeheader()
        for row in row_list:
            writer.writerow({key: f"{value:.6f}" if isinstance(value, float) else value for key, value in row.items()})


def write_report(path: Path, skel: dict[str, Any], swap: list[dict[str, Any]], prob: list[dict[str, Any]]) -> None:
    lines = [
        "# Theory Validation Diagnostics",
        "",
        "Source-backed lightweight diagnostics for the theory-practice bridge in the SSR appendix.",
        "",
        "## Skeleton Leakage",
        "",
        "| N | Avg lines | Avg HIGH | Avg skeleton tokens | Skeleton A_lex_QF (%) | Median skeleton A_lex_QF (%) | Final SSR reason A_lex_QF on same input (%) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {skel['N']} | {skel['Avg_skeleton_lines']:.2f} | {skel['Avg_HIGH_lines']:.2f} | {skel['Avg_skeleton_tokens']:.1f} | {skel['Skeleton_Alex_QF_pct']:.3f} | {skel['Skeleton_Alex_QF_median_pct']:.3f} | {skel['SSR_reason_Alex_QF_pct_same_input']:.3f} |",
        "",
        "## Answer-Swap Sensitivity",
        "",
        "| Metric | NEU | SUP | AUG-SUP | SSR | SSR-NEU | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in swap:
        lines.append(
            f"| {row['metric']} | {row['NEU']:.4f} | {row['SUP']:.4f} | {row['AUG-SUP']:.4f} | "
            f"{row['SSR']:.4f} | {row['SSR_minus_NEU']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
        )
    lines += [
        "",
        "## Probabilistic Boundary Diagnostics",
        "",
        "| Method | N | clipped A_prob (%) | B100 mean | B100 median | B100 < 0 | Aprob = 1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in prob:
        lines.append(
            f"| {row['Method']} | {row['N']} | {row['Aprob_clipped_mean_pct']:.3f} | {row['B100_mean']:.3f} | "
            f"{row['B100_median']:.3f} | {100.0 * row['Negative_B100_rate']:.1f}% | {100.0 * row['Aprob_saturated_rate']:.1f}% |"
        )
    lines += [
        "",
        "Note: raw prob files store clipped normalized Aprob and unnormalized B100. B100<0 identifies examples whose trace increases answer surprisal; Aprob=1 identifies upper-bound saturation. The denominator needed to reconstruct every pre-clipped normalized value is not stored in this run.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--ssr-input", type=Path, default=None)
    parser.add_argument("--method", default="SSR_PLUS_STRUCT_BALANCED_DERIVATIONAL_R")
    args = parser.parse_args()

    root = args.root.resolve()
    run = root / "runs/paper_fixed_method_set_qwen3_4b/n1000"
    out_dir = args.out_dir or (run / "theory_validation")
    ssr_input = args.ssr_input or (root / "runs/ssr_plus_struct_balanced_derivational_r_full1k/inputs/balanced_n1000.metric.jsonl")

    skel = skeleton_leakage(ssr_input, args.method)
    swap = answer_swap(run / "answer_swap/results/summary_by_method.csv", run / "answer_swap/results/paired_deltas_vs_neu.csv")
    prob = prob_boundaries(run / "prob_metrics")

    write_csv(out_dir / "skeleton_leakage_summary.csv", skel)
    write_csv(out_dir / "answer_swap_theory_summary.csv", swap)
    write_csv(out_dir / "prob_boundary_diagnostics.csv", prob)
    write_report(out_dir / "theory_validation_report.md", skel, swap, prob)
    print(f"Wrote theory validation diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
