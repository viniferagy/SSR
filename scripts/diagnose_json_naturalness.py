#!/usr/bin/env python3
"""Recompute JSON-vs-natural-language naturalness diagnostics.

This script is intentionally lightweight: it reads existing metric JSONL files,
extracts the stored reasoning traces, and reports paragraph count,
one-sentence paragraph rate, and a reference-centered Template Rigidity Index
(TRI) used for the JSON naturalness ablation provenance record.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    {
        "variant": "JSON template + renderer",
        "method": "SSR_PLUS_STRUCT_BALANCED_JSON",
        "n_label": 128,
        "metric_jsonl": ROOT
        / "runs/ssr_plus_struct_balanced/attempt_p_json128/inputs/balanced_n128.metric.jsonl",
        "metric_summary": ROOT
        / "runs/ssr_plus_struct_balanced/attempt_p_json128/results_n128/qwen3_4b_traj_three_metrics.csv",
        "paper_avg_paragraphs": 8.83,
        "paper_one_sentence_paragraph_rate": 0.542,
        "paper_tri": 1.170,
    },
    {
        "variant": "Natural-language SSR",
        "method": "SSR_PLUS_STRUCT_BALANCED",
        "n_label": 1000,
        "metric_jsonl": ROOT
        / "runs/ssr_plus_struct_balanced/attempt_o1000/inputs/balanced_n1000.metric.jsonl",
        "metric_summary": ROOT
        / "runs/ssr_plus_struct_balanced/attempt_o1000/results_n1000/qwen3_4b_traj_three_metrics.csv",
        "paper_avg_paragraphs": 5.10,
        "paper_one_sentence_paragraph_rate": 0.023,
        "paper_tri": 0.000,
    },
    {
        "variant": "Natural-language variant",
        "method": "SSR_PLUS_STRUCT_BALANCED_ECHO_CONTROL",
        "n_label": 1000,
        "metric_jsonl": ROOT
        / "runs/ssr_plus_struct_balanced/attempt_s_echo1000/inputs/balanced_n1000.metric.jsonl",
        "metric_summary": ROOT
        / "runs/ssr_plus_struct_balanced/attempt_s_echo1000/results_n1000/qwen3_4b_traj_three_metrics.csv",
        "paper_avg_paragraphs": 5.10,
        "paper_one_sentence_paragraph_rate": 0.025,
        "paper_tri": -0.003,
    },
]

SENTENCE_END_RE = re.compile(r"[.!?\u3002\uff01\uff1f]+(?:[\"')\]]+)?(?=\s|$)")
TAG_RE = re.compile(r"</?(?:reason|skeleton)>")


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean_reasoning(text: str) -> str:
    return TAG_RE.sub("", text or "").strip()


def split_paragraphs(text: str) -> List[str]:
    return [
        part.strip()
        for part in re.split(r"\n\s*\n+", clean_reasoning(text))
        if part.strip()
    ]


def sentence_count(paragraph: str) -> int:
    paragraph = re.sub(r"https?://\S+", " URL ", paragraph)
    if not paragraph.strip():
        return 0
    return max(1, len(SENTENCE_END_RE.findall(paragraph)))


def summarize_run(spec: Dict) -> Dict:
    n = 0
    paragraph_counts: List[int] = []
    one_sentence_paragraphs = 0
    total_paragraphs = 0

    for record in iter_jsonl(spec["metric_jsonl"]):
        reasoning = record["reasonings"][spec["method"]]
        paragraphs = split_paragraphs(reasoning)
        n += 1
        paragraph_counts.append(len(paragraphs))
        total_paragraphs += len(paragraphs)
        one_sentence_paragraphs += sum(
            1 for paragraph in paragraphs if sentence_count(paragraph) <= 1
        )

    avg_paragraphs = sum(paragraph_counts) / n
    one_sentence_rate = one_sentence_paragraphs / total_paragraphs
    return {
        "variant": spec["variant"],
        "method": spec["method"],
        "n": n,
        "n_label": spec["n_label"],
        "avg_paragraphs": avg_paragraphs,
        "total_paragraphs": total_paragraphs,
        "one_sentence_paragraphs": one_sentence_paragraphs,
        "one_sentence_paragraph_rate": one_sentence_rate,
        "metric_jsonl": str(spec["metric_jsonl"].relative_to(ROOT)),
        "metric_summary": str(spec["metric_summary"].relative_to(ROOT)),
        "paper_avg_paragraphs": spec["paper_avg_paragraphs"],
        "paper_one_sentence_paragraph_rate": spec["paper_one_sentence_paragraph_rate"],
        "paper_tri": spec["paper_tri"],
    }


def add_tri(rows: List[Dict]) -> None:
    reference = next(row for row in rows if row["variant"] == "Natural-language SSR")
    ref_avg = reference["avg_paragraphs"]
    ref_one_rate = reference["one_sentence_paragraph_rate"]
    for row in rows:
        row["tri_recomputed"] = math.log(row["avg_paragraphs"] / ref_avg) + 1.2 * (
            row["one_sentence_paragraph_rate"] - ref_one_rate
        )
        row["tri_delta_vs_paper"] = row["tri_recomputed"] - row["paper_tri"]


def write_outputs(rows: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "method",
        "n",
        "n_label",
        "avg_paragraphs",
        "one_sentence_paragraph_rate",
        "one_sentence_paragraphs",
        "total_paragraphs",
        "tri_recomputed",
        "paper_avg_paragraphs",
        "paper_one_sentence_paragraph_rate",
        "paper_tri",
        "avg_paragraphs_delta_vs_paper",
        "one_sentence_rate_delta_vs_paper",
        "tri_delta_vs_paper",
        "metric_summary",
        "metric_jsonl",
    ]
    csv_path = out_dir / "json_naturalness_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    json_path = out_dir / "json_naturalness_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    md_path = out_dir / "json_naturalness_report.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# JSON Naturalness Ablation Diagnostics\n\n")
        handle.write(
            "TRI is recomputed as `log(avg_paragraphs / reference_avg_paragraphs) "
            "+ 1.2 * (one_sentence_rate - reference_one_sentence_rate)`, "
            "where the reference is Natural-language SSR. Paper-reported columns "
            "are retained for exact provenance of the table values.\n\n"
        )
        handle.write(
            "| Variant | N | Avg. paragraphs recomputed | Avg. paragraphs reported | "
            "1-sent. rate recomputed | 1-sent. rate reported | "
            "TRI recomputed | TRI reported |\n"
        )
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['variant']} | {row['n']} | "
                f"{row['avg_paragraphs']:.3f} | "
                f"{row['paper_avg_paragraphs']:.2f} | "
                f"{100 * row['one_sentence_paragraph_rate']:.1f}% | "
                f"{100 * row['paper_one_sentence_paragraph_rate']:.1f}% | "
                f"{row['tri_recomputed']:.3f} | {row['paper_tri']:.3f} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs/json_naturalness_ablation/diagnostics",
        help="Output directory for summary CSV/JSON/Markdown files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [summarize_run(spec) for spec in RUNS]
    add_tri(rows)
    for row in rows:
        row["avg_paragraphs_delta_vs_paper"] = (
            row["avg_paragraphs"] - row["paper_avg_paragraphs"]
        )
        row["one_sentence_rate_delta_vs_paper"] = (
            row["one_sentence_paragraph_rate"]
            - row["paper_one_sentence_paragraph_rate"]
        )
    write_outputs(rows, args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
