#!/usr/bin/env python3
"""Audit incremental mismatch-dimension runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
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


def read_methods(run_dir: Path, manifest: Dict[str, Any]) -> List[str]:
    methods_file = run_dir / "inputs" / "methods.txt"
    if methods_file.exists():
        return [line.strip() for line in methods_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: List[str] = []
    seen: set[str] = set()
    for spec in manifest.get("sources", []):
        for method in spec.get("methods", {}).values():
            if method not in seen:
                out.append(str(method))
                seen.add(str(method))
    return out


def expected_pairs(rows: Sequence[Dict[str, Any]], methods: Sequence[str], start: int, end: int) -> Tuple[int, Dict[str, int]]:
    total = 0
    by_method = {m: 0 for m in methods}
    for sample_idx, row in enumerate(rows):
        if not (start <= sample_idx < end):
            continue
        for method in methods:
            if (
                ensure_text(row.get("questions", {}).get(method)).strip()
                and ensure_text(row.get("answers", {}).get(method)).strip()
                and ensure_text(row.get("reasonings", {}).get(method)).strip()
            ):
                total += 1
                by_method[method] += 1
    return total, by_method


def count_metric_rows(path: Path, pattern: str, methods: Sequence[str], start: int, end: int) -> Tuple[int, Dict[str, int]]:
    seen: set[Tuple[int, str]] = set()
    by_method = {m: 0 for m in methods}
    for file in sorted(path.glob(pattern)):
        for row in read_jsonl(file):
            try:
                sample_idx = int(row.get("sample_idx", -1))
            except Exception:
                continue
            method = str(row.get("method", ""))
            if start <= sample_idx < end and method in by_method:
                key = (sample_idx, method)
                if key not in seen:
                    seen.add(key)
                    by_method[method] += 1
    return len(seen), by_method


def invalid_count(path: Path, pattern: str) -> int:
    return sum(len(read_jsonl(file)) for file in sorted(path.glob(pattern)))


def csv_methods(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return sorted({str(row.get("method", "")) for row in reader if row.get("method")})


def csv_metric_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {str(row.get("metric", "")) for row in reader if row.get("metric")}


def expected_path_pairs(rows: Sequence[Dict[str, Any]], methods: Sequence[str], end: int) -> Tuple[int, Dict[str, int]]:
    total = 0
    by_method = {m: 0 for m in methods}
    for sample_idx, row in enumerate(rows):
        if sample_idx >= end:
            continue
        for method in methods:
            if ensure_text(row.get("questions", {}).get(method)).strip():
                total += 1
                by_method[method] += 1
    return total, by_method


def path_diversity_methods(manifest: Dict[str, Any], methods: Sequence[str]) -> List[str]:
    allowed = {str(v.get("method")) for v in manifest.get("variants", []) if v.get("path_diversity_exact_generator")}
    return [method for method in methods if method in allowed]


def read_path_pairs(path: Path, k: int) -> Tuple[set[Tuple[int, str]], Dict[str, int], set[Tuple[int, str]], Dict[str, int]]:
    present: set[Tuple[int, str]] = set()
    complete: set[Tuple[int, str]] = set()
    present_by_method: Dict[str, int] = {}
    by_method: Dict[str, int] = {}
    if not path.exists():
        return present, present_by_method, complete, by_method
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sample_idx = int(row.get("sample_idx", -1))
                method = str(row.get("method", ""))
                conditioned_k = int(float(row.get("conditioned_k", 0) or 0))
                blind_k = int(float(row.get("blind_k", 0) or 0))
                div = float(row.get("DivCompress", "nan"))
            except Exception:
                continue
            if method:
                present.add((sample_idx, method))
                present_by_method[method] = present_by_method.get(method, 0) + 1
            if method and conditioned_k >= k and blind_k >= k and div == div:
                complete.add((sample_idx, method))
                by_method[method] = by_method.get(method, 0) + 1
    return present, present_by_method, complete, by_method


def audit(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    methods = read_methods(run_dir, manifest)
    path_methods_expected = path_diversity_methods(manifest, methods)
    stages = [int(x) for x in args.stages.split(",") if x.strip()]
    input_path = run_dir / "inputs" / f"mismatch_variants_{max(stages)}.metric.jsonl"
    rows = read_jsonl(input_path)
    if not rows:
        raise SystemExit(f"missing input rows: {input_path}")

    lines: List[str] = []
    lines.append("# Mismatch Dimensions Audit")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- variants: {len(methods)}")
    lines.append(f"- stages: {', '.join(str(x) for x in stages)}")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append("| method | manifest_n | source | path_diversity_exact |")
    lines.append("| --- | ---: | --- | --- |")
    variant_by_method = {str(v.get("method")): v for v in manifest.get("variants", [])}
    for method in methods:
        v = variant_by_method.get(method, {})
        lines.append(
            f"| {method} | {v.get('n', '')} | {v.get('source_path', '')} | {bool(v.get('path_diversity_exact_generator'))} |"
        )

    rows_out: List[Dict[str, Any]] = []
    prev = 0
    for stage in stages:
        prefix_chunk = run_dir / "chunks" / "prefix" / f"rows_{prev:04d}_{stage:04d}"
        commit_chunk = run_dir / "chunks" / "commitment_confidence" / f"rows_{prev:04d}_{stage:04d}"
        exp, _exp_by_method = expected_pairs(rows, methods, prev, stage)
        prefix_got, _ = count_metric_rows(prefix_chunk, "prefix_metrics_rank*.jsonl", methods, prev, stage)
        commit_got, _ = count_metric_rows(commit_chunk, "commitment_metrics_rank*.jsonl", methods, prev, stage)
        stage_dir = run_dir / f"n{stage:04d}"
        path_results = stage_dir / "path_diversity" / "results" / "path_diversity_summary.csv"
        path_methods = csv_methods(path_results)
        prefix_metrics = csv_metric_names(stage_dir / "prefix_results" / "prefix_means.csv")
        commitment_metrics = csv_metric_names(stage_dir / "commitment_confidence_results" / "commitment_means.csv")
        path_expected, path_expected_by_method = expected_path_pairs(rows, path_methods_expected, stage)
        path_present_pairs, path_present_by_method, path_complete_pairs, path_complete_by_method = read_path_pairs(
            stage_dir / "path_diversity" / "results" / "per_sample_path_diversity.csv",
            args.path_diversity_k,
        )
        path_present = {
            (sample_idx, method)
            for sample_idx, method in path_present_pairs
            if sample_idx < stage and method in path_expected_by_method
        }
        path_complete = {
            (sample_idx, method)
            for sample_idx, method in path_complete_pairs
            if sample_idx < stage and method in path_expected_by_method
        }
        path_excluded_pairs = len(path_present - path_complete)
        path_methods_complete = sum(
            1
            for method, expected_count in path_expected_by_method.items()
            if expected_count > 0 and path_complete_by_method.get(method, 0) > 0
        )
        path_all_methods = sum(1 for count in path_expected_by_method.values() if count > 0)
        rows_out.append(
            {
                "stage": stage,
                "chunk": f"{prev}-{stage}",
                "expected_pairs": exp,
                "prefix_pairs": prefix_got,
                "prefix_complete": prefix_got >= exp,
                "prefix_invalid_rows": invalid_count(prefix_chunk, "prefix_invalid_rank*.jsonl"),
                "prefix_report": (stage_dir / "prefix_results" / "prefix_report.md").exists(),
                "commitment_pairs": commit_got,
                "commitment_complete": commit_got >= exp,
                "commitment_invalid_rows": invalid_count(commit_chunk, "commitment_invalid_rank*.jsonl"),
                "commitment_report": (stage_dir / "commitment_confidence_results" / "commitment_report.md").exists(),
                "path_report": (stage_dir / "path_diversity" / "results" / "path_diversity_report.md").exists(),
                "path_methods": len(path_methods),
                "path_expected_pairs": path_expected,
                "path_present_pairs": len(path_present),
                "path_complete_pairs": len(path_complete),
                "path_excluded_pairs": path_excluded_pairs,
                "path_complete_methods": path_methods_complete,
                "path_expected_methods": path_all_methods,
                "dim_commitment_profile": {"B_025", "B_100", "EarlyShare25", "CommitmentConcentration"}.issubset(prefix_metrics),
                "dim_confidence_gap": "ConfidenceGap_mean" in commitment_metrics,
                "dim_commitment_kl": "KL_mean" in commitment_metrics,
                "dim_path_diversity": (
                    bool(path_methods)
                    and (stage_dir / "path_diversity" / "results" / "path_diversity_report.md").exists()
                    and len(path_present) >= path_expected
                    and len(path_complete) > 0
                    and path_methods_complete >= path_all_methods
                ),
            }
        )
        prev = stage

    lines.append("")
    lines.append("## Stage Status")
    lines.append("")
    lines.append("| stage | chunk | expected | prefix | prefix report | commitment | commitment report | path present | path valid | excluded | path methods | path report |")
    lines.append("| ---: | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |")
    for row in rows_out:
        lines.append(
            f"| {row['stage']} | {row['chunk']} | {row['expected_pairs']} | "
            f"{row['prefix_pairs']} | {row['prefix_report']} | {row['commitment_pairs']} | "
            f"{row['commitment_report']} | {row['path_present_pairs']}/{row['path_expected_pairs']} | "
            f"{row['path_complete_pairs']} | {row['path_excluded_pairs']} | "
            f"{row['path_complete_methods']}/{row['path_expected_methods']} | {row['path_report']} |"
        )

    lines.append("")
    lines.append("## Dimension Gates")
    lines.append("")
    lines.append("| stage | commitment profile | confidence gap | commitment KL | path diversity |")
    lines.append("| ---: | --- | --- | --- | --- |")
    for row in rows_out:
        lines.append(
            f"| {row['stage']} | {row['dim_commitment_profile']} | {row['dim_confidence_gap']} | "
            f"{row['dim_commitment_kl']} | {row['dim_path_diversity']} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_out = args.output.with_suffix(".json")
    json_out.write_text(json.dumps({"variants": methods, "stages": rows_out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.output), "json": str(json_out)}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stages", default="64,256,512,1000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-diversity-k", type=int, default=5)
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
