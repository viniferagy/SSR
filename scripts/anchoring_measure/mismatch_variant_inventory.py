#!/usr/bin/env python3
"""Inventory metric-format JSONL files and write a mismatch source manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


PREFERRED_SOURCES = [
    {
        "path": "runs/enhanced_suppression_variants_qwen3_8b_1k/inputs/methods_with_enhanced.metric.jsonl",
        "methods": ["NEU", "SUP", "AUG-SUP", "SSR", "SSR_PLUS", "SSR_PLUS_STRUCT", "QA-SUP", "PG-SUP"],
        "family": "main-enhanced",
        "split": "full-1k",
    },
    {
        "path": "runs/ssr_plus_struct_variants_1k_newmetrics/stable_20260618_170144/inputs/methods_1k_plus_struct_variants_drop712.metric.jsonl",
        "methods": ["SSR_PLUS_STRUCT_C1_STEPS", "SSR_PLUS_STRUCT_LONG"],
        "family": "struct-variants",
        "split": "full-1k-drop712",
        "allow_missing_records": True,
    },
    {
        "path": "runs/ssr_plus_struct_balanced/inputs/balanced.metric.jsonl",
        "methods": ["SSR_PLUS_STRUCT_BALANCED"],
        "family": "struct-balanced",
        "split": "full-1k",
    },
    {
        "path": "runs/qskel_answer_masked_1k/full/inputs/qskel_1k.metric.jsonl",
        "methods": ["SSR_QSKEL"],
        "family": "answer-masked",
        "split": "full-1k",
    },
    {
        "path": "runs/ssr_2step_qa_qwen3_8b/full/inputs/ssr_2step_qa_1k.metric.jsonl",
        "methods": ["SSR_2STEP_QA"],
        "family": "two-step",
        "split": "full-1k",
    },
    {
        "path": "runs/tier1_baselines/1k/inputs/tier1_final.metric.jsonl",
        "methods": ["A1-FDB", "A6-Gist", "B11-NGramBlock", "B13-BoN"],
        "family": "tier1-baselines",
        "split": "full-1k",
    },
    {
        "path": "runs/tier1_baselines/1k/inputs/b12_pilot_100.metric.jsonl",
        "methods": ["B12-Contrastive-g0.5"],
        "family": "tier1-baselines",
        "split": "pilot-100",
    },
    {
        "path": "runs/entropy_targeted_suppression_smoke/inputs/methods_with_plus.metric.jsonl",
        "methods": ["CV-SUP", "DL-SUP"],
        "family": "entropy-targeted",
        "split": "smoke-128",
    },
    {
        "path": "runs/entropy_targeted_suppression_fs_smoke/inputs/methods_with_plus.metric.jsonl",
        "methods": ["FS-SUP"],
        "family": "entropy-targeted",
        "split": "smoke-128",
    },
    {
        "path": "runs/ssr_plus_struct_compact_smoke/inputs/plus_methods.metric.jsonl",
        "methods": ["SSR_PLUS_STRUCT_COMPACT"],
        "family": "struct-smoke",
        "split": "smoke-128",
    },
    {
        "path": "runs/ssr_plus_struct_mid_smoke/inputs/plus_methods.metric.jsonl",
        "methods": ["SSR_PLUS_STRUCT_MID"],
        "family": "struct-smoke",
        "split": "smoke-128",
    },
    {
        "path": "runs/ssr_plus_struct_long_smoke/inputs/plus_methods.metric.jsonl",
        "methods": ["SSR_PLUS_STRUCT_LONG"],
        "family": "struct-smoke",
        "split": "smoke-128",
        "dedupe_if_seen": True,
    },
    {
        "path": "runs/struct_prompt_ablation_32/inputs/long_direct_256.metric.jsonl",
        "methods": ["SSR_PLUS_STRUCT_LONG"],
        "family": "struct-ablation",
        "split": "ablation-256",
        "dedupe_if_seen": True,
    },
]


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def count_method(path: Path, method: str) -> int:
    n = 0
    for row in read_jsonl(path):
        q = ensure_text(row.get("questions", {}).get(method))
        a = ensure_text(row.get("answers", {}).get(method))
        r = ensure_text(row.get("reasonings", {}).get(method))
        if q.strip() and a.strip() and r.strip():
            n += 1
    return n


def write_manifest(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    seen_methods: set[str] = set()
    sources: List[Dict[str, Any]] = []
    variants: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for spec in PREFERRED_SOURCES:
        path = root / spec["path"]
        if not path.exists():
            skipped.append({"path": spec["path"], "reason": "missing_path"})
            continue
        methods: Dict[str, str] = {}
        counts: Dict[str, int] = {}
        for method in spec["methods"]:
            if spec.get("dedupe_if_seen") and method in seen_methods:
                skipped.append({"path": spec["path"], "method": method, "reason": "dedupe_seen_method"})
                continue
            n = count_method(path, method)
            if n <= args.min_n:
                skipped.append({"path": spec["path"], "method": method, "n": n, "reason": "n_not_gt_min"})
                continue
            methods[method] = method
            counts[method] = n
            seen_methods.add(method)
            variants.append(
                {
                    "method": method,
                    "n": n,
                    "source_path": spec["path"],
                    "family": spec.get("family", ""),
                    "split": spec.get("split", ""),
                    "path_diversity_exact_generator": method
                    in {
                        "NEU", "SUP", "AUG-SUP", "SSR", "SSR_PLUS", "SSR_PLUS_STRUCT", "QA-SUP", "PG-SUP",
                        "CV-SUP", "DL-SUP", "FS-SUP", "SSR_PLUS_STRUCT_COMPACT", "SSR_PLUS_STRUCT_MID",
                        "SSR_PLUS_STRUCT_LONG", "SSR_PLUS_STRUCT_C1_STEPS", "SSR_PLUS_STRUCT_C2_COMPACT",
                        "SSR_PLUS_STRUCT_BALANCED",
                        "SSR_PLUS_STRUCT_C3_EXACT2", "SSR_PLUS_STRUCT_C4_BLANKLINE", "SSR_PLUS_STRUCT_C5_SEPARATE_RULE",
                        "SSR_PLUS_STRUCT_C6_LONG_STEPS", "SSR_PLUS_STRUCT_C7_DEVELOPED", "SSR_PLUS_STRUCT_C8_LONG_REASON",
                        "SSR_PLUS_STRUCT_C9_LONG_TEMPLATE", "SSR_QSKEL", "SSR_2STEP_QA", "A1-FDB", "A6-Gist", "B11-NGramBlock",
                        "B13-BoN", "B12-Contrastive-g0.5",
                    },
                }
            )
        if methods:
            out_spec = {
                "path": spec["path"],
                "methods": methods,
                "family": spec.get("family", ""),
                "split": spec.get("split", ""),
                "counts": counts,
            }
            if spec.get("allow_missing_records"):
                out_spec["allow_missing_records"] = True
            sources.append(out_spec)

    manifest = {
        "root": str(root),
        "min_n_exclusive": args.min_n,
        "sources": sources,
        "variants": variants,
        "skipped": skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sources": len(sources), "variants": len(variants), "skipped": len(skipped)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-n", type=int, default=64)
    args = parser.parse_args()
    write_manifest(args)


if __name__ == "__main__":
    main()
