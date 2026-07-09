#!/usr/bin/env python3
"""Summarize final-paragraph meta/check artifacts in generated traces."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


CHECK_TERMS_RE = re.compile(
    r"\b(?:check|checked|checking|verify|verified|verification|validate|validated|"
    r"validation|confirm|confirmed|confirmation|ensure|ensures|ensured|make sure|"
    r"double-check)\b",
    flags=re.IGNORECASE,
)
META_ALIGNMENT_RE = re.compile(
    r"(?:\b(?:trace|reasoning|paragraph|skeleton|assistant|response|answer|output|"
    r"final|endpoint|alignment|provided exchange|provided answer)\b.{0,90}"
    r"\b(?:aligns?|aligned|alignment|matches?|matched|fits?|consistent|"
    r"supports?|points?|leads?|preserves?|maintains?)\b)"
    r"|(?:\b(?:aligns?|aligned|alignment|matches?|matched|fits?|consistent|"
    r"supports?|points?|leads?|preserves?|maintains?)\b.{0,90}"
    r"\b(?:trace|reasoning|paragraph|skeleton|assistant|response|answer|output|"
    r"final|endpoint|provided exchange|provided answer)\b)",
    flags=re.IGNORECASE | re.DOTALL,
)
RESPONSE_FIT_RE = re.compile(
    r"\b(?:response|answer|output|final)\b.{0,70}"
    r"\b(?:aligns?|matched|matches?|fits?|consistent|meets?|satisfies|"
    r"supports?|points?|leads?)\b"
    r"|"
    r"\b(?:aligns?|matched|matches?|fits?|consistent|meets?|satisfies|"
    r"supports?|points?|leads?)\b.{0,70}\b(?:response|answer|output|final)\b",
    flags=re.IGNORECASE | re.DOTALL,
)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def final_paragraph(trace: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", trace.strip()) if part.strip()]
    return paragraphs[-1] if paragraphs else trace.strip()


def compact(text: str, width: int = 260) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= width:
        return text
    return text[: width - 3].rstrip() + "..."


def methods_from_text(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def count_term(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--examples-json", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    methods = methods_from_text(args.methods)
    stats: Dict[str, Dict[str, Any]] = {
        method: {
            "Method": method,
            "N": 0,
            "literal_check_final_hits": 0,
            "meta_alignment_final_hits": 0,
            "response_fit_final_hits": 0,
            "assistant_count": 0,
            "response_count": 0,
            "final_count": 0,
            "answer_count": 0,
            "check_count": 0,
            "ensure_count": 0,
            "align_count": 0,
        }
        for method in methods
    }
    examples: Dict[str, List[Dict[str, Any]]] = {method: [] for method in methods}

    for row in read_jsonl(args.input):
        row_id = str(row.get("id", ""))
        reasonings = row.get("reasonings", {})
        for method in methods:
            trace = str(reasonings.get(method, "") or "")
            if not trace:
                continue
            tail = final_paragraph(trace)
            literal = bool(CHECK_TERMS_RE.search(tail))
            meta = bool(META_ALIGNMENT_RE.search(tail))
            fit = bool(RESPONSE_FIT_RE.search(tail))

            item = stats[method]
            item["N"] += 1
            item["literal_check_final_hits"] += int(literal)
            item["meta_alignment_final_hits"] += int(meta)
            item["response_fit_final_hits"] += int(fit)
            item["assistant_count"] += count_term(r"\bassistant\b", tail)
            item["response_count"] += count_term(r"\bresponse\b", tail)
            item["final_count"] += count_term(r"\bfinal\b", tail)
            item["answer_count"] += count_term(r"\banswer\b", tail)
            item["check_count"] += count_term(r"\bcheck(?:ed|ing)?\b", tail)
            item["ensure_count"] += count_term(r"\bensure[sd]?\b", tail)
            item["align_count"] += count_term(r"\balign(?:s|ed|ment)?\b", tail)

            if (literal or meta or fit) and len(examples[method]) < args.max_examples:
                examples[method].append(
                    {
                        "id": row_id,
                        "literal_check_final": literal,
                        "meta_alignment_final": meta,
                        "response_fit_final": fit,
                        "final_paragraph": compact(tail),
                    }
                )

    rows: List[Dict[str, Any]] = []
    for method in methods:
        item = stats[method]
        n = max(int(item["N"]), 1)
        item["literal_check_final_rate"] = item["literal_check_final_hits"] / n
        item["meta_alignment_final_rate"] = item["meta_alignment_final_hits"] / n
        item["response_fit_final_rate"] = item["response_fit_final_hits"] / n
        rows.append(item)

    fields = [
        "Method",
        "N",
        "literal_check_final_hits",
        "literal_check_final_rate",
        "meta_alignment_final_hits",
        "meta_alignment_final_rate",
        "response_fit_final_hits",
        "response_fit_final_rate",
        "assistant_count",
        "response_count",
        "final_count",
        "answer_count",
        "check_count",
        "ensure_count",
        "align_count",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    args.examples_json.parent.mkdir(parents=True, exist_ok=True)
    with args.examples_json.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    with args.output_csv.open("r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
