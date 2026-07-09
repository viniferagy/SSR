#!/usr/bin/env python3
"""Diagnose detached self-review artifacts in reasoning trace endings."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


CHECK_FAMILY_RE = re.compile(
    r"\b(?:check|checked|checking|verify|verified|verification|validate|validated|"
    r"validation|confirm|confirmed|confirmation|ensure|ensures|ensured|make sure|"
    r"double-check)\b",
    flags=re.IGNORECASE,
)
EXPLICIT_CHECK_RE = re.compile(
    r"\b(?:check|checked|checking|verify|verified|verification|validate|validated|"
    r"validation|confirm|confirmed|confirmation|double-check)\b",
    flags=re.IGNORECASE,
)
FIRST_PERSON_ENSURE_RE = re.compile(
    r"\b(?:I|we)\s+(?:need|must|should|can|will|have)\s+to\s+ensure\b"
    r"|\b(?:I|we)\s+(?:ensure|check|verify|confirm|validate)\b",
    flags=re.IGNORECASE,
)
DETACHED_SUBJECT_RE = re.compile(
    r"\b(?:assistant|response|answer|output|final|final response|final answer|"
    r"trace|reasoning|paragraph|endpoint|alignment)\b",
    flags=re.IGNORECASE,
)
DETACHED_REVIEW_RE = re.compile(
    r"(?:(?:assistant|response|answer|output|final|final response|final answer|"
    r"trace|reasoning|paragraph|endpoint|alignment)\b.{0,90}"
    r"\b(?:ensure|ensures|ensured|check|checked|verify|verified|validate|validated|"
    r"confirm|confirmed|aligns?|aligned|alignment|matches?|matched|fits?|"
    r"consistent|meets?|satisfies|supports?|preserves?|maintains?)\b)"
    r"|(?:\b(?:ensure|ensures|ensured|check|checked|verify|verified|validate|validated|"
    r"confirm|confirmed|aligns?|aligned|alignment|matches?|matched|fits?|"
    r"consistent|meets?|satisfies|supports?|preserves?|maintains?)\b.{0,90}"
    r"\b(?:assistant|response|answer|output|final|final response|final answer|"
    r"trace|reasoning|paragraph|endpoint|alignment)\b)",
    flags=re.IGNORECASE | re.DOTALL,
)
THIRD_PERSON_ASSISTANT_RE = re.compile(
    r"\b(?:the\s+)?assistant\b.{0,120}"
    r"\b(?:ensure|ensures|ensured|check|checked|verify|verified|validate|validated|"
    r"confirm|confirmed|aligns?|aligned|matches?|fits?|meets?|satisfies)\b"
    r"|"
    r"\b(?:ensure|ensures|ensured|check|checked|verify|verified|validate|validated|"
    r"confirm|confirmed|aligns?|aligned|matches?|fits?|meets?|satisfies)\b.{0,120}"
    r"\b(?:the\s+)?assistant\b",
    flags=re.IGNORECASE | re.DOTALL,
)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def methods_from_text(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def final_paragraph(trace: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", trace.strip()) if part.strip()]
    return paragraphs[-1] if paragraphs else trace.strip()


def compact(text: str, width: int = 260) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= width:
        return text
    return text[: width - 3].rstrip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--examples-json", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    methods = methods_from_text(args.methods)
    stats: Dict[str, Dict[str, Any]] = {
        method: {
            "Method": method,
            "N": 0,
            "broad_check_final_hits": 0,
            "explicit_check_final_hits": 0,
            "first_person_ensure_final_hits": 0,
            "detached_subject_final_hits": 0,
            "detached_review_final_hits": 0,
            "third_person_assistant_review_final_hits": 0,
        }
        for method in methods
    }
    examples: Dict[str, List[Dict[str, Any]]] = {method: [] for method in methods}

    for idx, row in enumerate(read_jsonl(args.input)):
        if args.limit is not None and idx >= args.limit:
            break
        row_id = str(row.get("id", ""))
        reasonings = row.get("reasonings", {})
        for method in methods:
            trace = str(reasonings.get(method, "") or "")
            if not trace:
                continue
            tail = final_paragraph(trace)
            broad = bool(CHECK_FAMILY_RE.search(tail))
            explicit = bool(EXPLICIT_CHECK_RE.search(tail))
            first_person = bool(FIRST_PERSON_ENSURE_RE.search(tail))
            detached_subject = bool(DETACHED_SUBJECT_RE.search(tail))
            detached_review = bool(DETACHED_REVIEW_RE.search(tail))
            third_person = bool(THIRD_PERSON_ASSISTANT_RE.search(tail))

            item = stats[method]
            item["N"] += 1
            item["broad_check_final_hits"] += int(broad)
            item["explicit_check_final_hits"] += int(explicit)
            item["first_person_ensure_final_hits"] += int(first_person)
            item["detached_subject_final_hits"] += int(detached_subject)
            item["detached_review_final_hits"] += int(detached_review)
            item["third_person_assistant_review_final_hits"] += int(third_person)

            if (detached_review or third_person) and len(examples[method]) < args.max_examples:
                examples[method].append(
                    {
                        "id": row_id,
                        "broad_check_final": broad,
                        "first_person_ensure_final": first_person,
                        "detached_review_final": detached_review,
                        "third_person_assistant_review_final": third_person,
                        "final_paragraph": compact(tail),
                    }
                )

    fields = [
        "Method",
        "N",
        "broad_check_final_hits",
        "broad_check_final_rate",
        "explicit_check_final_hits",
        "explicit_check_final_rate",
        "first_person_ensure_final_hits",
        "first_person_ensure_final_rate",
        "detached_subject_final_hits",
        "detached_subject_final_rate",
        "detached_review_final_hits",
        "detached_review_final_rate",
        "third_person_assistant_review_final_hits",
        "third_person_assistant_review_final_rate",
    ]
    rows: List[Dict[str, Any]] = []
    for method in methods:
        item = stats[method]
        n = max(int(item["N"]), 1)
        for key in [
            "broad_check_final",
            "explicit_check_final",
            "first_person_ensure_final",
            "detached_subject_final",
            "detached_review_final",
            "third_person_assistant_review_final",
        ]:
            item[f"{key}_rate"] = item[f"{key}_hits"] / n
        rows.append(item)

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
