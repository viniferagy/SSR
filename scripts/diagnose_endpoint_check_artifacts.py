#!/usr/bin/env python3
"""Diagnose visible endpoint-check artifacts in generated reasoning traces."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


STRICT_RE = re.compile(
    r"\b(?:consistency check|core endpoint|endpoint alignment|same endpoint)\b"
    r"|(?:aligns?|aligned|alignment)\s+(?:with|to)\s+(?:the\s+)?(?:assistant|answer|final|reference|user)"
    r"|(?:final\s+(?:output|answer|response)\s+(?:meets?|matches?|aligns?|adheres?))"
    r"|(?:verify|validate|confirm)\s+(?:that\s+)?(?:the\s+)?(?:final|answer|response|output)",
    flags=re.IGNORECASE,
)
BROAD_RE = re.compile(
    r"\b(?:check|checked|checking|verify|verified|verification|validate|validated|validation|"
    r"confirm|confirmed|confirmation|ensure|ensures|ensured|make sure|double-check)\b",
    flags=re.IGNORECASE,
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


def compact(text: str, width: int = 220) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    if len(one_line) <= width:
        return one_line
    return one_line[: width - 3] + "..."


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
            "strict_final_hits": 0,
            "broad_final_hits": 0,
            "strict_any_hits": 0,
            "broad_any_hits": 0,
        }
        for method in methods
    }
    examples: Dict[str, List[Dict[str, str]]] = {method: [] for method in methods}

    for row in read_jsonl(args.input):
        reasonings = row.get("reasonings", {})
        row_id = str(row.get("id", ""))
        for method in methods:
            trace = str(reasonings.get(method, "") or "")
            if not trace:
                continue
            tail = final_paragraph(trace)
            strict_final = bool(STRICT_RE.search(tail))
            broad_final = bool(BROAD_RE.search(tail))
            strict_any = bool(STRICT_RE.search(trace))
            broad_any = bool(BROAD_RE.search(trace))
            item = stats[method]
            item["N"] += 1
            item["strict_final_hits"] += int(strict_final)
            item["broad_final_hits"] += int(broad_final)
            item["strict_any_hits"] += int(strict_any)
            item["broad_any_hits"] += int(broad_any)
            if (strict_final or broad_final) and len(examples[method]) < args.max_examples:
                examples[method].append(
                    {
                        "id": row_id,
                        "strict_final": str(strict_final),
                        "broad_final": str(broad_final),
                        "final_paragraph": compact(tail),
                    }
                )

    fields = [
        "Method",
        "N",
        "strict_final_hits",
        "strict_final_rate",
        "broad_final_hits",
        "broad_final_rate",
        "strict_any_hits",
        "strict_any_rate",
        "broad_any_hits",
        "broad_any_rate",
    ]
    rows: List[Dict[str, Any]] = []
    for method in methods:
        item = stats[method]
        n = max(int(item["N"]), 1)
        item["strict_final_rate"] = item["strict_final_hits"] / n
        item["broad_final_rate"] = item["broad_final_hits"] / n
        item["strict_any_rate"] = item["strict_any_hits"] / n
        item["broad_any_rate"] = item["broad_any_hits"] / n
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
