#!/usr/bin/env python3
"""Prepare and validate anchoring-analysis JSONL inputs.

The original repository mixes notebook-only data preparation with the metric
runner. This script keeps the metric input contract explicit:

{
  "id": ...,
  "questions": {"NEU": "...", ...},
  "answers": {"NEU": "...", ...},
  "contexts": {"NEU": "...", ...},
  "reasonings": {"NEU": "...", ...}
}

It can also convert the public SSR-RCoT-16K dataset rows into an SSR-only
metric input, and build controlled-reference rows used for Figure 3.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_METHODS = ["NEU", "SUP", "AUG-SUP", "SSR"]


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def format_conversation(conversations: Any) -> str:
    if not isinstance(conversations, list):
        return ensure_text(conversations)
    lines: List[str] = []
    for msg in conversations:
        if not isinstance(msg, dict):
            lines.append(ensure_text(msg))
            continue
        role = msg.get("role") or msg.get("from") or "user"
        content = msg.get("content") or msg.get("value") or ""
        lines.append(f"{str(role).capitalize()}: {content}")
    return "\n".join(lines).strip()


def extract_question_answer(conversations: Any) -> tuple[str, str]:
    if not isinstance(conversations, list):
        return ensure_text(conversations), ""
    last_user = ""
    last_assistant = ""
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or msg.get("from") or "").lower()
        content = ensure_text(msg.get("content") or msg.get("value") or "")
        if role in {"user", "human"}:
            last_user = content
        elif role in {"assistant", "gpt", "model"}:
            last_assistant = content
    return last_user or format_conversation(conversations), last_assistant


def validate_metric_row(row: Dict[str, Any], methods: List[str]) -> tuple[bool, str]:
    for section in ["questions", "answers", "contexts", "reasonings"]:
        value = row.get(section)
        if not isinstance(value, dict):
            return False, f"missing dict section: {section}"
        missing = [m for m in methods if not ensure_text(value.get(m)).strip()]
        if missing:
            return False, f"section {section} missing methods: {missing}"
    return True, ""


def validate_metric_input(args: argparse.Namespace) -> None:
    methods = args.methods.split(",")
    rows = []
    total = 0
    skipped = 0
    for row in read_jsonl(args.input):
        total += 1
        ok, reason = validate_metric_row(row, methods)
        if ok:
            rows.append(row)
        else:
            skipped += 1
            if args.strict:
                raise ValueError(f"row {row.get('id', total)} invalid: {reason}")
    written = write_jsonl(args.output, rows)
    print(json.dumps({
        "mode": "validate",
        "input": str(args.input),
        "output": str(args.output),
        "methods": methods,
        "total": total,
        "written": written,
        "skipped": skipped,
    }, ensure_ascii=False, indent=2))


def convert_ssr_rcot(args: argparse.Namespace) -> None:
    rows = []
    for idx, row in enumerate(read_jsonl(args.input)):
        conversations = row.get("conversations")
        question, answer = extract_question_answer(conversations)
        reasoning = ensure_text(row.get("reasoning_trace"))
        if not question.strip() or not answer.strip() or not reasoning.strip():
            continue
        method = args.method
        rows.append({
            "id": row.get("id", idx),
            "questions": {method: question},
            "answers": {method: answer},
            "contexts": {method: format_conversation(conversations)},
            "reasonings": {method: reasoning},
        })
    written = write_jsonl(args.output, rows)
    print(json.dumps({
        "mode": "convert-ssr-rcot",
        "input": str(args.input),
        "output": str(args.output),
        "method": args.method,
        "written": written,
    }, ensure_ascii=False, indent=2))


STOPWORD_RE = re.compile(
    r"\b(?:a|an|the|and|or|but|if|then|so|to|of|in|on|for|with|as|by|is|are|"
    r"was|were|be|been|being|it|this|that|these|those|from|at|into|about)\b",
    flags=re.IGNORECASE,
)


def function_word_skeleton(text: str) -> str:
    words = STOPWORD_RE.findall(text)
    if not words:
        words = re.findall(r"\w+", text)[:20]
    if not words:
        return text[:200]
    chunks = [" ".join(words[i:i + 16]) for i in range(0, len(words), 16)]
    return "\n\n".join(chunks[:8])


def build_controlled_reference(args: argparse.Namespace) -> None:
    base_method = args.base_method
    rows = []
    for idx, row in enumerate(read_jsonl(args.input)):
        questions = row.get("questions", {})
        answers = row.get("answers", {})
        contexts = row.get("contexts", {})
        reasonings = row.get("reasonings", {})
        q = ensure_text(questions.get(base_method))
        a = ensure_text(answers.get(base_method))
        c = ensure_text(contexts.get(base_method))
        r = ensure_text(reasonings.get(base_method))
        if not (q.strip() and a.strip() and r.strip()):
            continue
        methods = {
            "Real CoT": r,
            "+Prob Anchor": f"{r}\n\n{a}",
            "+Entropy Anchor": function_word_skeleton(a),
            "Response as CoT": a,
        }
        rows.append({
            "id": row.get("id", idx),
            "questions": {m: q for m in methods},
            "answers": {m: a for m in methods},
            "contexts": {m: c or q for m in methods},
            "reasonings": methods,
        })
    written = write_jsonl(args.output, rows)
    print(json.dumps({
        "mode": "controlled-reference",
        "input": str(args.input),
        "output": str(args.output),
        "base_method": base_method,
        "written": written,
        "note": "Controlled references are generated from available traces; Real CoT is approximated by the chosen base method.",
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="Validate/copy existing metric-format JSONL")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=validate_metric_input)

    p = sub.add_parser("convert-ssr-rcot", help="Convert public SSR-RCoT rows to SSR-only metric input")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", default="SSR")
    p.set_defaults(func=convert_ssr_rcot)

    p = sub.add_parser("controlled-reference", help="Build Figure 3 controlled-reference metric input")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-method", default="NEU")
    p.set_defaults(func=build_controlled_reference)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
