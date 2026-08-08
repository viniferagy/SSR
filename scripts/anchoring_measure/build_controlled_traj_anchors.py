#!/usr/bin/env python3
"""Build the four controlled references used by the behavioral-zone figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"\s+|[\w]+(?:[-'][\w]+)*|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"^[\w]+(?:[-'][\w]+)*$", re.UNICODE)
NUMBER_RE = re.compile(r"^\d+(?:[.,:/-]\d+)*$")
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "to",
    "of", "in", "on", "for", "with", "as", "by", "is", "are", "was",
    "were", "be", "been", "being", "it", "this", "that", "these", "those",
    "from", "at", "into", "about", "not", "no", "do", "does", "did",
    "can", "could", "would", "should", "will", "may", "might", "must",
    "have", "has", "had", "there", "here", "which", "what", "when",
    "where", "why", "how", "also", "than", "because", "therefore", "thus",
}
NEUTRAL_SENTENCES = (
    "Separate the request, assumptions, evidence, and output constraints.",
    "Consider alternatives before selecting the most defensible route.",
    "Check calculations, source limits, and edge cases where applicable.",
    "Keep the reasoning ordered and connect each step to the task.",
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def content_terms(answer: str, max_terms: int) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for token in TOKEN_RE.findall(answer):
        normalized = token.lower()
        if not WORD_RE.fullmatch(token) or normalized in STOPWORDS:
            continue
        if len(normalized) == 1 and not normalized.isdigit():
            continue
        if normalized not in seen:
            seen.add(normalized)
            terms.append(token)
    terms.sort(key=lambda item: hashlib.sha256(item.lower().encode("utf-8")).hexdigest())
    return terms[:max_terms]


def neutral_padding(min_words: int) -> str:
    parts: list[str] = []
    words = 0
    index = 0
    while words < min_words:
        sentence = NEUTRAL_SENTENCES[index % len(NEUTRAL_SENTENCES)]
        parts.append(sentence)
        words += len(sentence.split())
        index += 1
    return " ".join(parts)


def prob_anchor(blind_reasoning: str, answer: str, max_terms: int, pad_ratio: int) -> str:
    terms = content_terms(answer, max_terms)
    pad = neutral_padding(max(1, len(terms) * pad_ratio))
    tail = ", ".join(terms) if terms else "none"
    return f"{blind_reasoning}\n\n{pad}\n\nEvidence terms (unordered): {tail}".strip()


def traj_anchor(answer: str, max_tokens: int, copy_stride: int) -> str:
    tokens = TOKEN_RE.findall(answer)[:max_tokens]
    rendered: list[str] = []
    content_index = 0
    for token in tokens:
        lower = token.lower()
        if token.isspace() or not WORD_RE.fullmatch(token) or lower in STOPWORDS:
            rendered.append(token)
            continue
        if NUMBER_RE.fullmatch(token):
            rendered.append("<N>")
            continue
        content_index += 1
        rendered.append(token if content_index % copy_stride == 0 else "<C>")
    rule = (
        f"RULE: Transform the first {max_tokens} answer tokens. Preserve spacing, "
        f"punctuation, and function words; copy every {copy_stride}nd content word; "
        "replace other content words with <C> and numbers with <N>."
    )
    return f"{rule}\n{''.join(rendered).strip()}"


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(args.input)):
        if args.limit is not None and len(output) >= args.limit:
            break
        method = args.blind_method
        questions = row.get("questions", {})
        answers = row.get("answers", {})
        contexts = row.get("contexts", {})
        reasonings = row.get("reasonings", {})
        question = text(questions.get(method))
        answer = text(answers.get(method))
        context = text(contexts.get(method))
        blind = text(reasonings.get(method))
        if not all(value.strip() for value in (question, answer, blind)):
            raise ValueError(
                f"row {row.get('id', index)!r} is missing question, answer, or reasoning for {method!r}"
            )
        methods = {
            "Blind CoT": blind,
            "+Prob Anchor": prob_anchor(blind, answer, args.prob_max_terms, args.prob_pad_ratio),
            "+Traj Anchor": traj_anchor(answer, args.traj_prefix_tokens, args.traj_copy_stride),
            "Response-as-CoT": answer,
        }
        output.append(
            {
                "id": row.get("id", index),
                "questions": {name: question for name in methods},
                "answers": {name: answer for name in methods},
                "contexts": {name: context or question for name in methods},
                "reasonings": methods,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blind-method", default="Blind CoT")
    parser.add_argument("--prob-max-terms", type=int, default=192)
    parser.add_argument("--prob-pad-ratio", type=int, default=8)
    parser.add_argument("--traj-prefix-tokens", type=int, default=128)
    parser.add_argument("--traj-copy-stride", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if min(args.prob_max_terms, args.prob_pad_ratio, args.traj_prefix_tokens, args.traj_copy_stride) <= 0:
        parser.error("anchor sizes and ratios must be positive")
    rows = build_rows(args)
    written = write_jsonl(args.output, rows)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "written": written}, indent=2))


if __name__ == "__main__":
    main()
