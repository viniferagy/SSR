#!/usr/bin/env python3
"""Prepare 1k reviewer-protocol input rows from SSR-RCoT-16K."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


METHODS = ["NEU", "SUP", "AUG-SUP", "SSR"]


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
        role = ensure_text(msg.get("role") or msg.get("from") or "user").capitalize()
        content = ensure_text(msg.get("content") or msg.get("value") or "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def extract_question_answer(conversations: Any) -> tuple[str, str]:
    if not isinstance(conversations, list):
        return ensure_text(conversations), ""
    last_user = ""
    last_assistant = ""
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        role = ensure_text(msg.get("role") or msg.get("from") or "").lower()
        content = ensure_text(msg.get("content") or msg.get("value") or "")
        if role in {"user", "human"}:
            last_user = content
        elif role in {"assistant", "gpt", "model"}:
            last_assistant = content
    return last_user or format_conversation(conversations), last_assistant


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def skeleton_text(row: Dict[str, Any]) -> str:
    skeleton = row.get("structural_skeleton")
    if not isinstance(skeleton, list):
        return ""
    lines = []
    for idx, item in enumerate(skeleton, 1):
        if not isinstance(item, dict):
            continue
        tag = ensure_text(item.get("tag")).strip().strip("[]")
        summary = ensure_text(item.get("summary")).strip()
        if tag and summary:
            lines.append(f"{idx}. [{tag}] {summary}")
    return "\n".join(lines)


def dataset_to_metric_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    question, answer = extract_question_answer(row.get("conversations"))
    context = format_conversation(row.get("conversations"))
    reasoning = ensure_text(row.get("reasoning_trace"))
    skel = skeleton_text(row)
    out = {
        "id": row.get("id", idx),
        "questions": {m: question for m in METHODS},
        "answers": {m: answer for m in METHODS},
        "contexts": {m: context for m in METHODS},
        "reasonings": {m: reasoning for m in METHODS},
    }
    if skel:
        out["dataset_skeleton"] = skel
    return out


def sample_hf(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    rows = []
    selected = []
    skipped = []
    for idx in indices:
        row = dataset_to_metric_row(ds[int(idx)], int(idx))
        q = row["questions"]["NEU"]
        a = row["answers"]["NEU"]
        r = row["reasonings"]["NEU"]
        if not (q.strip() and a.strip() and r.strip()):
            skipped.append({"source_index": int(idx), "reason": "missing_q_a_or_r"})
            continue
        if tokenizer is not None:
            q_tokens = len(tokenizer(q, add_special_tokens=False).input_ids)
            a_tokens = len(tokenizer(a, add_special_tokens=False).input_ids)
            ctx_tokens = len(tokenizer(row["contexts"]["NEU"], add_special_tokens=False).input_ids)
            r_tokens = len(tokenizer(r, add_special_tokens=False).input_ids)
            if a_tokens > args.max_answer_tokens:
                skipped.append({"source_index": int(idx), "reason": "answer_too_long", "answer_tokens": a_tokens})
                continue
            if q_tokens > args.max_question_tokens:
                skipped.append({"source_index": int(idx), "reason": "question_too_long", "question_tokens": q_tokens})
                continue
            if ctx_tokens + args.generation_max_tokens > args.max_generation_context_tokens:
                skipped.append({"source_index": int(idx), "reason": "generation_context_too_long", "context_tokens": ctx_tokens})
                continue
            if q_tokens + r_tokens + a_tokens > args.max_scoring_tokens:
                skipped.append({
                    "source_index": int(idx),
                    "reason": "scoring_context_too_long",
                    "question_tokens": q_tokens,
                    "reasoning_tokens": r_tokens,
                    "answer_tokens": a_tokens,
                })
                continue
            row["token_budget"] = {
                "question_tokens": q_tokens,
                "answer_tokens": a_tokens,
                "context_tokens": ctx_tokens,
                "dataset_reasoning_tokens": r_tokens,
            }
        else:
            if len(row["contexts"]["NEU"]) > args.max_context_chars:
                skipped.append({"source_index": int(idx), "reason": "context_too_long_chars"})
                continue
            if len(a) > args.max_answer_chars:
                skipped.append({"source_index": int(idx), "reason": "answer_too_long_chars"})
                continue
        selected.append(int(idx))
        row["source_index"] = int(idx)
        rows.append(row)
        if len(rows) >= args.n:
            break

    if len(rows) < args.n:
        raise RuntimeError(f"Only found {len(rows)} usable rows, requested {args.n}")

    write_jsonl(args.output, rows)
    manifest = {
        "mode": "sample-hf",
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "n": args.n,
        "output": str(args.output),
        "source_indices": selected,
        "filters": {
            "tokenizer": str(args.tokenizer) if args.tokenizer else None,
            "max_answer_tokens": args.max_answer_tokens,
            "max_question_tokens": args.max_question_tokens,
            "max_generation_context_tokens": args.max_generation_context_tokens,
            "generation_max_tokens": args.generation_max_tokens,
            "max_scoring_tokens": args.max_scoring_tokens,
            "max_context_chars": args.max_context_chars,
            "max_answer_chars": args.max_answer_chars,
        },
        "skipped_count": len(skipped),
        "skipped_preview": skipped[:50],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def merge_ssr(args: argparse.Namespace) -> None:
    base_rows = list(read_jsonl(args.base))
    ssr_rows = list(read_jsonl(args.ssr_verbose))
    if len(base_rows) != len(ssr_rows):
        raise ValueError(f"row count mismatch: base={len(base_rows)} ssr={len(ssr_rows)}")
    out = []
    for idx, (base, ssr) in enumerate(zip(base_rows, ssr_rows)):
        row = json.loads(json.dumps(base, ensure_ascii=False))
        ssr_reason = ensure_text(ssr.get("reasonings", {}).get(args.ssr_method))
        if not ssr_reason.strip():
            raise ValueError(f"missing SSR verbose reasoning at row {idx}")
        row["reasonings"]["SSR"] = ssr_reason
        if ssr.get("generated_skeletons", {}).get(args.ssr_method):
            row.setdefault("generated_skeletons", {})["SSR"] = ssr["generated_skeletons"][args.ssr_method]
        out.append(row)
    n = write_jsonl(args.output, out)
    print(json.dumps({"mode": "merge-ssr", "rows": n, "output": str(args.output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample-hf")
    p.add_argument("--dataset", default="Nanbeige/SSR-RCoT-16K")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--max-answer-tokens", type=int, default=4096)
    p.add_argument("--max-question-tokens", type=int, default=2048)
    p.add_argument("--max-generation-context-tokens", type=int, default=24576)
    p.add_argument("--generation-max-tokens", type=int, default=8192)
    p.add_argument("--max-scoring-tokens", type=int, default=24576)
    p.add_argument("--max-context-chars", type=int, default=100000)
    p.add_argument("--max-answer-chars", type=int, default=20000)
    p.set_defaults(func=sample_hf)

    p = sub.add_parser("merge-ssr")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--ssr-verbose", type=Path, required=True)
    p.add_argument("--ssr-method", default="SSR")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=merge_ssr)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
