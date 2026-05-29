#!/usr/bin/env python3
"""Generate answer-blind reference reasoning R0 with vLLM.

R0 is generated from Q only, without the pre-committed answer A. The output is
a one-method metric-format JSONL with method name ``R0`` so the existing paper
formula metric runner can score how much answer information ordinary
answer-blind reasoning contributes.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


R0_PROMPT = """You are an expert AI assistant. You will be given a user query only, without any final answer. Generate a concise, step-by-step reasoning process that could help solve the query from first principles. Do not assume or reveal any pre-existing final answer. Do not include the final answer itself. Output only the reasoning enclosed within <reason> and </reason> tags."""

REASON_RE = re.compile(r"<reason>(.*?)</reason>", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Request:
    row_idx: int
    sample_id: Any
    prompt: str
    question: str


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
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


def first_method(row: Dict[str, Any]) -> str:
    questions = row.get("questions", {})
    if "NEU" in questions:
        return "NEU"
    if questions:
        return next(iter(questions))
    return "R0"


def apply_chat_template(tokenizer: Any, system_prompt: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_requests(rows: List[Dict[str, Any]], tokenizer: Any, limit: int | None) -> List[Request]:
    selected = rows[:limit] if limit is not None else rows
    requests: List[Request] = []
    for row_idx, row in enumerate(selected):
        method = first_method(row)
        question = ensure_text(row.get("questions", {}).get(method))
        if not question.strip():
            continue
        prompt = apply_chat_template(tokenizer, R0_PROMPT, question)
        requests.append(Request(row_idx=row_idx, sample_id=row.get("id", row_idx), prompt=prompt, question=question))
    return requests


def load_completed(raw_output: Path) -> Dict[int, Dict[str, Any]]:
    completed: Dict[int, Dict[str, Any]] = {}
    if not raw_output.exists():
        return completed
    with raw_output.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed[int(row["row_idx"])] = row
    return completed


def parse_reasoning(raw_output: str) -> tuple[str, str]:
    raw = raw_output.strip()
    match = REASON_RE.search(raw)
    if match:
        return match.group(1).strip(), "tagged"
    lower = raw.lower()
    if "<reason>" in lower:
        start = lower.index("<reason>") + len("<reason>")
        return raw[start:].strip(), "open_reason_tag"
    return raw, "raw"


def build_metric_rows(
    rows: List[Dict[str, Any]],
    completed: Dict[int, Dict[str, Any]],
    limit: int | None,
    method_name: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    selected = rows[:limit] if limit is not None else rows
    metric_rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    parse_counts: Dict[str, int] = {}
    for row_idx, row in enumerate(selected):
        source_method = first_method(row)
        item = completed.get(row_idx)
        if not item:
            rejected.append({"row_idx": row_idx, "id": row.get("id", row_idx), "reason": "missing_raw_output"})
            continue
        reasoning, parse_mode = parse_reasoning(ensure_text(item.get("raw_output")))
        parse_counts[parse_mode] = parse_counts.get(parse_mode, 0) + 1
        if not reasoning.strip():
            rejected.append({"row_idx": row_idx, "id": row.get("id", row_idx), "reason": "empty_reasoning"})
            continue
        q = ensure_text(row.get("questions", {}).get(source_method))
        a = ensure_text(row.get("answers", {}).get(source_method))
        metric_rows.append({
            "id": row.get("id", row_idx),
            "questions": {method_name: q},
            "answers": {method_name: a},
            "contexts": {method_name: q},
            "reasonings": {method_name: reasoning},
            "generated_parse_modes": {method_name: parse_mode},
        })
    summary = {
        "input_rows": len(selected),
        "complete_rows": len(metric_rows),
        "rejected_rows": len(rejected),
        "method": method_name,
        "parse_counts": parse_counts,
    }
    return metric_rows, rejected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--method-name", default="R0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--max-seq-len-to-capture", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required. Install the project environment with `uv pip install -e .`.") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    completed = load_completed(args.raw_output)

    if not args.build_only:
        requests = build_requests(rows, tokenizer, args.limit)
        pending = [req for req in requests if req.row_idx not in completed]
        print(json.dumps({
            "input": str(args.input),
            "model": str(args.model),
            "requests": len(requests),
            "completed": len(completed),
            "pending": len(pending),
            "method": args.method_name,
        }, ensure_ascii=False, indent=2), flush=True)
        if pending:
            try:
                from vllm import LLM, SamplingParams
            except ImportError as exc:
                raise SystemExit("vLLM is required for generation. Install vLLM, or use --build-only with existing raw outputs.") from exc
            llm = LLM(
                model=str(args.model),
                tensor_parallel_size=args.tensor_parallel_size,
                trust_remote_code=args.trust_remote_code,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
                disable_custom_all_reduce=args.disable_custom_all_reduce,
                max_seq_len_to_capture=args.max_seq_len_to_capture,
                seed=args.seed,
            )
            sampling = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            for start in range(0, len(pending), args.chunk_size):
                chunk = pending[start:start + args.chunk_size]
                outputs = llm.generate([req.prompt for req in chunk], sampling)
                raw_rows = []
                for req, out in zip(chunk, outputs):
                    text = out.outputs[0].text if out.outputs else ""
                    finish_reason = out.outputs[0].finish_reason if out.outputs else None
                    token_ids = out.outputs[0].token_ids if out.outputs else []
                    raw_rows.append({
                        "row_idx": req.row_idx,
                        "id": req.sample_id,
                        "method": args.method_name,
                        "raw_output": text,
                        "finish_reason": finish_reason,
                        "output_tokens": len(token_ids),
                        "prompt_chars": len(req.prompt),
                        "prompt_question_chars": len(req.question),
                    })
                append_jsonl(args.raw_output, raw_rows)
                completed.update({int(r["row_idx"]): r for r in raw_rows})
                print(json.dumps({
                    "chunk_start": start,
                    "chunk_size": len(chunk),
                    "completed": len(completed),
                    "raw_output": str(args.raw_output),
                }, ensure_ascii=False), flush=True)

    completed = load_completed(args.raw_output)
    metric_rows, rejected, summary = build_metric_rows(rows, completed, args.limit, args.method_name)
    write_jsonl(args.output, metric_rows)
    rejected_output = args.output.with_suffix(".rejected.jsonl")
    summary_output = args.output.with_suffix(".summary.json")
    write_jsonl(rejected_output, rejected)
    summary.update({
        "input": str(args.input),
        "output": str(args.output),
        "raw_output": str(args.raw_output),
        "rejected_output": str(rejected_output),
    })
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
