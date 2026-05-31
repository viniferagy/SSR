#!/usr/bin/env python3
"""Generate four reverse-CoT variants with vLLM.

The input is the metric-format JSONL used by the anchoring analysis:

{
  "id": ...,
  "questions": {"NEU": "...", ...},
  "answers": {"NEU": "...", ...},
  "contexts": {"NEU": "...", ...},
  "reasonings": {"NEU": "...", ...}
}

This script reuses the prompt templates under `scripts/rcot_generation`,
generates NEU/SUP/AUG-SUP/SSR reasoning traces, writes resumable raw outputs,
then builds a new metric-format JSONL where only `reasonings` are replaced by
the generated traces.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "scripts" / "rcot_generation"
if str(PROMPT_DIR) not in sys.path:
    sys.path.insert(0, str(PROMPT_DIR))

from rcot_prompt import AUGSUP_PROMPT, NEU_PROMPT, SSR_PROMPT, SUP_PROMPT  # noqa: E402


METHOD_ORDER = ["NEU", "SUP", "AUG-SUP", "SSR"]
PROMPTS = {
    "NEU": NEU_PROMPT,
    "SUP": SUP_PROMPT,
    "AUG-SUP": AUGSUP_PROMPT,
    "SSR": SSR_PROMPT,
}


@dataclass(frozen=True)
class Request:
    row_idx: int
    sample_id: Any
    method: str
    prompt: str
    prompt_context: str


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


def fallback_context(row: Dict[str, Any], method: str) -> str:
    question = ensure_text(row.get("questions", {}).get(method))
    answer = ensure_text(row.get("answers", {}).get(method))
    if answer:
        return f"User: {question}\n\nAssistant: {answer}".strip()
    return question


def choose_prompt_context(row: Dict[str, Any], method: str, context_method: str) -> str:
    contexts = row.get("contexts", {})
    selected = method if context_method == "same" else context_method
    context = ensure_text(contexts.get(selected))
    if context.strip():
        return context
    return fallback_context(row, method)


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


def load_completed(raw_output: Path) -> Dict[Tuple[int, str], Dict[str, Any]]:
    completed: Dict[Tuple[int, str], Dict[str, Any]] = {}
    if not raw_output.exists():
        return completed
    with raw_output.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed[(int(row["row_idx"]), row["method"])] = row
    return completed


def build_requests(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    methods: List[str],
    context_method: str,
    limit: int | None,
) -> List[Request]:
    selected_rows = rows[:limit] if limit is not None else rows
    requests: List[Request] = []
    for row_idx, row in enumerate(selected_rows):
        for method in methods:
            context = choose_prompt_context(row, method, context_method)
            if not context.strip():
                continue
            prompt = apply_chat_template(tokenizer, PROMPTS[method], context)
            requests.append(Request(row_idx=row_idx, sample_id=row.get("id", row_idx), method=method, prompt=prompt, prompt_context=context))
    return requests


REASON_RE = re.compile(r"<reason>(.*?)</reason>", flags=re.IGNORECASE | re.DOTALL)
SKELETON_RE = re.compile(r"<skeleton>(.*?)</skeleton>", flags=re.IGNORECASE | re.DOTALL)


def extract_block(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def parse_reasoning(raw_output: str) -> tuple[str, str, str]:
    raw = raw_output.strip()
    skeleton = extract_block(raw, SKELETON_RE)
    reason = extract_block(raw, REASON_RE)
    if reason:
        return reason, skeleton, "tagged"
    # Some generations omit the closing tag; keep the content after the opening
    # tag rather than discarding an otherwise usable trace.
    lower = raw.lower()
    if "<reason>" in lower:
        start = lower.index("<reason>") + len("<reason>")
        return raw[start:].strip(), skeleton, "open_reason_tag"
    return raw, skeleton, "raw"


def build_metric_rows(
    input_rows: List[Dict[str, Any]],
    completed: Dict[Tuple[int, str], Dict[str, Any]],
    methods: List[str],
    limit: int | None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    selected_rows = input_rows[:limit] if limit is not None else input_rows
    metric_rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    parse_counts: Dict[str, int] = {}
    for row_idx, row in enumerate(selected_rows):
        reasonings: Dict[str, str] = {}
        skeletons: Dict[str, str] = {}
        parse_modes: Dict[str, str] = {}
        missing = []
        for method in methods:
            item = completed.get((row_idx, method))
            if not item:
                missing.append(method)
                continue
            reasoning, skeleton, parse_mode = parse_reasoning(ensure_text(item.get("raw_output")))
            parse_counts[parse_mode] = parse_counts.get(parse_mode, 0) + 1
            if not reasoning.strip():
                missing.append(method)
                continue
            reasonings[method] = reasoning
            if skeleton:
                skeletons[method] = skeleton
            parse_modes[method] = parse_mode
        if missing:
            rejected.append({"row_idx": row_idx, "id": row.get("id", row_idx), "missing": missing})
            continue
        out = {
            "id": row.get("id", row_idx),
            "questions": {m: row.get("questions", {}).get(m, "") for m in methods},
            "answers": {m: row.get("answers", {}).get(m, "") for m in methods},
            "contexts": {m: row.get("contexts", {}).get(m, "") for m in methods},
            "reasonings": reasonings,
            "generated_parse_modes": parse_modes,
        }
        if skeletons:
            out["generated_skeletons"] = skeletons
        metric_rows.append(out)
    summary = {
        "input_rows": len(selected_rows),
        "complete_rows": len(metric_rows),
        "rejected_rows": len(rejected),
        "methods": methods,
        "parse_counts": parse_counts,
    }
    return metric_rows, rejected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "scripts" / "examples" / "examples.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Metric-format JSONL with regenerated reasonings")
    parser.add_argument("--raw-output", type=Path, required=True, help="Resumable raw generation JSONL")
    parser.add_argument("--rejected-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=Path("/home/pengguangyue/workspace/models/Qwen/Qwen3-8B"))
    parser.add_argument("--methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--prompt-context-method", default="same", help="'same' or one method name whose context is reused for all prompts")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile/cudagraph capture in vLLM")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--max-seq-len-to-capture", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--build-only", action="store_true", help="Only rebuild parsed metric rows from existing raw output")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in PROMPTS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; known={sorted(PROMPTS)}")

    rows = read_jsonl(args.input)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required. Install the project environment with `uv pip install -e .`.") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    completed = load_completed(args.raw_output)
    if not args.build_only:
        requests = build_requests(rows, tokenizer, methods, args.prompt_context_method, args.limit)
        pending = [req for req in requests if (req.row_idx, req.method) not in completed]
        print(json.dumps({
            "input": str(args.input),
            "model": str(args.model),
            "requests": len(requests),
            "completed": len(completed),
            "pending": len(pending),
            "methods": methods,
            "prompt_context_method": args.prompt_context_method,
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
                        "method": req.method,
                        "raw_output": text,
                        "finish_reason": finish_reason,
                        "output_tokens": len(token_ids),
                        "prompt_chars": len(req.prompt),
                        "prompt_context_chars": len(req.prompt_context),
                    })
                append_jsonl(args.raw_output, raw_rows)
                completed.update({(int(r["row_idx"]), r["method"]): r for r in raw_rows})
                print(json.dumps({
                    "chunk_start": start,
                    "chunk_size": len(chunk),
                    "completed": len(completed),
                    "raw_output": str(args.raw_output),
                }, ensure_ascii=False), flush=True)

    completed = load_completed(args.raw_output)
    metric_rows, rejected, summary = build_metric_rows(rows, completed, methods, args.limit)
    write_jsonl(args.output, metric_rows)
    rejected_output = args.rejected_output or args.output.with_suffix(".rejected.jsonl")
    summary_output = args.summary_output or args.output.with_suffix(".summary.json")
    write_jsonl(rejected_output, rejected)
    summary.update({
        "input": str(args.input),
        "output": str(args.output),
        "raw_output": str(args.raw_output),
        "rejected_output": str(rejected_output),
    })
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
