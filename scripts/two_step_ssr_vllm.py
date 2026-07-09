#!/usr/bin/env python3
"""Generate SSR with a two-step skeleton-then-reasoning process."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


SKELETON_RE = re.compile(r"<skeleton>(.*?)</skeleton>", flags=re.IGNORECASE | re.DOTALL)
REASON_RE = re.compile(r"<reason>(.*?)</reason>", flags=re.IGNORECASE | re.DOTALL)

STEP_TAGS = """| Tag | Description |
| :--- | :--- |
| [PLAN] | Understand the request, constraints, and goal. |
| [RETR] | Retrieve relevant facts, rules, formulas, or context. |
| [INFR] | Perform inference, deduction, calculation, or transformation. |
| [EVAL] | Check correctness, consistency, or sufficiency. |
| [SUMM] | Integrate intermediate conclusions and refine expression. |
| [BTRK] | Revise a prior decision after detecting a problem. |
| [RFLX] | Reflect on the reasoning path and possible improvements. |
| [BRCH] | Consider multiple paths and choose one. |"""

SKELETON_SYSTEM = f"""You are an expert AI assistant reconstructing the hidden reasoning plan for a final assistant response.

The user content contains a conversation whose final assistant response is visible. Create only an abstract skeleton for the reasoning that could lead to that response.

Step tags:
{STEP_TAGS}

Rules:
1. Output only one <skeleton> block.
2. Each line must follow exactly: n. [STEP TAG] <single-sentence skeleton under 20 words>
3. Skeleton lines must match the input language when practical.
4. Do not reveal final answer values, exact outputs, or distinctive answer phrases.
5. Use 10-16 steps for ordinary tasks and 14-22 steps for multi-part, math, coding, or long answers.
6. Split composite actions into separate steps.
7. Include evaluation or reflection steps when correctness checks are meaningful.
8. Do not output a <reason> block.
"""

REASON_SYSTEM = """You are an expert AI assistant expanding a fixed reasoning skeleton into a detailed hidden reasoning process.

Rules:
1. Output only one <reason> block.
2. Use the provided skeleton as a fixed plan; do not add, remove, merge, or reorder steps.
3. Write one paragraph per skeleton step, in the same order.
4. Each paragraph should contain 2-4 substantive sentences.
5. Explain the local goal, relevant constraints, inference, and verification for that step.
6. Do not number paragraphs or copy skeleton labels in the reason block.
7. Do not output the final assistant answer.
8. Avoid directly copying final answer sentences; expand task-relevant reasoning instead.
"""

REASON_SYSTEM_SSR_ALIGNED = f"""You are an expert AI assistant tasked with reconstructing the hidden reasoning process for a given final assistant turn in a dialogue.

A skeleton has already been generated. Use the provided skeleton as the fixed structure for the reasoning process.

### Step Tag Definitions

{STEP_TAGS}

### Output Format and Constraint Rules

**The `<reason>` Block**

1. **Goal:** Provide hidden chain-of-thought, private internal deliberations, or detailed intermediate computations based on steps listed in the provided `<skeleton>` block.
2. **Structure:** The reasoning text must correspond directly to each step listed in the provided `<skeleton>` block, maintaining the exact same order.
3. **Constraint:** For each skeleton intent, write developed reasoning that explains why the step is needed, what local inference is made, and how the inference is checked against the task constraints.
4. **Formatting:** Write continuous, coherent reasoning text or paragraphs. Separate each step with '\\n\\n' for clarity.
5. **Constraint:** Do not explicitly number the steps or use the step labels (e.g., `1.`, `[PLAN]`) within this block.
6. **Constraint:** Stop once each skeleton intent has been sufficiently justified and close `</reason>`.
7. **Constraint:** Do not output the final assistant answer in this response.

**Overall Output Constraint**

Do not output anything outside the required `<reason>` and `</reason>` block.
"""


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


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def extract_block(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def parse_skeleton(text: str) -> tuple[str, str]:
    skeleton = extract_block(text, SKELETON_RE)
    if skeleton:
        return skeleton, "tagged"
    raw = text.strip()
    return raw, "raw"


def parse_reason(text: str) -> tuple[str, str]:
    reason = extract_block(text, REASON_RE)
    if reason:
        return reason, "tagged"
    lower = text.lower()
    if "<reason>" in lower:
        start = lower.index("<reason>") + len("<reason>")
        return text[start:].strip(), "open_reason_tag"
    return text.strip(), "raw"


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


def row_context(row: Dict[str, Any], method: str) -> str:
    context = ensure_text(row.get("contexts", {}).get(method))
    if context.strip():
        return context
    question = ensure_text(row.get("questions", {}).get(method))
    answer = ensure_text(row.get("answers", {}).get(method))
    if answer:
        return f"User: {question}\n\nAssistant: {answer}".strip()
    return question


def build_skeleton_prompts(rows: List[Dict[str, Any]], tokenizer: Any, method: str) -> List[str]:
    prompts = []
    for row in rows:
        context = row_context(row, method)
        user_content = f"Conversation:\n{context}\n\nCreate the abstract skeleton now."
        prompts.append(apply_chat_template(tokenizer, SKELETON_SYSTEM, user_content))
    return prompts


def row_question(row: Dict[str, Any], method: str) -> str:
    return ensure_text(row.get("questions", {}).get(method))


def reason_context(row: Dict[str, Any], method: str, mode: str) -> str:
    if mode == "question":
        return row_question(row, method)
    if mode == "conversation":
        return row_context(row, method)
    raise ValueError(f"Unknown reason context mode: {mode}")


def build_reason_prompts(
    rows: List[Dict[str, Any]],
    skeletons: List[str],
    tokenizer: Any,
    method: str,
    context_mode: str,
    prompt_style: str,
) -> List[str]:
    prompts = []
    system_prompt = REASON_SYSTEM_SSR_ALIGNED if prompt_style == "ssr" else REASON_SYSTEM
    for row, skeleton in zip(rows, skeletons):
        context = reason_context(row, method, context_mode)
        user_content = (
            f"Input:\n{context}\n\n"
            f"Fixed skeleton:\n<skeleton>\n{skeleton}\n</skeleton>\n\n"
            "Expand the reasoning now."
        )
        prompts.append(apply_chat_template(tokenizer, system_prompt, user_content))
    return prompts


def generate(llm: Any, prompts: List[str], sampling: Any, chunk_size: int) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for start in range(0, len(prompts), chunk_size):
        chunk = prompts[start : start + chunk_size]
        outputs = llm.generate(chunk, sampling)
        for local_idx, out in enumerate(outputs):
            text = out.outputs[0].text if out.outputs else ""
            token_ids = out.outputs[0].token_ids if out.outputs else []
            finish_reason = out.outputs[0].finish_reason if out.outputs else None
            out_rows.append(
                {
                    "request_idx": start + local_idx,
                    "raw_output": text,
                    "finish_reason": finish_reason,
                    "output_tokens": len(token_ids),
                    "prompt_chars": len(chunk[local_idx]),
                }
            )
        print(json.dumps({"chunk_start": start, "chunk_size": len(chunk), "completed": len(out_rows)}), flush=True)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-skeleton-output", type=Path, required=True)
    parser.add_argument("--raw-reason-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--method", default="SSR")
    parser.add_argument("--output-method", default="SSR_QSKEL")
    parser.add_argument(
        "--reason-context",
        choices=["question", "conversation"],
        default="question",
        help="Context shown to the second reasoning pass. Use 'question' to mask the final answer.",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--skeleton-max-tokens", type=int, default=2048)
    parser.add_argument("--reason-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--reason-prompt-style",
        choices=["expand", "ssr"],
        default="expand",
        help="Use the original expansion prompt or an SSR-aligned prompt that only emits <reason>.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--max-seq-len-to-capture", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    input_rows = read_jsonl(args.input)
    selected = input_rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        seed=args.seed,
    )

    skeleton_prompts = build_skeleton_prompts(selected, tokenizer, args.method)
    skeleton_raw = generate(
        llm,
        skeleton_prompts,
        SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.skeleton_max_tokens),
        args.chunk_size,
    )
    for row, raw in zip(selected, skeleton_raw):
        raw["id"] = row.get("id")
        raw["method"] = args.output_method
    write_jsonl(args.raw_skeleton_output, skeleton_raw)
    skeletons, skeleton_parse = [], {}
    for raw in skeleton_raw:
        skeleton, mode = parse_skeleton(ensure_text(raw.get("raw_output")))
        skeletons.append(skeleton)
        skeleton_parse[mode] = skeleton_parse.get(mode, 0) + 1

    reason_prompts = build_reason_prompts(
        selected,
        skeletons,
        tokenizer,
        args.method,
        args.reason_context,
        args.reason_prompt_style,
    )
    reason_raw = generate(
        llm,
        reason_prompts,
        SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.reason_max_tokens),
        args.chunk_size,
    )
    for row, raw in zip(selected, reason_raw):
        raw["id"] = row.get("id")
        raw["method"] = args.output_method
    write_jsonl(args.raw_reason_output, reason_raw)

    metric_rows = []
    rejected = []
    reason_parse: Dict[str, int] = {}
    for idx, (row, skeleton, raw) in enumerate(zip(selected, skeletons, reason_raw)):
        reason, mode = parse_reason(ensure_text(raw.get("raw_output")))
        reason_parse[mode] = reason_parse.get(mode, 0) + 1
        if not skeleton.strip() or not reason.strip():
            rejected.append({"row_idx": idx, "id": row.get("id", idx), "reason": "empty_skeleton_or_reason"})
            continue
        method = args.method
        out_method = args.output_method
        out_context = reason_context(row, method, args.reason_context)
        metric_rows.append(
            {
                "id": row.get("id", idx),
                "questions": {out_method: row.get("questions", {}).get(method, "")},
                "answers": {out_method: row.get("answers", {}).get(method, "")},
                "contexts": {out_method: out_context},
                "reasonings": {out_method: reason},
                "generated_skeletons": {out_method: skeleton},
                "generated_parse_modes": {out_method: mode},
            }
        )
    write_jsonl(args.output, metric_rows)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "limit": args.limit,
        "complete_rows": len(metric_rows),
        "rejected_rows": len(rejected),
        "skeleton_parse_counts": skeleton_parse,
        "reason_parse_counts": reason_parse,
        "skeleton_raw_output": str(args.raw_skeleton_output),
        "reason_raw_output": str(args.raw_reason_output),
        "reason_context": args.reason_context,
        "reason_prompt_style": args.reason_prompt_style,
        "rejected": rejected,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
