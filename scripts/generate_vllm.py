#!/usr/bin/env python3
"""Generate RCoT or answer-blind R0 traces with vLLM.

Modes:

- ``rcot`` generates NEU/SUP/AUG-SUP/SSR reverse-CoT traces from metric-format
  rows and writes a metric-format JSONL with regenerated ``reasonings``.
- ``r0`` generates answer-blind reference reasoning from Q only and writes a
  one-method metric-format JSONL named ``Blind CoT`` by default.
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

from rcot_prompt import (  # noqa: E402
    AUGSUP_PROMPT,
    NEU_PROMPT,
    PG_SUP_PROMPT,
    QA_SUP_PROMPT,
    SSR_DENSE_PROMPT,
    SSR_PROMPT,
    SSR_SCHEMA_PROMPT,
    SUP_PROMPT,
)


METHOD_ORDER = ["NEU", "SUP", "AUG-SUP", "SSR"]
PROMPTS = {
    "NEU": NEU_PROMPT,
    "SUP": SUP_PROMPT,
    "AUG-SUP": AUGSUP_PROMPT,
    "SSR": SSR_PROMPT,
    "SSR-SCHEMA": SSR_SCHEMA_PROMPT,
    "SSR-DENSE": SSR_DENSE_PROMPT,
    "QA-SUP": QA_SUP_PROMPT,
    "PG-SUP": PG_SUP_PROMPT,
}

R0_PROMPT = (
    "You are an expert AI assistant. You will be given a user query only, "
    "without any final answer. Generate a concise, step-by-step reasoning "
    "process that could help solve the query from first principles. Do not "
    "assume or reveal any pre-existing final answer. Do not include the final "
    "answer itself. Output only the reasoning enclosed within <reason> and "
    "</reason> tags."
)

REASON_RE = re.compile(r"<reason>(.*?)</reason>", flags=re.IGNORECASE | re.DOTALL)
SKELETON_RE = re.compile(r"<skeleton>(.*?)</skeleton>", flags=re.IGNORECASE | re.DOTALL)
EXPLANATION_RE = re.compile(r"<\|begin_of_explanation\|>(.*?)<\|end_of_explanation\|>", flags=re.IGNORECASE | re.DOTALL)
JSON_RENDER_METHODS = {"SSR-SCHEMA"}
JSON_STEP_RANGES = {"SSR-SCHEMA": (6, 12)}
JSON_TAGS = {"PLAN", "RETR", "INFR", "EVAL", "SUMM", "BRCH", "RFLX", "BTRK"}
JSON_IMPORTANCE = {"HIGH", "LOW"}
JSON_BAD_RE = re.compile(
    r"<\/?(?:skeleton|reason)>|final output structure|skeleton_example|begin_of_|end_of_|^#+\s",
    flags=re.IGNORECASE | re.MULTILINE,
)
JSON_NUMBERED_ACTION_RE = re.compile(r"^\s*(?:\d+[\.)]\s*)?\[(?:PLAN|RETR|INFR|EVAL|SUMM|BRCH|RFLX|BTRK)\]", re.IGNORECASE)
JSON_LEADING_NUMBER_RE = re.compile(r"^\s*\d+[\.)]\s+")


@dataclass(frozen=True)
class Request:
    row_idx: int
    sample_id: Any
    method: str
    prompt: str
    prompt_context: str
    source_method: str


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
    return "Blind CoT"


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
    if context_method == "same" and method not in contexts:
        ssr_context = ensure_text(contexts.get("SSR"))
        if ssr_context.strip():
            return ssr_context
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
            completed[(int(row["row_idx"]), row.get("method", "R0"))] = row
    return completed


def build_rcot_requests(
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
            source_method = method if context_method == "same" else context_method
            if source_method not in row.get("contexts", {}) and "SSR" in row.get("contexts", {}):
                source_method = "SSR"
            prompt = apply_chat_template(tokenizer, PROMPTS[method], context)
            requests.append(
                Request(
                    row_idx=row_idx,
                    sample_id=row.get("id", row_idx),
                    method=method,
                    prompt=prompt,
                    prompt_context=context,
                    source_method=source_method,
                )
            )
    return requests


def build_r0_requests(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    method_name: str,
    limit: int | None,
) -> List[Request]:
    selected_rows = rows[:limit] if limit is not None else rows
    requests: List[Request] = []
    for row_idx, row in enumerate(selected_rows):
        source_method = first_method(row)
        question = ensure_text(row.get("questions", {}).get(source_method))
        if not question.strip():
            continue
        prompt = apply_chat_template(tokenizer, R0_PROMPT, question)
        requests.append(
            Request(
                row_idx=row_idx,
                sample_id=row.get("id", row_idx),
                method=method_name,
                prompt=prompt,
                prompt_context=question,
                source_method=source_method,
            )
        )
    return requests


def extract_block(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def extract_json_object(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(raw):
        if char != "{":
            continue
        try:
            _obj, end = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        return raw[idx : idx + end]
    return raw


def repetition_score(text: str, n: int = 4) -> float:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n * 2:
        return 0.0
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def validate_json_step(value: Any, index: int) -> tuple[str, str, str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"step_{index}_not_object")
    unexpected = set(value) - {"tag", "importance", "action", "reason"}
    missing = {"tag", "importance", "action", "reason"} - set(value)
    if unexpected:
        raise ValueError(f"step_{index}_unexpected_fields:{sorted(unexpected)}")
    if missing:
        raise ValueError(f"step_{index}_missing_fields:{sorted(missing)}")
    tag = ensure_text(value.get("tag")).strip().upper()
    importance = ensure_text(value.get("importance")).strip().upper()
    if importance == "MEDIUM":
        importance = "LOW"
    action = ensure_text(value.get("action")).strip()
    reason = ensure_text(value.get("reason")).strip()
    if tag not in JSON_TAGS:
        raise ValueError(f"step_{index}_bad_tag:{tag}")
    if importance not in JSON_IMPORTANCE:
        raise ValueError(f"step_{index}_bad_importance:{importance}")
    if not action:
        raise ValueError(f"step_{index}_empty_action")
    if not reason:
        raise ValueError(f"step_{index}_empty_reason")
    if len(action.split()) > 24:
        raise ValueError(f"step_{index}_action_too_long")
    if len(reason.split()) > 95:
        raise ValueError(f"step_{index}_reason_too_long")
    if JSON_BAD_RE.search(action) or JSON_BAD_RE.search(reason):
        raise ValueError(f"step_{index}_bad_markup")
    if JSON_NUMBERED_ACTION_RE.search(action) or JSON_LEADING_NUMBER_RE.search(action):
        raise ValueError(f"step_{index}_action_contains_prefix")
    if repetition_score(reason) > 0.35:
        raise ValueError(f"step_{index}_repetition")
    return tag, importance, action, reason


def render_json_reasoning(raw_output: str, method: str) -> tuple[str, str, str]:
    obj_text = extract_json_object(raw_output)
    data = json.loads(obj_text)
    if not isinstance(data, dict):
        raise ValueError("json_top_level_not_object")
    if set(data) != {"steps"}:
        raise ValueError(f"json_bad_top_level_keys:{sorted(data)}")
    steps = data.get("steps")
    if not isinstance(steps, list):
        raise ValueError("json_steps_not_list")
    min_steps, max_steps = JSON_STEP_RANGES.get(method, (6, 12))
    if not min_steps <= len(steps) <= max_steps:
        raise ValueError(f"json_bad_step_count:{len(steps)}")

    rendered: list[tuple[str, str, str, str]] = []
    high = 0
    run_tag = ""
    run_len = 0
    for idx, step in enumerate(steps, start=1):
        tag, importance, action, reason = validate_json_step(step, idx)
        rendered.append((tag, importance, action, reason))
        high += int(importance == "HIGH")
        if tag == run_tag:
            run_len += 1
            if run_len > 3:
                raise ValueError(f"json_repeated_tag_run:{tag}")
        else:
            run_tag = tag
            run_len = 1
    if high * 2 < len(rendered):
        raise ValueError("json_too_few_high_steps")

    all_reason = "\n".join(reason for *_prefix, reason in rendered)
    if repetition_score(all_reason) > 0.24:
        raise ValueError("json_global_repetition")

    skeleton = "\n".join(
        f"{idx}. [{tag}][{importance}] {action}"
        for idx, (tag, importance, action, _reason) in enumerate(rendered, start=1)
    )
    reason = "\n\n".join(reason for *_prefix, reason in rendered)
    return reason, skeleton, "json_rendered"


def parse_reasoning(raw_output: str, method: str = "") -> tuple[str, str, str]:
    if method in JSON_RENDER_METHODS:
        return render_json_reasoning(raw_output, method)
    raw = raw_output.strip()
    skeleton = extract_block(raw, SKELETON_RE)
    reason = extract_block(raw, REASON_RE)
    if reason:
        if method == "SSR-DENSE":
            paragraphs = [
                re.sub(r"^\s*\d+[.)]\s*", "", paragraph.strip())
                for paragraph in re.split(r"\n\s*\n", reason)
                if paragraph.strip()
            ]
            reason = "\n\n".join(paragraphs)
        return reason, skeleton, "tagged"
    explanation = extract_block(raw, EXPLANATION_RE)
    if explanation:
        return explanation, skeleton, "explanation_tagged"
    lower = raw.lower()
    if "<reason>" in lower:
        start = lower.index("<reason>") + len("<reason>")
        return raw[start:].strip(), skeleton, "open_reason_tag"
    return raw, skeleton, "raw"


def build_rcot_metric_rows(
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
        questions: Dict[str, str] = {}
        answers: Dict[str, str] = {}
        contexts: Dict[str, str] = {}
        missing = []
        detailed_reject = False
        for method in methods:
            item = completed.get((row_idx, method))
            if not item:
                missing.append(method)
                continue
            try:
                reasoning, skeleton, parse_mode = parse_reasoning(ensure_text(item.get("raw_output")), method)
            except (json.JSONDecodeError, ValueError) as exc:
                parse_mode = "json_rejected" if method in JSON_RENDER_METHODS else "parse_error"
                parse_counts[parse_mode] = parse_counts.get(parse_mode, 0) + 1
                missing.append(method)
                detailed_reject = True
                rejected.append(
                    {
                        "row_idx": row_idx,
                        "id": row.get("id", row_idx),
                        "method": method,
                        "reason": parse_mode,
                        "message": str(exc),
                    }
                )
                continue
            parse_counts[parse_mode] = parse_counts.get(parse_mode, 0) + 1
            if not reasoning.strip():
                missing.append(method)
                continue
            source_method = ensure_text(item.get("source_method")) or method
            questions[method] = (
                ensure_text(row.get("questions", {}).get(method))
                or ensure_text(row.get("questions", {}).get(source_method))
            )
            answers[method] = (
                ensure_text(row.get("answers", {}).get(method))
                or ensure_text(row.get("answers", {}).get(source_method))
            )
            contexts[method] = (
                ensure_text(row.get("contexts", {}).get(method))
                or ensure_text(row.get("contexts", {}).get(source_method))
                or ensure_text(item.get("prompt_context"))
            )
            reasonings[method] = reasoning
            if skeleton:
                skeletons[method] = skeleton
            parse_modes[method] = parse_mode
        if missing:
            if detailed_reject:
                continue
            rejected.append({"row_idx": row_idx, "id": row.get("id", row_idx), "missing": missing})
            continue
        out = {
            "id": row.get("id", row_idx),
            "questions": questions,
            "answers": answers,
            "contexts": contexts,
            "reasonings": reasonings,
            "generated_parse_modes": parse_modes,
        }
        if skeletons:
            out["generated_skeletons"] = skeletons
        metric_rows.append(out)
    summary = {
        "mode": "rcot",
        "input_rows": len(selected_rows),
        "complete_rows": len(metric_rows),
        "rejected_rows": len(rejected),
        "methods": methods,
        "parse_counts": parse_counts,
    }
    return metric_rows, rejected, summary


def build_r0_metric_rows(
    input_rows: List[Dict[str, Any]],
    completed: Dict[Tuple[int, str], Dict[str, Any]],
    limit: int | None,
    method_name: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    selected_rows = input_rows[:limit] if limit is not None else input_rows
    metric_rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    parse_counts: Dict[str, int] = {}
    for row_idx, row in enumerate(selected_rows):
        source_method = first_method(row)
        item = completed.get((row_idx, method_name))
        if not item:
            rejected.append({"row_idx": row_idx, "id": row.get("id", row_idx), "reason": "missing_raw_output"})
            continue
        reasoning, _skeleton, parse_mode = parse_reasoning(ensure_text(item.get("raw_output")), method_name)
        parse_counts[parse_mode] = parse_counts.get(parse_mode, 0) + 1
        if not reasoning.strip():
            rejected.append({"row_idx": row_idx, "id": row.get("id", row_idx), "reason": "empty_reasoning"})
            continue
        q = ensure_text(row.get("questions", {}).get(source_method))
        a = ensure_text(row.get("answers", {}).get(source_method))
        metric_rows.append(
            {
                "id": row.get("id", row_idx),
                "questions": {method_name: q},
                "answers": {method_name: a},
                "contexts": {method_name: q},
                "reasonings": {method_name: reasoning},
                "generated_parse_modes": {method_name: parse_mode},
            }
        )
    summary = {
        "mode": "r0",
        "input_rows": len(selected_rows),
        "complete_rows": len(metric_rows),
        "rejected_rows": len(rejected),
        "method": method_name,
        "parse_counts": parse_counts,
    }
    return metric_rows, rejected, summary


def build_requests(args: argparse.Namespace, rows: List[Dict[str, Any]], tokenizer: Any) -> tuple[List[Request], List[str]]:
    if args.mode == "rcot":
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in methods if m not in PROMPTS]
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}; known={sorted(PROMPTS)}")
        return build_rcot_requests(rows, tokenizer, methods, args.prompt_context_method, args.limit), methods
    return build_r0_requests(rows, tokenizer, args.method_name, args.limit), [args.method_name]


def build_metric_rows(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    completed: Dict[Tuple[int, str], Dict[str, Any]],
    methods: List[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if args.mode == "rcot":
        return build_rcot_metric_rows(rows, completed, methods, args.limit)
    return build_r0_metric_rows(rows, completed, args.limit, args.method_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["rcot", "r0"], default="rcot")
    parser.add_argument("--input", type=Path, default=ROOT / "scripts" / "examples" / "examples.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Metric-format JSONL with generated reasonings")
    parser.add_argument("--raw-output", type=Path, required=True, help="Resumable raw generation JSONL")
    parser.add_argument("--rejected-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--model", type=Path, help="Generator checkpoint (required unless --build-only)")
    parser.add_argument("--methods", default=",".join(METHOD_ORDER), help="Comma-separated methods for --mode rcot")
    parser.add_argument("--method-name", default="Blind CoT", help="Output method name for --mode r0")
    parser.add_argument("--prompt-context-method", default="same", help="'same' or one method name reused as prompt context")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile/cudagraph capture in vLLM")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--max-seq-len-to-capture", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--build-only", action="store_true", help="Only rebuild parsed metric rows from existing raw output")
    args = parser.parse_args()
    if not args.build_only and args.model is None:
        parser.error("--model is required unless --build-only is set")

    rows = read_jsonl(args.input)
    tokenizer = None
    if not args.build_only:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise SystemExit("transformers is required. Install the project environment with `uv pip install -e .`.") from exc
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    if args.mode == "rcot":
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in methods if m not in PROMPTS]
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}; known={sorted(PROMPTS)}")
        requests = [] if args.build_only else build_rcot_requests(rows, tokenizer, methods, args.prompt_context_method, args.limit)
    else:
        methods = [args.method_name]
        requests = [] if args.build_only else build_r0_requests(rows, tokenizer, args.method_name, args.limit)

    completed = load_completed(args.raw_output)
    if not args.build_only:
        pending = [req for req in requests if (req.row_idx, req.method) not in completed]
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "input": str(args.input),
                    "model": str(args.model),
                    "requests": len(requests),
                    "completed": len(completed),
                    "pending": len(pending),
                    "methods": methods,
                    "prompt_context_method": args.prompt_context_method if args.mode == "rcot" else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
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
                max_num_seqs=args.max_num_seqs,
                enforce_eager=args.enforce_eager,
                disable_custom_all_reduce=args.disable_custom_all_reduce,
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
                    raw_rows.append(
                        {
                            "row_idx": req.row_idx,
                            "id": req.sample_id,
                            "method": req.method,
                            "raw_output": text,
                            "finish_reason": finish_reason,
                            "output_tokens": len(token_ids),
                            "prompt_chars": len(req.prompt),
                            "prompt_context_chars": len(req.prompt_context),
                            "source_method": req.source_method,
                        }
                    )
                append_jsonl(args.raw_output, raw_rows)
                completed.update({(int(r["row_idx"]), r["method"]): r for r in raw_rows})
                print(
                    json.dumps(
                        {
                            "chunk_start": start,
                            "chunk_size": len(chunk),
                            "completed": len(completed),
                            "raw_output": str(args.raw_output),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    completed = load_completed(args.raw_output)
    metric_rows, rejected, summary = build_metric_rows(args, rows, completed, methods)
    write_jsonl(args.output, metric_rows)
    rejected_output = args.rejected_output or args.output.with_suffix(".rejected.jsonl")
    summary_output = args.summary_output or args.output.with_suffix(".summary.json")
    write_jsonl(rejected_output, rejected)
    summary.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "raw_output": str(args.raw_output),
            "rejected_output": str(rejected_output),
        }
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
