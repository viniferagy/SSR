#!/usr/bin/env python3
"""Plan one-pass SSR-plus experiments without running them by default.

The script has two purposes:

1. Write a runnable command script for generating and evaluating plus variants.
2. Merge completed plus-method JSONL files with an existing multi-method
   metric JSONL once generation has been run.

By default the script only writes commands and metadata. Pass ``--execute`` to
run the planned shell script explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
BASE_METHODS = ["NEU", "SUP", "AUG-SUP", "SSR"]
DEFAULT_PLUS_METHODS = "SSR_PLUS,SSR_PLUS_STRUCT"


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


def shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def parse_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def method_order(plus_methods: List[str]) -> str:
    out = BASE_METHODS[:]
    for method in plus_methods:
        if method not in out:
            out.append(method)
    return ",".join(out)


def delta_pairs(plus_methods: List[str]) -> str:
    pairs = ["SUP-NEU", "AUG-SUP-NEU", "SSR-NEU"]
    for method in plus_methods:
        pairs.append(f"{method}-NEU")
        pairs.append(f"{method}-SSR")
    if len(plus_methods) > 1:
        anchor = plus_methods[0]
        for method in plus_methods[1:]:
            pairs.append(f"{method}-{anchor}")
    return ",".join(pairs)


def run_line(cmd: List[str], env_prefix: str = "") -> str:
    body = shell_join(cmd)
    return f"{env_prefix}{body}" if env_prefix else body


def abspath(path: Path) -> Path:
    return Path(os.path.abspath(path))


def merge_ssr_plus(args: argparse.Namespace) -> None:
    base_rows = read_jsonl(args.base_input)
    plus_input = args.plus_input or args.ssr_plus_input
    if plus_input is None:
        raise ValueError("merge requires --plus-input or --ssr-plus-input")
    plus_rows = read_jsonl(plus_input)
    if len(base_rows) != len(plus_rows):
        raise ValueError(f"row count mismatch: base={len(base_rows)} plus={len(plus_rows)}")

    plus_methods = parse_csv(args.plus_methods) if args.plus_methods else [args.output_method]

    out_rows: List[Dict[str, Any]] = []
    for idx, (base, plus) in enumerate(zip(base_rows, plus_rows)):
        row = json.loads(json.dumps(base, ensure_ascii=False))
        for method in plus_methods:
            src_method = args.ssr_plus_source_method if len(plus_methods) == 1 else method
            out_method = args.output_method if len(plus_methods) == 1 and args.output_method else method

            question = ensure_text(plus.get("questions", {}).get(src_method))
            answer = ensure_text(plus.get("answers", {}).get(src_method))
            context = ensure_text(plus.get("contexts", {}).get(src_method))
            reasoning = ensure_text(plus.get("reasonings", {}).get(src_method))
            if not reasoning.strip():
                raise ValueError(f"missing {src_method} reasoning at row {idx}")

            row.setdefault("questions", {})[out_method] = question or ensure_text(base.get("questions", {}).get("SSR"))
            row.setdefault("answers", {})[out_method] = answer or ensure_text(base.get("answers", {}).get("SSR"))
            row.setdefault("contexts", {})[out_method] = context or ensure_text(base.get("contexts", {}).get("SSR"))
            row.setdefault("reasonings", {})[out_method] = reasoning

            skeleton = ensure_text(plus.get("generated_skeletons", {}).get(src_method))
            if skeleton.strip():
                row.setdefault("generated_skeletons", {})[out_method] = skeleton
            parse_mode = ensure_text(plus.get("generated_parse_modes", {}).get(src_method))
            if parse_mode.strip():
                row.setdefault("generated_parse_modes", {})[out_method] = parse_mode
        out_rows.append(row)

    n = write_jsonl(args.output, out_rows)
    manifest = {
        "mode": "merge",
        "base_input": str(args.base_input),
        "plus_input": str(plus_input),
        "output": str(args.output),
        "rows": n,
        "plus_methods": plus_methods,
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_plan(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = args.run_dir.resolve()
    input_path = args.input.resolve()
    base_methods_input = args.base_methods_input.resolve()
    model = args.model.resolve()
    python_vllm = abspath(args.python_vllm)
    python_metrics = abspath(args.python_metrics)
    run_and_hold = args.run_and_hold.resolve() if args.run_and_hold else None
    plus_methods = parse_csv(args.plus_methods)
    methods_csv = method_order(plus_methods)
    deltas_csv = delta_pairs(plus_methods)

    generation_dir = run_dir / "generation"
    inputs_dir = run_dir / "inputs"
    metrics_dir = run_dir / "metrics"
    results_dir = run_dir / "results"
    deep_dir = run_dir / "deep"
    swap_dir = run_dir / "answer_swap"

    plus_input = inputs_dir / "plus_methods.metric.jsonl"
    merged_input = inputs_dir / "methods_with_plus.metric.jsonl"
    plus_raw = generation_dir / "plus_raw.jsonl"
    plus_summary = inputs_dir / "plus.summary.json"
    plus_rejected = inputs_dir / "plus.rejected.jsonl"
    merge_manifest = inputs_dir / "merge_plus_manifest.json"

    env_prefix = ""
    if args.cuda_lib_dir:
        cuda_lib_dir = args.cuda_lib_dir.resolve()
        env_prefix = (
            f"LIBRARY_PATH={shlex.quote(str(cuda_lib_dir))}:${{LIBRARY_PATH:-}} "
            f"LD_LIBRARY_PATH={shlex.quote(str(cuda_lib_dir))}:${{LD_LIBRARY_PATH:-}} "
        )

    gen_cmd = [
        str(python_vllm),
        "scripts/generate_vllm.py",
        "--mode",
        "rcot",
        "--input",
        str(input_path),
        "--output",
        str(plus_input),
        "--raw-output",
        str(plus_raw),
        "--rejected-output",
        str(plus_rejected),
        "--summary-output",
        str(plus_summary),
        "--model",
        str(model),
        "--methods",
        ",".join(plus_methods),
        "--prompt-context-method",
        args.prompt_context_method,
        "--chunk-size",
        str(args.chunk_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
    ]
    if args.limit is not None:
        gen_cmd.extend(["--limit", str(args.limit)])
    if args.max_num_seqs is not None:
        gen_cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if args.enforce_eager:
        gen_cmd.append("--enforce-eager")
    if args.disable_custom_all_reduce:
        gen_cmd.append("--disable-custom-all-reduce")
    if run_and_hold:
        gen_cmd = ["bash", str(run_and_hold), args.gpus, *gen_cmd]

    merge_cmd = [
        str(python_metrics),
        "scripts/plan_ssr_plus_experiment.py",
        "merge",
        "--base-input",
        str(base_methods_input),
        "--plus-input",
        str(plus_input),
        "--plus-methods",
        ",".join(plus_methods),
        "--output",
        str(merged_input),
        "--manifest",
        str(merge_manifest),
    ]

    score_cmd = [
        str(python_metrics),
        "-m",
        "anchoring_measure.reviewer_protocol",
        "score",
        "--input",
        str(merged_input),
        "--paraphrases",
        str(args.paraphrases.resolve()),
        "--scoring-model",
        str(model),
        "--output-dir",
        str(metrics_dir / "methods"),
        "--methods",
        methods_csv,
    ]
    if run_and_hold:
        score_cmd = ["bash", str(run_and_hold), args.gpus, *score_cmd]

    aggregate_cmd = [
        str(python_metrics),
        "-m",
        "anchoring_measure.reviewer_protocol",
        "aggregate",
        "--method-metrics",
        str(metrics_dir / "methods"),
        "--blind-metrics",
        str(args.blind_metrics.resolve()),
        "--control-metrics",
        str(args.control_metrics.resolve()),
        "--output-dir",
        str(results_dir),
        "--method-order",
        methods_csv,
        "--zone-quantile",
        str(args.zone_quantile),
        "--uninformative-quantile",
        str(args.uninformative_quantile),
        "--bootstrap",
        str(args.bootstrap),
    ]

    deep_cmd = [
        str(python_metrics),
        "scripts/analyze_deep_anchoring_metrics.py",
        "--input",
        str(merged_input),
        "--output-dir",
        str(deep_dir / "compression"),
        "--methods",
        methods_csv,
        "--delta-pairs",
        deltas_csv,
        "--bootstrap",
        str(args.bootstrap),
    ]

    suppression_cmd = [
        str(python_metrics),
        "scripts/analyze_suppression_support.py",
        "--input",
        str(merged_input),
        "--output-dir",
        str(deep_dir / "suppression_support"),
        "--excess",
        str(results_dir / "method_excess_metrics.jsonl"),
        "--delta-pairs",
        deltas_csv,
        "--bootstrap",
        str(args.bootstrap),
    ]

    swap_build_cmd = [
        str(python_metrics),
        "scripts/analyze_answer_swap_sensitivity.py",
        "build",
        "--input",
        str(merged_input),
        "--output",
        str(swap_dir / "swap_input.metric.jsonl"),
        "--summary-output",
        str(swap_dir / "swap_input.summary.json"),
        "--methods",
        methods_csv,
        "--seed",
        str(args.seed),
    ]

    swap_gen_cmd = [
        str(python_vllm),
        "scripts/generate_vllm.py",
        "--mode",
        "rcot",
        "--input",
        str(swap_dir / "swap_input.metric.jsonl"),
        "--output",
        str(swap_dir / "swap_generated.metric.jsonl"),
        "--raw-output",
        str(swap_dir / "swap_raw.jsonl"),
        "--model",
        str(model),
        "--methods",
        methods_csv,
        "--chunk-size",
        str(args.chunk_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
    ]
    if args.max_num_seqs is not None:
        swap_gen_cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if args.enforce_eager:
        swap_gen_cmd.append("--enforce-eager")
    if args.disable_custom_all_reduce:
        swap_gen_cmd.append("--disable-custom-all-reduce")
    if run_and_hold:
        swap_gen_cmd = ["bash", str(run_and_hold), args.gpus, *swap_gen_cmd]

    swap_score_cmd = [
        str(python_metrics),
        "scripts/analyze_answer_swap_sensitivity.py",
        "score",
        "--original",
        str(merged_input),
        "--swap",
        str(swap_dir / "swap_generated.metric.jsonl"),
        "--output-dir",
        str(swap_dir / "results"),
        "--methods",
        methods_csv,
        "--delta-pairs",
        deltas_csv,
        "--bootstrap",
        str(args.bootstrap),
    ]
    if args.no_swap_embeddings:
        swap_score_cmd.append("--no-embeddings")

    commands = [
        ("generate_plus_methods", run_line(gen_cmd, env_prefix)),
        ("merge_plus_methods_input", run_line(merge_cmd)),
        ("score_reviewer_protocol", run_line(score_cmd)),
        ("aggregate_reviewer_protocol", run_line(aggregate_cmd)),
        ("compression_diagnostics", run_line(deep_cmd)),
        ("suppression_support_diagnostics", run_line(suppression_cmd)),
        ("build_answer_swap_input", run_line(swap_build_cmd)),
        ("generate_answer_swap_traces", run_line(swap_gen_cmd, env_prefix)),
        ("score_answer_swap", run_line(swap_score_cmd)),
    ]

    return {
        "run_dir": str(run_dir),
        "input": str(input_path),
        "base_methods_input": str(base_methods_input),
        "model": str(model),
        "plus_methods": plus_methods,
        "method_order": methods_csv,
        "delta_pairs": deltas_csv,
        "artifacts": {
            "plus_metric_input": str(plus_input),
            "merged_method_input": str(merged_input),
            "reviewer_results": str(results_dir),
            "compression_results": str(deep_dir / "compression"),
            "suppression_support_results": str(deep_dir / "suppression_support"),
            "answer_swap_results": str(swap_dir / "results"),
        },
        "commands": commands,
    }


def plan(args: argparse.Namespace) -> None:
    plan_data = build_plan(args)
    run_dir = Path(plan_data["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    commands_path = args.commands_output or (run_dir / "commands.sh")
    manifest_path = args.manifest or (run_dir / "plan_manifest.json")
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(str(ROOT / 'scripts'))}:${{PYTHONPATH:-}}",
        "",
        "# Generated by scripts/plan_ssr_plus_experiment.py.",
        "# This script is not executed unless --execute is passed to the planner or you run it manually.",
        "",
    ]
    for name, command in plan_data["commands"]:
        lines.extend([f"echo '=== {name} ==='", command, ""])
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.write_text("\n".join(lines), encoding="utf-8")
    commands_path.chmod(0o755)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(plan_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "mode": "plan",
        "commands": str(commands_path),
        "manifest": str(manifest_path),
        "execute": args.execute,
    }, ensure_ascii=False, indent=2))

    if args.execute:
        subprocess.run(["bash", str(commands_path)], cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--input", type=Path, default=ROOT / "runs/final_reviewer_protocol_1k/inputs_filtered/base_1k.metric.jsonl")
    p.add_argument("--base-methods-input", type=Path, default=ROOT / "runs/final_reviewer_protocol_1k/full/inputs/methods_1k.metric.jsonl")
    p.add_argument("--paraphrases", type=Path, default=ROOT / "runs/final_reviewer_protocol_1k/full/inputs/paraphrases_1k.jsonl")
    p.add_argument("--blind-metrics", type=Path, default=ROOT / "runs/final_reviewer_protocol_1k/full/metrics/blind")
    p.add_argument("--control-metrics", type=Path, default=ROOT / "runs/final_reviewer_protocol_1k/full/metrics/controlled")
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs/ssr_plus_qwen3_8b_1k")
    p.add_argument("--model", type=Path, default=Path("/home/pengguangyue/workspace/models/Qwen/Qwen3-8B"))
    p.add_argument("--plus-methods", default=DEFAULT_PLUS_METHODS)
    p.add_argument("--python-vllm", type=Path, default=ROOT / "venv-vllm-cu124/bin/python")
    p.add_argument("--python-metrics", type=Path, default=ROOT / "venv/bin/python")
    p.add_argument("--run-and-hold", type=Path, default=Path("/home/pengguangyue/workspace/proj/run_and_hold.sh"))
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--cuda-lib-dir", type=Path, default=ROOT / ".cuda-lib")
    p.add_argument("--prompt-context-method", default="SSR")
    p.add_argument("--limit", type=int)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--max-num-seqs", type=int)
    p.add_argument("--enforce-eager", dest="enforce_eager", action="store_true", default=True)
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    p.add_argument("--disable-custom-all-reduce", dest="disable_custom_all_reduce", action="store_true", default=True)
    p.add_argument("--enable-custom-all-reduce", dest="disable_custom_all_reduce", action="store_false")
    p.add_argument("--zone-quantile", type=float, default=0.90)
    p.add_argument("--uninformative-quantile", type=float, default=0.10)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-swap-embeddings", action="store_true")
    p.add_argument("--commands-output", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--execute", action="store_true", help="Actually run the generated command script.")
    p.set_defaults(func=plan)

    p = sub.add_parser("merge")
    p.add_argument("--base-input", type=Path, required=True)
    p.add_argument("--plus-input", type=Path)
    p.add_argument("--plus-methods", default="")
    p.add_argument("--ssr-plus-input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--ssr-plus-source-method", default="SSR_PLUS")
    p.add_argument("--output-method", default="SSR_PLUS")
    p.set_defaults(func=merge_ssr_plus)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
