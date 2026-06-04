#!/usr/bin/env python3
"""Run the SSR anchoring-analysis pipeline.

This wrapper provides a CLI around data preparation, metric computation,
anchoring figures and tables, and report metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
PREPARE_SCRIPT = SCRIPTS_DIR / "prepare_anchoring_input.py"

METHOD_ORDER = ["NEU", "SUP", "AUG-SUP", "SSR"]
CONTROL_ORDER = ["Real CoT", "+Prob Anchor", "+Entropy Anchor", "Response as CoT"]

def run(cmd: List[str], *, cwd: Path | None = None, env: Dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd or ROOT), env=env, check=True)


def read_metrics(path_or_dir: Path) -> List[Dict[str, Any]]:
    files = [path_or_dir] if path_or_dir.is_file() else sorted(path_or_dir.glob("metrics_rank*.jsonl"))
    if not files and path_or_dir.is_dir():
        files = sorted(path_or_dir.glob("*.jsonl"))
    records: List[Dict[str, object]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        raise FileNotFoundError(f"No metric records found under {path_or_dir}")
    return records

def metric_files_complete(output_dir: Path, expected_methods: List[str]) -> bool:
    try:
        records = read_metrics(output_dir)
    except FileNotFoundError:
        return False
    seen = {r.get("method") for r in records}
    return set(expected_methods).issubset(seen)


def run_metrics(args: argparse.Namespace, metric_input: Path, output_dir: Path, methods: List[str]) -> None:
    if args.skip_metrics:
        if not metric_files_complete(output_dir, methods):
            raise FileNotFoundError(f"--skip-metrics requested but {output_dir} is incomplete")
        return
    if output_dir.exists() and not args.force:
        if metric_files_complete(output_dir, methods):
            print(f"[reuse] metrics already exist: {output_dir}")
            return
        raise FileExistsError(f"Output dir exists but is incomplete: {output_dir}; pass --force to overwrite")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.python),
        "-m",
        "anchoring_measure.metrics",
        "--input",
        str(metric_input),
        "--scoring-model",
        str(args.model),
        "--output-dir",
        str(output_dir),
    ]
    if args.disable_shared_prefix_past:
        cmd.append("--disable-shared-prefix-past")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    if args.torchrun_nproc > 1:
        torchrun = Path(args.python).parent / "torchrun"
        cmd = [
            str(torchrun),
            "--standalone",
            f"--nproc_per_node={args.torchrun_nproc}",
            "-m",
            "anchoring_measure.metrics",
            "--input",
            str(metric_input),
            "--scoring-model",
            str(args.model),
            "--output-dir",
            str(output_dir),
        ]
        if args.disable_shared_prefix_past:
            cmd.append("--disable-shared-prefix-past")
    if args.run_and_hold:
        cmd = ["bash", str(args.run_and_hold), args.gpus, *cmd]
    run(cmd, cwd=ROOT, env=env)


def plot_results(args: argparse.Namespace, metrics_dir: Path, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python),
        "-m",
        "anchoring_measure.plot",
        "--metrics",
        str(metrics_dir),
        "--out-dir",
        str(output_dir),
        "--prefix",
        prefix,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    run(cmd, cwd=ROOT, env=env)


def prepare_inputs(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path]:
    metric_input = run_dir / "inputs" / "example.metric.jsonl"
    controlled_input = run_dir / "inputs" / "controlled_reference.metric.jsonl"
    methods = ",".join(METHOD_ORDER)
    if args.force or not metric_input.exists():
        run([
            str(args.python),
            str(PREPARE_SCRIPT),
            "validate",
            "--input",
            str(args.input),
            "--output",
            str(metric_input),
            "--methods",
            methods,
            "--strict",
        ])
    if args.force or not controlled_input.exists():
        run([
            str(args.python),
            str(PREPARE_SCRIPT),
            "controlled-reference",
            "--input",
            str(metric_input),
            "--output",
            str(controlled_input),
            "--base-method",
            args.control_base_method,
        ])
    return metric_input, controlled_input


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SCRIPTS_DIR / "examples" / "examples.jsonl")
    parser.add_argument("--model", type=Path, default=Path("/path/to/Qwen3-8B"))
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "anchoring_example")
    parser.add_argument("--python", type=Path, default=ROOT / "venv" / "bin" / "python")
    parser.add_argument("--torchrun-nproc", type=int, default=4)
    parser.add_argument("--run-and-hold", type=Path, default=None)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--control-base-method", default="NEU")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--skip-controlled-metrics", action="store_true")
    parser.add_argument("--disable-shared-prefix-past", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metric_input, controlled_input = prepare_inputs(args, run_dir)

    method_metrics_dir = run_dir / "metrics" / "methods"
    control_metrics_dir = run_dir / "metrics" / "controlled"

    run_metrics(args, metric_input, method_metrics_dir, METHOD_ORDER)
    plot_results(args, method_metrics_dir, run_dir / "figures" / "methods", "methods")

    if not args.skip_controlled_metrics:
        control_args = argparse.Namespace(**vars(args))
        control_args.skip_metrics = False
        run_metrics(control_args, controlled_input, control_metrics_dir, CONTROL_ORDER)
        plot_results(args, control_metrics_dir, run_dir / "figures" / "controlled", "controlled")

    manifest = {
        "input": str(args.input),
        "model": str(args.model),
        "run_dir": str(run_dir),
        "python": str(args.python),
        "method_metrics": str(method_metrics_dir),
        "controlled_metrics": str(control_metrics_dir),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
