#!/usr/bin/env python3
"""Run the final SSR anchoring metrics and behavioral-zone analysis."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PREPARE = SCRIPTS / "prepare_anchoring_input.py"
PROB_TRAJ = SCRIPTS / "anchoring_measure" / "prob_lex_traj_metrics.py"
TRAJ = SCRIPTS / "anchoring_measure" / "commitment_kl.py"
CONTROLS = SCRIPTS / "anchoring_measure" / "build_controlled_traj_anchors.py"
PLOT = SCRIPTS / "anchoring_measure" / "plot_paper_style_behavior_zones.py"
REPORT = SCRIPTS / "report_anchoring_results.py"

DEFAULT_METHODS = ["NEU", "SUP", "AUG-SUP", "SSR"]
CONTROL_METHODS = ["Blind CoT", "+Prob Anchor", "+Traj Anchor", "Response-as-CoT"]


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def distributed_command(python: Path, nproc: int, script: Path, arguments: list[str]) -> list[str]:
    if nproc <= 1:
        return [str(python), str(script), *arguments]
    torchrun = python.parent / "torchrun"
    if not torchrun.exists():
        raise FileNotFoundError(f"torchrun not found next to Python interpreter: {torchrun}")
    return [
        str(torchrun),
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(script),
        *arguments,
    ]


def validate_input(python: Path, source: Path, output: Path, methods: list[str]) -> None:
    run(
        [
            str(python),
            str(PREPARE),
            "validate",
            "--input",
            str(source),
            "--output",
            str(output),
            "--methods",
            ",".join(methods),
            "--strict",
        ]
    )


def score_group(
    args: argparse.Namespace,
    input_path: Path,
    methods: list[str],
    group: str,
) -> Path:
    group_dir = args.run_dir / "metrics" / group
    prob_dir = group_dir / "prob"
    traj_dir = group_dir / "traj"
    result_dir = args.run_dir / "results" / group
    prob_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    methods_csv = ",".join(methods)

    run(
        distributed_command(
            args.python,
            args.torchrun_nproc,
            PROB_TRAJ,
            [
                "score-prob",
                "--input",
                str(input_path),
                "--scoring-model",
                str(args.model),
                "--output-dir",
                str(prob_dir),
                "--methods",
                methods_csv,
            ],
        )
    )
    run(
        distributed_command(
            args.python,
            args.torchrun_nproc,
            TRAJ,
            [
                "score",
                "--input",
                str(input_path),
                "--scoring-model",
                str(args.model),
                "--output-dir",
                str(traj_dir),
                "--methods",
                methods_csv,
            ],
        )
    )
    run(
        [
            str(args.python),
            str(PROB_TRAJ),
            "aggregate",
            "--input",
            str(input_path),
            "--prob-metrics",
            str(prob_dir),
            "--traj-metrics",
            str(traj_dir),
            "--output-dir",
            str(result_dir),
            "--methods",
            methods_csv,
        ]
    )
    return result_dir / "anchoring_metrics_per_record.csv"


def merge_csv(inputs: list[Path], output: Path) -> None:
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    for path in inputs:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in fieldnames:
                    fieldnames.append(field)
            rows.extend(reader)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_run_dir(run_dir: Path, force: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        if not force:
            raise FileExistsError(f"Run directory is not empty: {run_dir}; pass --force to replace it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Metric-format main-method JSONL")
    parser.add_argument("--blind-input", type=Path, help="Independent question-only traces for controlled zones")
    parser.add_argument("--blind-method", default="Blind CoT")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--model", type=Path, help="Local scorer checkpoint (required unless --dry-run)")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "anchoring")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--torchrun-nproc", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and build controls without model scoring")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.run_dir = args.run_dir.resolve()
    args.python = args.python.resolve()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods:
        parser.error("--methods must contain at least one method")
    if args.torchrun_nproc < 1:
        parser.error("--torchrun-nproc must be positive")
    if not args.dry_run and args.model is None:
        parser.error("--model is required unless --dry-run is set")

    prepare_run_dir(args.run_dir, args.force)
    inputs_dir = args.run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    method_input = inputs_dir / "methods.metric.jsonl"
    validate_input(args.python, args.input.resolve(), method_input, methods)

    controlled_input: Path | None = None
    if args.blind_input:
        blind_validated = inputs_dir / "blind.metric.jsonl"
        validate_input(args.python, args.blind_input.resolve(), blind_validated, [args.blind_method])
        controlled_input = inputs_dir / "controlled.metric.jsonl"
        run(
            [
                str(args.python),
                str(CONTROLS),
                "--input",
                str(blind_validated),
                "--output",
                str(controlled_input),
                "--blind-method",
                args.blind_method,
            ]
        )

    manifest = {
        "input": str(args.input.resolve()),
        "blind_input": str(args.blind_input.resolve()) if args.blind_input else None,
        "methods": methods,
        "model": str(args.model.resolve()) if args.model else None,
        "metric_definitions": {
            "A_lex": "question-filtered IDF-weighted answer-content recall",
            "A_traj": "sampled-prefix next-token entropy reduction from answer visibility",
            "A_prob": "clipped normalized answer-surprisal reduction",
        },
    }
    if args.dry_run:
        (args.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({**manifest, "dry_run": True}, indent=2))
        return

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    method_csv = score_group(args, method_input, methods, "methods")
    if controlled_input is not None:
        controlled_csv = score_group(args, controlled_input, CONTROL_METHODS, "controlled")
        combined_csv = args.run_dir / "results" / "all_metrics_per_record.csv"
        merge_csv([method_csv, controlled_csv], combined_csv)
        run(
            [
                str(args.python),
                str(PLOT),
                "--csv",
                str(combined_csv),
                "--out-dir",
                str(args.run_dir / "figures"),
                "--prefix",
                "behavioral_zones",
            ],
            env=env,
        )
        manifest["behavioral_zones"] = str(args.run_dir / "figures")
    else:
        manifest["behavioral_zones"] = "skipped: provide --blind-input"

    (args.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run([str(args.python), str(REPORT), "--run-dir", str(args.run_dir)])


if __name__ == "__main__":
    main()
