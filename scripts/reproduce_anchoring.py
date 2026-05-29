#!/usr/bin/env python3
"""Run or summarize the SSR anchoring-analysis reproduction.

This wrapper provides a reproducible CLI around data preparation, metric
computation, table generation, plotting, and paper comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
PREPARE_SCRIPT = SCRIPTS_DIR / "prepare_anchoring_input.py"

METHOD_ORDER = ["NEU", "SUP", "AUG-SUP", "SSR"]
CONTROL_ORDER = ["Real CoT", "+Prob Anchor", "+Entropy Anchor", "Response as CoT"]

METRIC_KEYS = {
    "Alex": "lexical_anchoring",
    "Aent": "entropy_anchoring",
    "Aprob": "ProbabilisticAnchoring",
}

ZONE_KEYS = ["Reason", "Encode", "Cloze", "Copy"]

PAPER = {
    "qwen3_4b_thinking_2507": {
        "metrics": {
            "NEU": {"Alex": 48.5, "Aent": 55.9, "Aprob": 37.0},
            "SUP": {"Alex": 45.3, "Aent": 57.4, "Aprob": 43.6},
            "AUG-SUP": {"Alex": 42.9, "Aent": 58.4, "Aprob": 44.4},
            "SSR": {"Alex": 30.8, "Aent": 47.2, "Aprob": 34.1},
        },
        "zones": {
            "NEU": {"Reason": 12.7, "Encode": 4.0, "Cloze": 54.0, "Copy": 29.3},
            "SUP": {"Reason": 18.2, "Encode": 9.3, "Cloze": 46.6, "Copy": 25.9},
            "AUG-SUP": {"Reason": 5.2, "Encode": 7.6, "Cloze": 44.5, "Copy": 42.7},
            "SSR": {"Reason": 51.2, "Encode": 5.1, "Cloze": 22.9, "Copy": 20.8},
        },
    },
    "qwen3_8b": {
        "metrics": {
            "NEU": {"Alex": 46.3, "Aent": 52.5, "Aprob": 48.5},
            "SUP": {"Alex": 43.1, "Aent": 50.8, "Aprob": 54.1},
            "AUG-SUP": {"Alex": 38.3, "Aent": 51.4, "Aprob": 53.7},
            "SSR": {"Alex": 29.7, "Aent": 45.4, "Aprob": 45.2},
        },
        "zones": {
            "NEU": {"Reason": 24.9, "Encode": 8.2, "Cloze": 35.7, "Copy": 31.1},
            "SUP": {"Reason": 27.1, "Encode": 20.0, "Cloze": 25.7, "Copy": 27.1},
            "AUG-SUP": {"Reason": 28.4, "Encode": 17.3, "Cloze": 21.0, "Copy": 33.3},
            "SSR": {"Reason": 46.4, "Encode": 9.8, "Cloze": 20.7, "Copy": 23.2},
        },
    },
}


def run(cmd: List[str], *, cwd: Path | None = None, env: Dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd or ROOT), env=env, check=True)


def read_metrics(path_or_dir: Path) -> List[Dict[str, Any]]:
    files = [path_or_dir] if path_or_dir.is_file() else sorted(path_or_dir.glob("metrics_rank*.jsonl"))
    if not files and path_or_dir.is_dir():
        files = sorted(path_or_dir.glob("*.jsonl"))
    records: List[Dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        raise FileNotFoundError(f"No metric records found under {path_or_dir}")
    return records


def summarize(records: Iterable[Dict[str, Any]], method_order: List[str]) -> Dict[str, Any]:
    values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        method = row.get("method", "unknown")
        for label, key in METRIC_KEYS.items():
            if key in row and np.isfinite(row[key]):
                values[method][label].append(float(row[key]))
        if "entropy_anchoring" in row and "ProbabilisticAnchoring" in row:
            values[method]["_x"].append(float(row["entropy_anchoring"]))
            values[method]["_y"].append(float(row["ProbabilisticAnchoring"]))

    summary: Dict[str, Any] = {}
    ordered = [m for m in method_order if m in values] + [m for m in values if m not in method_order]
    for method in ordered:
        metric = {
            label: 100.0 * float(np.mean(values[method][label]))
            for label in METRIC_KEYS
            if values[method].get(label)
        }
        x = np.array(values[method].get("_x", []), dtype=float)
        y = np.array(values[method].get("_y", []), dtype=float)
        zones = {}
        if len(x) and len(x) == len(y):
            zones = {
                "Reason": 100.0 * float(np.mean((x < 0.5) & (y < 0.5))),
                "Encode": 100.0 * float(np.mean((x < 0.5) & (y >= 0.5))),
                "Cloze": 100.0 * float(np.mean((x >= 0.5) & (y < 0.5))),
                "Copy": 100.0 * float(np.mean((x >= 0.5) & (y >= 0.5))),
            }
        summary[method] = {"N": int(len(values[method].get("Alex", []))), "metrics": metric, "zones": zones}
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    def fmt(v: Any) -> str:
        return f"{v:.1f}" if isinstance(v, float) else str(v)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out) + "\n"


def write_tables(summary: Dict[str, Any], out_dir: Path, paper_key: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_rows = []
    zone_rows = []
    for method, item in summary.items():
        metric_rows.append({
            "Method": method,
            "N": item["N"],
            "Alex": item["metrics"].get("Alex", np.nan),
            "Aent": item["metrics"].get("Aent", np.nan),
            "Aprob": item["metrics"].get("Aprob", np.nan),
        })
        zone_rows.append({
            "Method": method,
            "N": item["N"],
            **{k: item["zones"].get(k, np.nan) for k in ZONE_KEYS},
        })

    write_csv(out_dir / "table1_metrics.csv", metric_rows, ["Method", "N", "Alex", "Aent", "Aprob"])
    write_csv(out_dir / "table2_zones.csv", zone_rows, ["Method", "N", *ZONE_KEYS])

    metric_md_rows = [[r["Method"], r["N"], r["Alex"], r["Aent"], r["Aprob"]] for r in metric_rows]
    zone_md_rows = [[r["Method"], r["N"], r["Reason"], r["Encode"], r["Cloze"], r["Copy"]] for r in zone_rows]
    (out_dir / "table1_metrics.md").write_text(
        markdown_table(["Method", "N", "Alex ↓", "Aent ↓", "Aprob ↓"], metric_md_rows),
        encoding="utf-8",
    )
    (out_dir / "table2_zones.md").write_text(
        markdown_table(["Method", "N", "Reason ↑", "Encode ↓", "Cloze ↓", "Copy ↓"], zone_md_rows),
        encoding="utf-8",
    )

    paper = PAPER[paper_key]
    comparison_rows = []
    for method in METHOD_ORDER:
        if method not in summary or method not in paper["metrics"]:
            continue
        for metric in ["Alex", "Aent", "Aprob"]:
            observed = summary[method]["metrics"].get(metric, np.nan)
            ref = paper["metrics"][method][metric]
            comparison_rows.append({
                "Method": method,
                "Quantity": metric,
                "Observed": observed,
                "Paper": ref,
                "Delta": observed - ref,
            })
        for zone in ZONE_KEYS:
            observed = summary[method]["zones"].get(zone, np.nan)
            ref = paper["zones"][method][zone]
            comparison_rows.append({
                "Method": method,
                "Quantity": zone,
                "Observed": observed,
                "Paper": ref,
                "Delta": observed - ref,
            })
    write_csv(out_dir / f"comparison_{paper_key}.csv", comparison_rows, ["Method", "Quantity", "Observed", "Paper", "Delta"])
    md_rows = [[r["Method"], r["Quantity"], r["Observed"], r["Paper"], r["Delta"]] for r in comparison_rows]
    (out_dir / f"comparison_{paper_key}.md").write_text(
        markdown_table(["Method", "Quantity", "Observed", f"Paper {paper_key}", "Delta"], md_rows),
        encoding="utf-8",
    )


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


def plot_metrics(args: argparse.Namespace, metrics_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python),
        "-m",
        "anchoring_measure.plot",
        "--input",
        str(metrics_dir),
        "--group-by",
        "method",
        "--x",
        "aent",
        "--y",
        "aprob",
        "--output",
        str(output_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    run(cmd, cwd=ROOT, env=env)


def prepare_inputs(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path]:
    metric_input = run_dir / "inputs" / "example.metric.jsonl"
    controlled_input = run_dir / "inputs" / "figure3_controlled.metric.jsonl"
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
    parser.add_argument("--input", type=Path, default=SCRIPTS_DIR / "examples" / "example.jsonl")
    parser.add_argument("--model", type=Path, default=Path("/home/pengguangyue/workspace/models/Qwen/Qwen3-8B"))
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "anchoring_example_qwen3_8b")
    parser.add_argument("--python", type=Path, default=ROOT / "venv" / "bin" / "python")
    parser.add_argument("--torchrun-nproc", type=int, default=4)
    parser.add_argument("--run-and-hold", type=Path, default=ROOT.parent / "run_and_hold.sh")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--paper-key", choices=sorted(PAPER), default="qwen3_8b")
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
    method_summary = summarize(read_metrics(method_metrics_dir), METHOD_ORDER)
    write_tables(method_summary, run_dir / "tables" / "methods", args.paper_key)
    plot_metrics(args, method_metrics_dir, run_dir / "figures" / "figure4_methods.pdf")

    if not args.skip_controlled_metrics:
        control_args = argparse.Namespace(**vars(args))
        control_args.skip_metrics = False
        run_metrics(control_args, controlled_input, control_metrics_dir, CONTROL_ORDER)
        control_summary = summarize(read_metrics(control_metrics_dir), CONTROL_ORDER)
        write_tables(control_summary, run_dir / "tables" / "controlled", args.paper_key)
        plot_metrics(args, control_metrics_dir, run_dir / "figures" / "figure3_controlled.pdf")

    manifest = {
        "input": str(args.input),
        "model": str(args.model),
        "run_dir": str(run_dir),
        "python": str(args.python),
        "method_metrics": str(method_metrics_dir),
        "controlled_metrics": str(control_metrics_dir),
        "paper_key": args.paper_key,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
