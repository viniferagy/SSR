#!/usr/bin/env python3
"""Run mismatch dimensions over method variants with incremental n stages."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGES = [64, 256, 512]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def run(cmd: Sequence[str], cwd: Path, env: Dict[str, str] | None = None, dry_run: bool = False) -> None:
    print("+ " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), env=env, check=True)


def run_parallel(cmds: Sequence[Tuple[Sequence[str], Dict[str, str] | None]], cwd: Path, dry_run: bool = False) -> None:
    for cmd, _cmd_env in cmds:
        print("+ " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    if dry_run:
        return
    procs = [subprocess.Popen([str(x) for x in cmd], cwd=str(cwd), env=cmd_env) for cmd, cmd_env in cmds]
    failed: List[Tuple[Sequence[str], int]] = []
    try:
        for cmd, proc in zip((cmd for cmd, _cmd_env in cmds), procs):
            rc = proc.wait()
            if rc:
                failed.append((cmd, rc))
    finally:
        if failed:
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()
            for proc in procs:
                if proc.poll() is None:
                    proc.wait()
    if failed:
        cmd, rc = failed[0]
        raise subprocess.CalledProcessError(rc, [str(x) for x in cmd])


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_stage_rows(root: Path, manifest: Dict[str, Any], max_stage: int) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    rows_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    methods: List[str] = []
    method_seen: set[str] = set()
    skipped: List[Dict[str, Any]] = []
    for spec in manifest.get("sources", []):
        path = Path(spec["path"])
        if not path.is_absolute():
            path = root / path
        source_rows = read_jsonl(path)
        method_map = spec.get("methods", {})
        for input_method, output_method in method_map.items():
            output_method = str(output_method)
            if output_method not in method_seen:
                methods.append(output_method)
                method_seen.add(output_method)
            kept = 0
            for source_sample_idx, row in enumerate(source_rows):
                if kept >= max_stage:
                    break
                q = ensure_text(row.get("questions", {}).get(input_method))
                a = ensure_text(row.get("answers", {}).get(input_method))
                r = ensure_text(row.get("reasonings", {}).get(input_method))
                if not (q.strip() and a.strip() and r.strip()):
                    skipped.append({"method": output_method, "input_method": input_method, "source_sample_idx": source_sample_idx, "reason": "missing_QAR"})
                    continue
                key = (output_method, kept)
                rows_by_key[key] = {
                    "id": f"{output_method}::{row.get('id', source_sample_idx)}",
                    "source_id": row.get("id", source_sample_idx),
                    "source_sample_idx": source_sample_idx,
                    "variant_sample_idx": kept,
                    "source_path": spec["path"],
                    "family": spec.get("family", ""),
                    "split": spec.get("split", ""),
                    "questions": {output_method: q},
                    "answers": {output_method: a},
                    "contexts": {output_method: ensure_text(row.get("contexts", {}).get(input_method))},
                    "reasonings": {output_method: r},
                }
                kept += 1
    merged: List[Dict[str, Any]] = []
    for sample_idx in range(max_stage):
        out = {"id": f"mismatch-stage-{sample_idx}", "questions": {}, "answers": {}, "contexts": {}, "reasonings": {}, "sources": {}}
        present = False
        for method in methods:
            item = rows_by_key.get((method, sample_idx))
            if item is None:
                continue
            for section in ("questions", "answers", "contexts", "reasonings"):
                out[section][method] = item[section][method]
            out["sources"][method] = {
                "source_id": item["source_id"],
                "source_sample_idx": item["source_sample_idx"],
                "variant_sample_idx": item["variant_sample_idx"],
                "source_path": item["source_path"],
                "family": item["family"],
                "split": item["split"],
            }
            present = True
        if present:
            merged.append(out)
    return merged, methods, skipped


def count_metric_rows(path: Path, glob_pattern: str, start_index: int, end_index: int, methods: Sequence[str]) -> int:
    files = sorted(path.glob(glob_pattern))
    if not files:
        return 0
    wanted = set(methods)
    seen: set[Tuple[int, str]] = set()
    for file in files:
        for row in read_jsonl(file):
            try:
                sample_idx = int(row.get("sample_idx", -1))
            except Exception:
                continue
            method = str(row.get("method", ""))
            if start_index <= sample_idx < end_index and method in wanted:
                seen.add((sample_idx, method))
    return len(seen)


def expected_pairs(input_path: Path, methods: Sequence[str], start_index: int, end_index: int) -> int:
    rows = read_jsonl(input_path)
    total = 0
    for sample_idx, row in enumerate(rows):
        if not (start_index <= sample_idx < end_index):
            continue
        for method in methods:
            if ensure_text(row.get("questions", {}).get(method)).strip() and ensure_text(row.get("answers", {}).get(method)).strip() and ensure_text(row.get("reasonings", {}).get(method)).strip():
                total += 1
    return total


def run_torchrun(python_bin: Path, script: str, args: List[str], nproc: int, cwd: Path, env: Dict[str, str], dry_run: bool) -> None:
    cmd = [str(python_bin), "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", script] + args
    run(cmd, cwd, env=env, dry_run=dry_run)


def summarize_combined_jsonl(output: Path, chunks_dir: Path, pattern: str) -> None:
    files = sorted(chunks_dir.glob(pattern))
    rows: List[Dict[str, Any]] = []
    for file in files:
        rows.extend(read_jsonl(file))
    write_jsonl(output, rows)


def summarize_stage_chunks(output: Path, chunks_root: Path, pattern: str, stage_end: int) -> None:
    rows: List[Dict[str, Any]] = []
    for file in sorted(chunks_root.glob(pattern)):
        parent = file.parent.name
        if not parent.startswith("rows_"):
            continue
        parts = parent.split("_")
        if len(parts) != 3:
            continue
        try:
            chunk_end = int(parts[2])
        except ValueError:
            continue
        if chunk_end <= stage_end:
            rows.extend(read_jsonl(file))
    write_jsonl(output, rows)


def stage_bounds(stages: Sequence[int]) -> List[Tuple[int, int]]:
    out = []
    prev = 0
    for stage in stages:
        out.append((prev, int(stage)))
        prev = int(stage)
    return out


def methods_for_path_diversity(manifest: Dict[str, Any], methods: Sequence[str]) -> List[str]:
    ok = {v["method"] for v in manifest.get("variants", []) if v.get("path_diversity_exact_generator")}
    return [m for m in methods if m in ok]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stages", default=",".join(str(x) for x in DEFAULT_STAGES))
    parser.add_argument("--scoring-model", type=Path, default=Path("/home/pengguangyue/workspace/models/Qwen/Qwen3-8B"))
    parser.add_argument("--embedding-model", type=Path, default=Path("/home/pengguangyue/workspace/models/xlm-roberta-large"))
    parser.add_argument("--python-bin", type=Path, default=ROOT / "venv/bin/python")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--nproc", type=int, default=0)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--path-diversity", action="store_true")
    parser.add_argument("--path-diversity-k", type=int, default=5)
    parser.add_argument("--path-diversity-max-stage", type=int, default=256)
    parser.add_argument("--path-diversity-max-tokens", type=int, default=1024)
    parser.add_argument("--path-diversity-score-python-bin", type=Path, default=None, help="Python binary for embedding scoring; generation still uses --python-bin.")
    parser.add_argument("--path-diversity-enforce-eager", action="store_true", help="Pass --enforce-eager to vLLM path-diversity generation.")
    parser.add_argument("--path-diversity-disable-custom-all-reduce", action="store_true", help="Pass --disable-custom-all-reduce to vLLM path-diversity generation.")
    parser.add_argument("--path-diversity-max-num-seqs", type=int, default=None)
    parser.add_argument("--path-diversity-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--only-path-diversity", action="store_true", help="Run only path-diversity generation/scoring for eligible methods.")
    parser.add_argument("--skip-prefix", action="store_true")
    parser.add_argument("--skip-commitment", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    python_bin = args.python_bin
    path_score_python_bin = args.path_diversity_score_python_bin or python_bin
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    stages = [int(x) for x in args.stages.split(",") if x.strip()]
    max_stage = max(stages)
    rows, methods, skipped = collect_stage_rows(root, manifest, max_stage)
    input_path = run_dir / "inputs" / f"mismatch_variants_{max_stage}.metric.jsonl"
    write_jsonl(input_path, rows)
    (run_dir / "inputs" / "methods.txt").write_text("\n".join(methods) + "\n", encoding="utf-8")
    write_jsonl(run_dir / "inputs" / "skipped_records.jsonl", skipped)
    methods_csv = ",".join(methods)
    env = os.environ.copy()
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
        gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
        gpu_count = len(gpu_ids)
    else:
        gpu_count = int(subprocess.check_output([str(python_bin), "-c", "import torch; print(torch.cuda.device_count())"], cwd=str(root)).decode().strip() or "1")
        gpu_ids = [str(i) for i in range(gpu_count)]
    nproc = args.nproc or max(gpu_count, 1)
    summary = {"input": str(input_path), "methods": methods, "stages": stages, "nproc": nproc, "gpu_count": gpu_count}
    (run_dir / "run_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    prefix_chunks_root = run_dir / "chunks" / "prefix"
    commitment_chunks_root = run_dir / "chunks" / "commitment_confidence"
    path_diversity_global = run_dir / "chunks" / "path_diversity"

    def add_vllm_path_args(cmd: List[str]) -> List[str]:
        if args.path_diversity_enforce_eager:
            cmd.append("--enforce-eager")
        if args.path_diversity_disable_custom_all_reduce:
            cmd.append("--disable-custom-all-reduce")
        if args.path_diversity_max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(args.path_diversity_max_num_seqs)])
        if args.path_diversity_gpu_memory_utilization is not None:
            cmd.extend(["--gpu-memory-utilization", str(args.path_diversity_gpu_memory_utilization)])
        return cmd

    for start, end in stage_bounds(stages):
        stage_dir = run_dir / f"n{end:04d}"
        chunk_name = f"rows_{start:04d}_{end:04d}"
        prefix_chunk = prefix_chunks_root / chunk_name
        prefix_chunk.mkdir(parents=True, exist_ok=True)
        exp = expected_pairs(input_path, methods, start, end)
        got = count_metric_rows(prefix_chunk, "prefix_metrics_rank*.jsonl", start, end, methods)
        if args.only_path_diversity or args.skip_prefix:
            print(json.dumps({"skip": "prefix", "stage": end, "reason": "disabled"}), flush=True)
        elif got < exp:
            run_torchrun(
                python_bin,
                "scripts/anchoring_measure/prefix_curve.py",
                [
                    "score", "--input", str(input_path), "--output-dir", str(prefix_chunk), "--scoring-model", str(args.scoring_model),
                    "--methods", methods_csv, "--start-index", str(start), "--limit", str(end), "--microbatch", str(args.microbatch),
                ],
                nproc,
                root,
                env,
                args.dry_run,
            )
        else:
            print(json.dumps({"skip": "prefix", "stage": end, "chunk": chunk_name, "got": got, "expected": exp}), flush=True)

        if not args.only_path_diversity and not args.skip_prefix:
            prefix_all = stage_dir / "prefix_all"
            summarize_stage_chunks(prefix_all / "prefix_metrics_rank0.jsonl", prefix_chunks_root, "rows_*/prefix_metrics_rank*.jsonl", end)
            run(
                [
                    str(python_bin), "scripts/anchoring_measure/prefix_curve.py", "summarize",
                    "--metrics", str(prefix_all), "--output-dir", str(stage_dir / "prefix_results"), "--methods", methods_csv,
                ],
                root,
                env=env,
                dry_run=args.dry_run,
            )

        commit_chunk = commitment_chunks_root / chunk_name
        commit_chunk.mkdir(parents=True, exist_ok=True)
        got = count_metric_rows(commit_chunk, "commitment_metrics_rank*.jsonl", start, end, methods)
        if args.only_path_diversity or args.skip_commitment:
            print(json.dumps({"skip": "commitment_confidence", "stage": end, "reason": "disabled"}), flush=True)
        elif got < exp:
            run_torchrun(
                python_bin,
                "scripts/anchoring_measure/commitment_kl.py",
                [
                    "score", "--input", str(input_path), "--output-dir", str(commit_chunk), "--scoring-model", str(args.scoring_model),
                    "--methods", methods_csv, "--start-index", str(start), "--limit", str(end),
                ],
                nproc,
                root,
                env,
                args.dry_run,
            )
        else:
            print(json.dumps({"skip": "commitment_confidence", "stage": end, "chunk": chunk_name, "got": got, "expected": exp}), flush=True)

        if not args.only_path_diversity and not args.skip_commitment:
            commit_all = stage_dir / "commitment_all"
            summarize_stage_chunks(commit_all / "commitment_metrics_rank0.jsonl", commitment_chunks_root, "rows_*/commitment_metrics_rank*.jsonl", end)
            run(
                [
                    str(python_bin), "scripts/anchoring_measure/commitment_kl.py", "summarize",
                    "--metrics", str(commit_all), "--output-dir", str(stage_dir / "commitment_confidence_results"), "--methods", methods_csv,
                ],
                root,
                env=env,
                dry_run=args.dry_run,
            )

        if (args.path_diversity or args.only_path_diversity) and end <= args.path_diversity_max_stage:
            pd_methods = methods_for_path_diversity(manifest, methods)
            if pd_methods:
                pd_dir = stage_dir / "path_diversity"
                path_diversity_global.mkdir(parents=True, exist_ok=True)
                raw_output = path_diversity_global / "path_samples.raw.jsonl"
                can_use_vllm = importlib.util.find_spec("vllm") is not None
                if not can_use_vllm:
                    raise RuntimeError(
                        "Path diversity generation requires vLLM. "
                        "Rerun with --python-bin pointing to the vLLM environment."
                    )
                run(
                    add_vllm_path_args([
                        str(python_bin), "scripts/anchoring_measure/path_diversity.py", "generate",
                        "--input", str(input_path), "--output", str(pd_dir / "path_samples.jsonl"),
                        "--raw-output", str(raw_output),
                        "--unsupported-output", str(pd_dir / "unsupported.jsonl"),
                        "--summary-output", str(pd_dir / "path_samples.summary.json"),
                        "--model", str(args.scoring_model), "--methods", ",".join(pd_methods), "--limit", str(end),
                        "--k", str(args.path_diversity_k), "--tensor-parallel-size", str(gpu_count),
                        "--max-tokens", str(args.path_diversity_max_tokens),
                        "--phase", "vllm", "--backend", "vllm",
                    ]),
                    root,
                    env=env,
                    dry_run=args.dry_run,
                )
                special_cmds: List[Tuple[Sequence[str], Dict[str, str] | None]] = []
                for shard_index, gpu_id in enumerate(gpu_ids):
                    shard_env = env.copy()
                    shard_env["CUDA_VISIBLE_DEVICES"] = gpu_id
                    special_cmds.append(
                        (
                            [
                                str(python_bin), "scripts/anchoring_measure/path_diversity.py", "generate",
                                "--input", str(input_path), "--output", str(pd_dir / f"path_samples.special_shard{shard_index}.jsonl"),
                                "--raw-output", str(raw_output),
                                "--unsupported-output", str(pd_dir / f"unsupported.special_shard{shard_index}.jsonl"),
                                "--summary-output", str(pd_dir / f"path_samples.special_shard{shard_index}.summary.json"),
                                "--model", str(args.scoring_model), "--methods", ",".join(pd_methods), "--limit", str(end),
                                "--k", str(args.path_diversity_k), "--max-tokens", str(args.path_diversity_max_tokens),
                                "--phase", "special", "--skip-blind", "--skip-finalize",
                                "--row-shard-count", str(gpu_count), "--row-shard-index", str(shard_index),
                            ],
                            shard_env,
                        )
                    )
                run_parallel(special_cmds, root, dry_run=args.dry_run)
                run(
                    [
                        str(python_bin), "scripts/anchoring_measure/path_diversity.py", "generate",
                        "--input", str(input_path), "--output", str(pd_dir / "path_samples.jsonl"),
                        "--raw-output", str(raw_output),
                        "--unsupported-output", str(pd_dir / "unsupported.jsonl"),
                        "--summary-output", str(pd_dir / "path_samples.summary.json"),
                        "--model", str(args.scoring_model), "--methods", ",".join(pd_methods), "--limit", str(end),
                        "--k", str(args.path_diversity_k), "--max-tokens", str(args.path_diversity_max_tokens),
                        "--phase", "finalize", "--skip-blind",
                    ],
                    root,
                    env=env,
                    dry_run=args.dry_run,
                )
                run(
                    [
                        str(path_score_python_bin), "scripts/anchoring_measure/path_diversity.py", "score",
                        "--input", str(pd_dir / "path_samples.jsonl"), "--output-dir", str(pd_dir / "results"),
                        "--embedding-model", str(args.embedding_model),
                        "--k", str(args.path_diversity_k),
                    ],
                    root,
                    env=env,
                    dry_run=args.dry_run,
                )

    print(json.dumps({"done": True, "run_dir": str(run_dir), "stages": stages}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
