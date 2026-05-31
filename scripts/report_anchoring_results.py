#!/usr/bin/env python3
"""Render a compact Markdown report for anchoring reproduction outputs."""

from __future__ import annotations

import argparse
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/anchoring_example_qwen3_8b"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir
    output = args.output or (run_dir / "report.md")

    method_tables = run_dir / "tables" / "methods"
    controlled_tables = run_dir / "tables" / "controlled"

    parts = [
        "# Anchoring Analysis Reproduction Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Table 1: Method Metrics",
        read(method_tables / "table1_metrics.md"),
        "",
        "## Table 2: Behavioral Zones",
        read(method_tables / "table2_zones.md"),
        "",
        "## Comparison With Paper Qwen3-8B",
        read(method_tables / "comparison_qwen3_8b.md"),
        "",
        "## Figures",
        f"- Figure 4 methods: `{run_dir / 'figures' / 'figure4_methods.pdf'}`",
    ]
    fig3 = run_dir / "figures" / "figure3_controlled.pdf"
    if fig3.exists():
        parts.append(f"- Figure 3 controlled reference: `{fig3}`")
    if controlled_tables.exists():
        parts.extend([
            "",
            "## Controlled-Reference Summary",
            "The original paper's controlled-reference data is not shipped in this repository; these rows are generated from `examples.jsonl` by `scripts/prepare_anchoring_input.py controlled-reference`.",
            "",
            read(controlled_tables / "table1_metrics.md"),
            "",
            read(controlled_tables / "table2_zones.md"),
        ])
    parts.append("")
    output.write_text("\n".join(parts), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
