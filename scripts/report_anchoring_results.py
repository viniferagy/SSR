#!/usr/bin/env python3
"""Render a compact Markdown report for anchoring-analysis outputs."""

from __future__ import annotations

import argparse
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/anchoring_example"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir
    output = args.output or (run_dir / "report.md")

    method_dir = run_dir / "figures" / "methods"
    controlled_dir = run_dir / "figures" / "controlled"

    parts = [
        "# Anchoring Analysis Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Table 1: Method Metrics",
        read(method_dir / "methods_table1_metrics.md"),
        "",
        "## Table 2: Behavioral Zones",
        read(method_dir / "methods_table2_zones.md"),
        "",
        "## Figures",
        f"- Methods: `{method_dir / 'methods.pdf'}`",
    ]
    controlled_plot = controlled_dir / "controlled.pdf"
    if controlled_plot.exists():
        parts.append(f"- Controlled reference: `{controlled_plot}`")
    if controlled_dir.exists():
        parts.extend([
            "",
            "## Controlled-Reference Summary",
            "Controlled-reference rows are generated from the selected metric-format input.",
            "",
            read(controlled_dir / "controlled_table1_metrics.md"),
            "",
            read(controlled_dir / "controlled_table2_zones.md"),
        ])
    parts.append("")
    output.write_text("\n".join(parts), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
