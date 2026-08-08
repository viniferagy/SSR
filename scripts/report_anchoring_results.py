#!/usr/bin/env python3
"""Render a compact report from the final anchoring pipeline outputs."""

from __future__ import annotations

import argparse
from pathlib import Path


def optional_read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else "Not generated."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir / "report.md"
    methods = args.run_dir / "results" / "methods" / "anchoring_metrics.md"
    controls = args.run_dir / "results" / "controlled" / "anchoring_metrics.md"
    figures = args.run_dir / "figures"

    lines = [
        "# SSR Anchoring Analysis",
        "",
        "## Main Methods",
        "",
        optional_read(methods),
    ]
    if controls.exists():
        lines.extend(["", "## Controlled References", "", optional_read(controls)])
    method_figure = figures / "behavioral_zones_methods.pdf"
    control_figure = figures / "behavioral_zones_controlled.pdf"
    if method_figure.exists() or control_figure.exists():
        lines.extend(["", "## Behavioral Zones", ""])
        if method_figure.exists():
            lines.append(f"- Main methods: `{method_figure}`")
        if control_figure.exists():
            lines.append(f"- Controlled references: `{control_figure}`")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
