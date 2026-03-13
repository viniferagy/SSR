import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

METRIC_MAP = {
    "aent": "entropy_anchoring",
    "aprob": "ProbabilisticAnchoring",   # 修复：与计算代码一致
    "alex": "lexical_anchoring"
}

def main():
    parser = argparse.ArgumentParser(description="Plot Anchoring Quadrants")
    parser.add_argument("--input", type=str, required=True, help="Input metrics.jsonl file or directory")
    parser.add_argument("--group-by", type=str, default="method", help="Field to group by")
    parser.add_argument("--x", type=str, default="aent", help="X-axis metric abbreviation")
    parser.add_argument("--y", type=str, default="aprob", help="Y-axis metric abbreviation")
    parser.add_argument("--output", type=str, required=True, help="Output PNG/PDF path")
    args = parser.parse_args()

    x_key = METRIC_MAP.get(args.x, args.x)
    y_key = METRIC_MAP.get(args.y, args.y)
    c_key = METRIC_MAP.get("alex")

    input_files = [args.input] if os.path.isfile(args.input) else [
        os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.jsonl')
    ]

    data = defaultdict(lambda: defaultdict(list))
    for filepath in input_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                g = record.get(args.group_by, "unknown")
                if x_key in record and y_key in record:
                    data[g][x_key].append(record[x_key])
                    data[g][y_key].append(record[y_key])
                    data[g][c_key].append(record.get(c_key, 0.5))

    methods = list(data.keys())
    ncols = len(methods)
    if ncols == 0:
        print("No valid data found to plot.")
        print(f"  Expected x key : '{x_key}'")
        print(f"  Expected y key : '{y_key}'")
        print("  Check that METRIC_MAP values match the keys in your JSONL.")
        return

    sns.reset_orig()
    fig, axes = plt.subplots(1, ncols, figsize=(4.3 * ncols, 4.3))
    if ncols == 1: axes = [axes]
    cmap, norm = plt.cm.coolwarm, plt.Normalize(0.0, 1.0)

    for i, m_key in enumerate(methods):
        ax = axes[i]
        x = np.array(data[m_key][x_key])
        y = np.array(data[m_key][y_key])
        c = np.array(data[m_key][c_key])

        if len(x) > 500:
            np.random.seed(42)
            idx = np.random.choice(len(x), 500, replace=False)
            xf, yf, cf = x[idx], y[idx], c[idx]
        else:
            xf, yf, cf = x, y, c

        ax.scatter(xf, yf, c=cf, cmap=cmap, norm=norm, s=60, alpha=0.7, rasterized=True)
        ax.set_title(m_key, fontsize=22, fontweight='bold', pad=15)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.tick_params(axis='both', labelsize=18)
        for s in ax.spines.values(): s.set_visible(False)

        if args.x == "aent" and args.y == "aprob":
            bg_alpha = 0.02
            colors = {'Reason': 'green', 'Encode': 'orange', 'Cloze': 'blue', 'Copy': 'red'}
            ax.fill_between([0, 0.5], 0, 0.5, color=colors['Reason'], alpha=bg_alpha)
            ax.fill_between([0, 0.5], 0.5, 1, color=colors['Encode'], alpha=bg_alpha)
            ax.fill_between([0.5, 1], 0, 0.5, color=colors['Cloze'], alpha=bg_alpha)
            ax.fill_between([0.5, 1], 0.5, 1, color=colors['Copy'], alpha=bg_alpha)

            lw, ls, pad = 3.5, '--', 0.08
            ax.plot([0, 0, 0.5], [0.5, 0, 0], color=colors['Reason'], ls=ls, lw=lw, clip_on=False)
            ax.text(pad, pad, "Reason", color=colors['Reason'], ha='left', va='bottom', fontsize=20, fontweight='bold')
            ax.plot([0, 0, 0.5], [0.5, 1, 1], color=colors['Encode'], ls=ls, lw=lw, clip_on=False)
            ax.text(pad, 1-pad, "Encode", color=colors['Encode'], ha='left', va='top', fontsize=20, fontweight='bold')
            ax.plot([0.5, 1, 1], [0, 0, 0.5], color=colors['Cloze'], ls=ls, lw=lw, clip_on=False)
            ax.text(1-pad, pad, "Cloze", color=colors['Cloze'], ha='right', va='bottom', fontsize=20, fontweight='bold')
            ax.plot([0.5, 1, 1], [1, 1, 0.5], color=colors['Copy'], ls=ls, lw=lw, clip_on=False)
            ax.text(1-pad, 1-pad, "Copy", color=colors['Copy'], ha='right', va='top', fontsize=20, fontweight='bold')

        if i == 0:
            ax.set_ylabel("Probabilistic Anchoring" if args.y == "aprob" else args.y, fontsize=22, fontweight='bold')

    fig.text(0.5, -0.05, "Entropic Anchoring" if args.x == "aent" else args.x, ha='center', fontsize=24, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
