<h1 align="center">SSR: Measuring and Mitigating Post-hoc Rationalization in Reverse Chain-of-Thought Generation</h1>

---

<p align="center">
<a href="https://arxiv.org/abs/2602.14469">📄 Paper</a> |
<a href="https://huggingface.co/datasets/Nanbeige/SSR-RCoT-16K">🤗 Dataset</a>
</p>

This is the official implementation of **[Measuring and Mitigating Post-hoc
Rationalization in Reverse Chain-of-Thought Generation](https://openreview.net/forum?id=cY0pi9WHvL)**,
**ICML 2026**.

## Environment Setup

```bash
UV_CACHE_DIR=.uv-cache uv venv venv --python 3.11

UV_CACHE_DIR=.uv-cache uv pip install --python venv/bin/python -e ".[generation]" \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match
```

## Model And Data Setup

Place a local model checkpoint before running generation or metric scoring. The
pipeline was tested with Qwen3-8B. For a quick code-path check, use the
generated example file:

```bash
scripts/examples/examples.jsonl
```

This file already contains regenerated NEU/SUP/AUG-SUP/SSR RCoT traces, so you
can skip the generation step in the pipeline and start directly from anchoring
metric scoring. You can also pass any metric-format JSONL through `--input`.

For larger-scale experiments, download the open-sourced public 16k subset, **SSR-RCoT-16K**, from Hugging Face:

```bash
venv/bin/python - <<'PY'
from datasets import load_dataset
ds = load_dataset("Nanbeige/SSR-RCoT-16K")
print(ds)
PY
```

If you export the dataset to JSONL, convert SSR-format rows to the anchoring
metric input format with:

```bash
venv/bin/python scripts/prepare_anchoring_input.py convert-ssr-rcot \
  --input path/to/ssr_rcot_16k.jsonl \
  --output runs/ssr_rcot_16k/inputs/ssr.metric.jsonl \
  --method SSR
```

## Experiment Pipeline

The code is organized under `scripts/`. Core anchoring metrics are in
`scripts/anchoring_measure/`, and RCoT prompt templates are in
`scripts/rcot_generation/`.

1. Optional: generate RCoT traces.

You can skip this step when using `scripts/examples/examples.jsonl`, because it
already contains generated RCoT traces for all four methods. Run this step only
when you want to regenerate RCoT traces from another input file.

```bash
venv/bin/python scripts/generate_vllm.py \
  --mode rcot \
  --input path/to/input.metric.jsonl \
  --output runs/rcot_example/inputs/generated.metric.jsonl \
  --raw-output runs/rcot_example/generation/raw_outputs.jsonl \
  --model /path/to/Qwen3-8B \
  --methods NEU,SUP,AUG-SUP,SSR \
  --tensor-parallel-size 4
```

2. Run anchoring metric scoring, tables, and plots.

```bash
venv/bin/python scripts/run_anchoring_pipeline.py \
  --input scripts/examples/examples.jsonl \
  --run-dir runs/rcot_example \
  --model /path/to/Qwen3-8B \
  --python venv/bin/python \
  --torchrun-nproc 4 \
  --force
```

3. Build a report.

```bash
venv/bin/python scripts/report_anchoring_results.py \
  --run-dir runs/rcot_example
```

## Result Illustration

The pipeline writes tables, plots, and reports under the selected `--run-dir`.

Typical outputs include:

- `figures/methods/methods_table1_metrics.md`
- `figures/methods/methods_table2_zones.md`
- `figures/methods/methods.pdf`
- `figures/controlled/controlled_table1_metrics.md`
- `figures/controlled/controlled_table2_zones.md`
- `figures/controlled/controlled.pdf`
- `report.md`

The metric table summarizes `Alex`, `Aent`, and `Aprob`. The zone table
summarizes `Reason`, `Encode`, `Cloze`, and `Copy`.

## Citation

```bibtex
@inproceedings{peng2026rcganchoring,
  title={Measuring and Mitigating Post-Hoc Rationalization in Reverse Chain-of-Thought Generation},
  author={Guangyue Peng and Zongchao Chen and Wen Luo and Yuntao Wen and Wei Li and Ruixiang Feng and Ran Le and Chen Yang and Zhenwei An and Yang Song and Tao Zhang and Houfeng Wang},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=cY0pi9WHvL}
}
```
