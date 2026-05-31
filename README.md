<h1 align="center">SSR: Measuring and Mitigating Post-hoc Rationalization in Reverse Chain-of-Thought Generation</h1>

---

<p align="center">
| <a href="https://arxiv.org/abs/2602.14469">Paper</a> |
<a href="https://huggingface.co/datasets/Nanbeige/SSR-RCoT-16K">Dataset</a> |
</p>

This is the official implementation of **Measuring and Mitigating Post-hoc
Rationalization in Reverse Chain-of-Thought Generation**, accepted by **ICML
2026**.

We open-source a public 16k subset, **SSR-RCoT-16K**, on Hugging Face.

## Environment Setup

The codebase uses Python 3.11 and `uv`.

```bash
UV_CACHE_DIR=.uv-cache uv venv venv --python 3.11

UV_CACHE_DIR=.uv-cache uv pip install --python venv/bin/python -e . \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match
```

The anchoring analysis scripts require PyTorch, Transformers, Datasets,
Matplotlib, Seaborn, Pandas, NumPy, and tqdm. RCoT generation uses vLLM.

To install the optional generation dependency:

```bash
UV_CACHE_DIR=.uv-cache uv pip install --python venv/bin/python -e ".[generation]" \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match
```

## Model And Data Setup

Place a local model checkpoint before running generation or metric scoring. The
reproduction scripts were tested with Qwen3-8B:

```bash
/path/to/Qwen3-8B
```

For a quick code-path check, use the generated example file:

```bash
scripts/examples/examples.jsonl
```

This file already contains regenerated NEU/SUP/AUG-SUP/SSR RCoT traces, so you
can skip the generation step in the pipeline and start directly from anchoring
metric scoring. You can also pass any metric-format JSONL through `--input`.

For larger-scale experiments, download the public dataset from Hugging Face:

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

The reproduction code is organized under `scripts/`. Core anchoring metrics are
in `scripts/anchoring_measure/`, and RCoT prompt templates are in
`scripts/rcot_generation/`.

1. Optional: generate RCoT traces.

You can skip this step when using `scripts/examples/examples.jsonl`, because it
already contains generated RCoT traces for all four methods. Run this step only
when you want to regenerate RCoT traces from another input file.

```bash
venv/bin/python scripts/generate_rcot_vllm.py \
  --input path/to/input.metric.jsonl \
  --output runs/rcot_example/inputs/generated.metric.jsonl \
  --raw-output runs/rcot_example/generation/raw_outputs.jsonl \
  --model /path/to/Qwen3-8B \
  --methods NEU,SUP,AUG-SUP,SSR \
  --tensor-parallel-size 4
```

2. Run anchoring metric scoring and paper-style aggregation.

```bash
venv/bin/python scripts/reproduce_anchoring.py \
  --input scripts/examples/examples.jsonl \
  --run-dir runs/rcot_example \
  --model /path/to/Qwen3-8B \
  --python venv/bin/python \
  --torchrun-nproc 4 \
  --gpus 0,1,2,3 \
  --paper-key qwen3_8b \
  --force
```

3. Plot calibrated probabilistic anchoring if needed.

```bash
venv/bin/python scripts/plot_aprob_visual_transforms.py \
  --metrics runs/rcot_example/calibrated_aprob/aprob_cal.metrics.jsonl \
  --out-dir runs/rcot_example/calibrated_aprob/visual_transforms \
  --prefix aprob_cal \
  --modes sqrt \
  --zone-threshold-mode quantile \
  --zone-quantile 0.6
```

On the original lab machines, long GPU jobs were launched through
`run_and_hold.sh`:

```bash
/home/pengguangyue/workspace/proj/run_and_hold.sh 0,1,2,3 ...
```

## Result Illustration

The pipeline writes paper-style tables, plots, and reports under the selected
`--run-dir`.

Typical outputs include:

- `tables/methods/table1_metrics.md`
- `tables/methods/table2_zones.md`
- `figures/figure3_controlled.pdf`
- `figures/figure4_methods.pdf`
- `report.md`

For the anchoring-analysis reproduction, `table1_metrics.md` summarizes
`Alex`, `Aent`, and `Aprob`; `table2_zones.md` summarizes the four behavior
zones: `Reason`, `Encode`, `Cloze`, and `Copy`.

The q60 calibrated view used in our reproduction defines probabilistic
anchoring as:

```text
Aprob_cal = clip(I(R; A | Q) / H(A | Q), 0, 1)
```

and displays the y-axis with `sqrt(Aprob_cal)`.

## Citation

```bibtex
@misc{peng2026rcganchoring,
  title={Measuring and Mitigating Post-hoc Rationalization in Reverse Chain-of-Thought Generation},
  author={Guangyue Peng and Zongchao Chen and Wen Luo and Yuntao Wen and Wei Li and Ruixiang Feng and Ran Le and Chen Yang and Zhenwei An and Yang Song and Tao Zhang and Houfeng Wang},
  year={2026},
  note={ICML 2026}
}
```
