# Anchoring Analysis Reproduction

This note documents the code path used to reproduce the first-stage anchoring
analysis. The GitHub-facing repository keeps runnable reproduction code under
`scripts/`; local experiment leftovers are kept under `experimental/` and are
not meant to be uploaded.

## Project Layout

- `scripts/anchoring_measure/`: anchoring metric implementation, aggregation,
  plotting, and the paper-formula metric variant.
- `scripts/rcot_generation/`: prompt templates for NEU, SUP, AUG-SUP, and SSR
  reverse-CoT generation.
- `scripts/generate_rcot_vllm.py`: vLLM generation wrapper for four RCoT
  variants.
- `scripts/prepare_anchoring_input.py`: validates metric-format JSONL, builds
  the Figure 3 controlled-reference approximation, and converts public
  SSR-RCoT-16K rows to SSR-only metric input.
- `scripts/reproduce_anchoring.py`: prepares inputs, launches metric scoring,
  summarizes Table 1/2-style numbers, plots Figure 3/4-style PDFs, and compares
  against paper reference values.
- `scripts/report_anchoring_results.py`: renders a compact Markdown report from
  generated tables.

## Missing or Ambiguous Information

- The full first-stage analysis dataset is not shipped in this repository.
- `scripts/examples/example.jsonl` is a local smoke-test example path and is
  ignored by Git. Place an example JSONL there or pass another path with
  `--input`.
- The exact controlled-reference source data for Figure 3 is not shipped. The
  reproduction constructs an approximation from available metric-format rows.
- The public `Nanbeige/SSR-RCoT-16K` dataset provides SSR-format traces, but not
  the NEU/SUP/AUG-SUP comparison traces. Full Table 1/2 reproduction on that
  dataset requires generating those method-specific traces first.

## Environment

```bash
UV_CACHE_DIR=.uv-cache uv venv venv --python 3.11

UV_CACHE_DIR=.uv-cache uv pip install --python venv/bin/python -e . \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match
```

Verified core versions in the reproduction environment:

- Python 3.11.15
- torch 2.6.0+cu124
- transformers 4.57.6
- numpy 2.4.6
- seaborn 0.13.2
- datasets 4.8.5

## Important Fixes

The original metric runner silently swallowed all per-sample exceptions. With
current Transformers, the shared-prefix cache path failed for Qwen3 because
`DynamicCache` changed its internal layout. The current code:

- exposes `--debug-errors`;
- exposes `--disable-shared-prefix-past`;
- supports current `DynamicCache.batch_repeat_interleave`;
- supports entropy computation when shared-prefix cache is disabled.

## Example Reproduction

Run Table 1/2 style metrics and Figure 4 on the local example data:

```bash
venv/bin/python scripts/reproduce_anchoring.py \
  --input scripts/examples/example.jsonl \
  --run-dir runs/anchoring_example_qwen3_8b \
  --model /path/to/Qwen3-8B \
  --python venv/bin/python \
  --torchrun-nproc 4 \
  --gpus 0,1,2,3 \
  --paper-key qwen3_8b \
  --skip-controlled-metrics \
  --force
```

Main outputs:

- `runs/anchoring_example_qwen3_8b/tables/methods/table1_metrics.md`
- `runs/anchoring_example_qwen3_8b/tables/methods/table2_zones.md`
- `runs/anchoring_example_qwen3_8b/tables/methods/comparison_qwen3_8b.md`
- `runs/anchoring_example_qwen3_8b/figures/figure4_methods.pdf`

## Figure 3

The paper's controlled-reference source data is not present in this repository.
`scripts/prepare_anchoring_input.py controlled-reference` constructs an
approximation from metric-format rows:

- `Real CoT`: approximated by the chosen base method, default `NEU`;
- `+Prob Anchor`: base reasoning with answer appended;
- `+Entropy Anchor`: function-word skeleton of the answer;
- `Response as CoT`: answer used as reasoning.

Run controlled-reference metrics and plot:

```bash
venv/bin/python scripts/reproduce_anchoring.py \
  --input scripts/examples/example.jsonl \
  --run-dir runs/anchoring_example_qwen3_8b \
  --model /path/to/Qwen3-8B \
  --python venv/bin/python \
  --torchrun-nproc 4 \
  --gpus 0,1,2,3 \
  --paper-key qwen3_8b \
  --skip-metrics
```

This produces `runs/anchoring_example_qwen3_8b/figures/figure3_controlled.pdf`.

## SSR-RCoT-16K

Download or load `Nanbeige/SSR-RCoT-16K` with the uv environment:

```bash
venv/bin/python - <<'PY'
from datasets import load_dataset
ds = load_dataset("Nanbeige/SSR-RCoT-16K")
print(ds)
PY
```

Rows can be converted to metric input with:

```bash
venv/bin/python scripts/prepare_anchoring_input.py convert-ssr-rcot \
  --input path/to/ssr_rcot_16k.jsonl \
  --output runs/ssr_rcot_16k/inputs/ssr.metric.jsonl \
  --method SSR
```

To reproduce full Table 1/2 on the public set, generate NEU/SUP/AUG-SUP/SSR
method-specific RCoT traces with `scripts/generate_rcot_vllm.py`.
