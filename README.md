# jlens-cli

**English** | [简体中文](README.zh-CN.md)

A command-line interface for [Anthropic's Jacobian lens (J-lens)](https://github.com/anthropics/jacobian-lens):
decode internal model activations into tokens, fit lenses, merge results, and export interactive HTML visualizations.

**This is an unofficial wrapper and is not affiliated with Anthropic.** Numerical
computation uses the upstream implementation, pinned to
`581d398613e5602a5af361e1c34d3a92ea82ba8e`; the research algorithm is not reimplemented.
The tool works with local or Hugging Face open-weight models. It does not access
internal activations through the Claude API.

The package also includes `jpplens` / `j-lens++`, a compatible inference CLI for
the published [J++ lens checkpoints](https://huggingface.co/koayon/jpp-lenses).
See [J++ usage and GPU benchmarking](scripts/README.jpp.md).

## Installation

Python 3.12 is recommended; Python 3.10 is the minimum. Git is required to install
the pinned upstream dependency. Run the following from the repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
# Install CPU PyTorch first on CPU systems; for GPUs, follow pytorch.org.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[engine]'
jlens --help
```

`jlens`, `j-lens`, and `python -m jlens_cli` are equivalent. Installing only
`pip install -e .` enables help, version information, and `presets`; computation
commands require the `engine` extra.

## Quick start

These presets come from the upstream notebook. The first run downloads the model
and lens and requires sufficient RAM or VRAM. The 4B/27B models are substantial;
CPU execution is possible but slow. Including a preset does not mean that these
large models have been downloaded or validated in the current environment.

```bash
jlens presets

jlens apply --preset qwen3.5-4b \
  --prompt 'Fact: The currency used in the country shaped like a boot is' \
  --positions=-2 --top-k 5 --format text

jlens render --preset qwen3.5-4b \
  --prompt 'Fact: The currency used in the country shaped like a boot is' \
  --layer-stride 2 --mask-display -o out/slice.html
```

`render` uses the upstream visualization template. During generation, it downloads
and verifies D3 from jsDelivr, then embeds D3 and the data in a single HTML file
that can be opened offline in a browser. `--local-files-only` does not affect this
D3 download. By default, up to 100 highly ranked tokens are tracked. For long
prompts, use `--last-n-tokens 64 --max-tracked 50` to limit output size.

## Custom models and lenses

```bash
jlens apply \
  --model /path/to/hf-model \
  --lens /path/to/lens.pt \
  --prompt-file prompt.txt \
  --layers 0,4,8 --positions=-3,-1 \
  --device cuda --dtype bfloat16 -o out/result.json

cat prompt.txt | jlens apply \
  --model /path/to/hf-model --lens /path/to/lens.pt \
  --prompt-file - --all-positions
```

Lenses can also be downloaded from the Hub using
`--lens org/repo --lens-file folder/lens.pt --lens-revision COMMIT`.
Pin the model version with `--model-revision COMMIT`.
For `apply`, `render`, and `fit`, `--local-files-only` restricts model, tokenizer,
and lens loading to local files or cached resources.

Prompts are encoded as raw text; chat templates are not added automatically.
Positions index the truncated token sequence, including special tokens such as
BOS added by the tokenizer. Only the final token is analyzed by default. Use the
equals sign in `--positions=-2,-1` to avoid argument parsing ambiguity.
Layer indices are zero-based. `apply --layers` accepts nonnegative indices;
`fit --layers` also accepts negative indices. All fitted layers in the lens are
used by default, and the model's final output is reported separately.

JSON goes to stdout and logs go to stderr. Use `--format text` for readable output.
`--logit-lens` selects the baseline without Jacobian transport; a lens is still
required to determine the default layers. Probabilities are computed by softmax
over the full vocabulary, so the displayed top-k probabilities need not sum to 1.

Output schema (`schema_version: 1`):

```text
model, lens, method
input_token_ids                         Full truncated input
positions, input_tokens                 Selected positions and their text
layers["layer_index"][position][rank]    token_id, token, probability
model_output[position][rank]             Final-layer predictions
```

Here, `position` in the nested arrays is the offset within the selected positions,
not the absolute token index.

## Fitting, resuming, and merging

Corpora can be UTF-8 `.txt` files (one prompt per nonempty line) or `.jsonl` files
(one `{"text": "..."}` object per line). By default, upstream fitting excludes
the first 16 tokens and the final position, so each prompt needs at least 18 tokens.
Short prompts are skipped; fitting fails if every prompt is too short.

```bash
jlens fit --model /path/to/hf-model \
  --prompts corpus.jsonl --limit 100 \
  --dim-batch 8 --max-seq-len 128 \
  --checkpoint out/fit.ckpt.pt -o out/lens.pt

# Repeat the same command with --resume; add --force if the output already exists.
jlens fit --model /path/to/hf-model \
  --prompts corpus.jsonl --limit 100 \
  --dim-batch 8 --max-seq-len 128 \
  --checkpoint out/fit.ckpt.pt --resume -o out/lens.pt

jlens inspect --lens out/lens.pt
jlens merge out/shard-a.pt out/shard-b.pt -o out/merged.pt
```

Checkpoints and final lenses have different formats and are not interchangeable.
The CLI writes an additional `.cli.json` metadata file to validate the corpus
hash, model identifier, and key settings when resuming; keep this file alongside
the checkpoint. For reproducible experiments, pin the model revision and do not
modify the local model directory between runs.

Shards must use the same model and layer configuration with disjoint corpora.
Merging weights each lens by its number of successfully processed prompts.
Lens files do not record model identity, so dimension checks cannot detect a
mismatched model with the same dimensions. The CLI saves final lenses in float32
to avoid additional precision loss during merging.

Existing outputs are protected unless `--force` is specified. A new fitting run
also refuses to overwrite an existing checkpoint; choose a new path or use
`--resume`. `--force` applies only to the final output.

## Runtime behavior and limitations

- `--device auto` selects CUDA, MPS, or CPU. Models must fit on one device; multi-GPU sharding is not implemented.
- `--dtype auto` uses BF16 on CUDA devices that support it, and float32 otherwise.
- Eager attention is used to support the backward passes required for Jacobian fitting.
- Custom Hugging Face repository code is disabled by default; enable it explicitly with `--trust-remote-code`.
- Model architecture compatibility follows upstream `jlens.from_hf`; not every Hugging Face model is supported.
- Exit codes: success 0, runtime error 1, argument error 2, interruption 130. Use `jlens --debug ...` for tracebacks.
- Upstream describes the project as an unmaintained reference implementation. PyTorch and Transformers constraints come from upstream; future dependency changes may require validation.

## Development and validation

```bash
pip install -e '.[engine,dev]'
ruff check .
ruff format --check .
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q
python -m build
```

Tests create a tiny Llama with random weights and a tokenizer locally, then run
actual fitting, checkpoint resumption, top-k analysis, weighted merging, and HTML
rendering. No models are downloaded; the rendering test requires network access
to fetch D3. These tests validate the CLI's integration with upstream, not the
interpretability quality of large pretrained models. GitHub Actions runs the
same checks on Python 3.10 and 3.12.

### GPU smoke test

On a machine with CUDA-enabled PyTorch and two visible GPUs, run:

```bash
python scripts/gpu_smoke.py --devices 0,1 --dtype bfloat16 --output out/gpu-smoke
```

The script fits a separate corpus shard on each GPU, checks checkpoint resumption,
applies each lens, renders HTML, and merges the lenses. It records device names,
peak allocated CUDA memory, command timings, and results in `report.json`.
It fails if CUDA is unavailable rather than using CPU. The output directory must
be new. GPUs are exercised sequentially with independent shards; this does not
implement model sharding or distributed training.

Validated on 2026-11-07 with two NVIDIA H100 80GB HBM3 GPUs, BF16, Python 3.11.2,
PyTorch 2.12.0 (CUDA 12.9), and Transformers 5.22.1: all 10 CLI invocations passed,
with CUDA allocations recorded on both GPUs. This checks functionality with a
tiny random model, not pretrained-model performance or interpretability quality.

### Pretrained Qwen resource benchmark

[`scripts/qwen_benchmark.py`](scripts/qwen_benchmark.py) measures revision-pinned
Qwen models on two independently assigned H100 or B200 GPUs. It records model
loading, resident fitting and analysis, GPU utilization, VRAM, process RAM, and CPU.
[`scripts/qwen_report.py`](scripts/qwen_report.py) exports measurements, summaries,
resource recommendations, and a PNG/SVG comparison. See the bilingual
[benchmark guide](scripts/README.qwen.md) for configuration and measurement limits.

## License and sources

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Paper: [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html).
The repository does not bundle model weights, lens files, or training corpora.
Resources downloaded at runtime are subject to their own licenses.
