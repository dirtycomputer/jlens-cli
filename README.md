# J-Lens CLI

An agent-friendly command-line interface for [Anthropic's Jacobian Lens](https://github.com/anthropics/jacobian-lens).
Fit a lens, read intermediate layer/token representations, compare with the vanilla logit lens, and export interactive HTML.

[中文说明](README.zh-CN.md) · [Agent integration](docs/agent-integration.md) · [Validation](docs/validation.md)

**Independent wrapper; not an official Anthropic product.** The numerical estimator and lens application are delegated to the official `jlens` package, pinned to commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`.

## Install

Python 3.10+ and Git are required. Install a PyTorch build appropriate for your machine first; see [PyTorch installation](https://pytorch.org/get-started/locally/). Then:

```bash
git clone https://github.com/dirtycomputer/j-lens-cli.git
cd j-lens-cli
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
j-lens doctor
```

Alternatively, install directly from GitHub:

```bash
python -m pip install git+https://github.com/dirtycomputer/j-lens-cli.git
```

The installed commands `j-lens` and `jlens` are equivalent; `python -m jlens_cli` also works.
Installation includes the pinned upstream source and the ML runtime. `--help`, `--version`, and `schema` do not import the ML runtime. This release is distributed through GitHub, not PyPI.

## Quick start

The following uses a small Hugging Face model as an example. The bundled synthetic corpus is for smoke testing, **not a calibrated scientific lens**. Use a representative corpus and matching model weights for research.

```bash
# 1. Fit (GPU recommended; change --device to cpu for a small CPU experiment).
j-lens fit \
  --model Qwen/Qwen3-0.6B \
  --prompts examples/corpus.jsonl \
  --device cuda --dtype float32 \
  --layers 0,4,8,12 --dim-batch 8 \
  --checkpoint out/fit.ckpt.pt \
  --output out/lens.pt

# 2. Inspect the final input token, including a logit-lens comparison.
j-lens apply \
  --model Qwen/Qwen3-0.6B --lens out/lens.pt \
  --prompt 'Fact: The currency used in the country shaped like a boot is' \
  --positions=-2,-1 --top-k 10 --baseline \
  --output out/readout.json

# 3. Export a self-contained layer × token report.
j-lens visualize \
  --model Qwen/Qwen3-0.6B --lens out/lens.pt \
  --prompt 'The cat sat on the mat.' \
  --output out/report.html
```

Open `out/report.html` locally. Click a cell to inspect its top tokens and probabilities. The report includes its data and JavaScript, requires no server/CDN, and does not send prompts anywhere. It is the wrapper's own compact view, not a copy of the upstream visualization.

For multi-line input, use `--prompt-file prompt.txt`, or `--prompt-file -` to read stdin. Prompts are raw text; no chat template is added automatically.

## Commands

| Command | Purpose |
| --- | --- |
| `fit` | Fit on text, JSON, or JSONL; checkpoint, resume, and shard |
| `apply` | Return top-k tokens, logits, probabilities and optional tracked ranks |
| `visualize` | Write a self-contained interactive HTML report for one prompt |
| `inspect` | Read lens dimensions, matrix norms and local metadata without loading a language model |
| `merge` | Merge local disjoint shards with upstream's prompt-count weighted mean |
| `doctor` | Report installed dependencies and CUDA/MPS availability |
| `schema` | Emit the machine-readable CLI contract |

Run `j-lens <command> --help` for all options. Global `--verbose` and `--debug` go **before** the subcommand.

## Pre-fitted lenses

You can skip fitting if a lens for the **same model checkpoint** is available:

```bash
j-lens apply \
  --model Qwen/Qwen3.5-4B \
  --lens neuronpedia/jacobian-lens \
  --lens-revision qwen-n1000 \
  --lens-filename qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt \
  --prompt 'The currency used in the country shaped like a boot is' \
  --positions=-2 --top-k 10
```

This model/lens mapping is from the pinned upstream walkthrough; the large model download and inference are not part of local validation. Model and lens assets may change independently. Prefer immutable `--revision` and `--lens-revision` values when reproducing a run.

`--lens` accepts a `.pt` file, a directory containing `--lens-filename`, or a Hugging Face repository ID. Model access uses the standard Hugging Face cache/authentication. `--local-files-only` on fit/apply/visualize restricts downloads to cached files; `HF_HUB_OFFLINE=1` also applies to inspect. Custom model code requires the explicit `--trust-remote-code` flag.

## Coordinates and interpretation

- Layers are zero-based **block outputs**, matching upstream `ActivationRecorder`. Fitting always targets the final block and excludes it from source layers by default.
- Positions index the actual tokenized sequence, after BOS handling and truncation. Negative positions count from its end. `--positions=-3,-1`, `--positions 0:20:2`, and `--positions all` are supported. Python slice stops are exclusive.
- `apply` defaults to the last input position; `visualize` defaults to all positions. Neither command generates new text.
- `jacobian` means transported activation readout; `logit` means the untransported baseline; `model` means the actual final-layer readout.
- Probabilities use the **full-vocabulary softmax**, not a top-k renormalization. Top-k probabilities need not sum to 1. `--top-k` is capped at vocabulary size.
- Repeat `--track-token-id 123` to get selected vocabulary tokens' full-vocabulary ranks at every selected layer/position. Ties share a rank; top-k ordering among exact ties is unspecified.
- These outputs are representation readouts, not causal attributions, parameter gradients, or recommendations for weight updates. An RSI agent should test interventions separately.
- Upstream automatic layout detection covers several Hugging Face decoder families. This wrapper loads `AutoModelForCausalLM`; not every multimodal or custom architecture is supported.

## Fit, resume and merge

Fit input formats:

```jsonl
{"id":"example-1","text":"A sufficiently long representative text sequence goes here ..."}
{"id":"example-2","text":"Another sequence goes here ..."}
```

Text files contain one prompt per nonblank line. JSON files contain an array of strings or objects. Use `--text-field prompt` for a different field. For stdin, specify `--input-format jsonl` explicitly; automatic format detection uses the file extension.

The upstream default excludes the first 16 positions and the last position from the fitting estimator. Short sequences may be skipped; the CLI reports the count. If all sequences are too short, it fails without producing a lens. `--max-seq-len` is a truncation limit, not padding. Use `--skip-first 0` only when intentionally changing that estimator (e.g. tiny tests).

Re-run the same fit command to resume its checkpoint. If a finished lens already exists, add `--force`. A sidecar fingerprint checks prompt contents, model identifiers/resolved model revision, tokenizer settings and estimator parameters. Missing or mismatched checkpoint metadata is rejected. To restart and replace a checkpoint, use **both** `--no-resume --force`.

```bash
# Run separately, potentially on different machines/devices.
j-lens fit --model /models/my-model --prompts corpus.jsonl \
  --num-shards 2 --shard-index 0 --output out/shard-0.pt
j-lens fit --model /models/my-model --prompts corpus.jsonl \
  --num-shards 2 --shard-index 1 --output out/shard-1.pt

j-lens merge out/shard-0.pt out/shard-1.pt --output out/merged.pt
```

Sharding uses `records[shard_index::num_shards]`, after `--limit`. Keep shards disjoint. Duplicate file paths and identical prompt-subset hashes are rejected; arbitrary partial overlap cannot be detected. Files without wrapper metadata can be merged when shapes agree, but their provenance cannot be verified. Preserve each `.pt.meta.json` beside its lens.

The CLI checks widths/layer bounds and known local metadata. Equal dimensions do **not** prove two models have matching weights. Metadata fingerprints identify inputs/configuration and are not full checksums of model weights. `--allow-model-mismatch` permits an intentional model-ID/revision alias override; it does not make a foreign lens valid.

Upstream fitting retains a backward graph: reduce `--dim-batch`, `--max-seq-len` or selected layers to manage memory. The number of backward passes per prompt is approximately `ceil(hidden_size / dim_batch)`. Final lenses use FP16 storage by default; choose `--save-dtype float32` when preserving full estimator precision matters. Checkpoints store running sums and are **not** final lens files.

For multi-device loading, install `python -m pip install '.[sharded]'` and pass `--device-map auto`. Single-device eager attention is the conservative default; alternative attention kernels, MPS and sharded fitting require validation on your hardware.

## Machine-readable output

Successful stdout is a JSON envelope:

```json
{"schema_version":"1.0","status":"ok","command":"apply","data":{"results":[]}}
```

The empty results above illustrate the envelope only. Actual results include prompt IDs, input token IDs, positions, per-layer readouts and run metadata. With `apply --output-format jsonl`, each line contains one envelope with a single `data.result`. A batch loads the model once but processes prompts sequentially; JSONL output is emitted after the batch succeeds, not streamed token by token.

`apply --output FILE` writes the full result atomically and prints a small output-file summary. Logs go to stderr. On failure, the **last stderr line** is a JSON error with `error.code` and `error.message`; earlier lines may contain library diagnostics. Exit codes: `0` success, `1` runtime/dependency failure, `2` invalid input/usage, `130` interruption. `doctor` succeeds when it produced its report; agents must check `data.runtime_ready`.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q
python -m build
```

Integration tests create a local, randomly initialized GPT-2 model and tokenizer. They exercise actual upstream fitting/backpropagation, apply, resume, merge, batch I/O and HTML export without model downloads. See [validation details](docs/validation.md).

## Attribution and license

Wrapper: Apache-2.0, copyright 2026 Xinyuan Wang. Upstream Jacobian Lens: copyright 2026 Anthropic PBC, Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Method and reference implementation: [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html). No model weights or research corpora are bundled; downloaded models and data retain their respective licenses.
