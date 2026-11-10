# Validation scope

The migrated agent integration suite uses a **real**, randomly initialized Hugging Face GPT-2 decoder with three layers, hidden width 8, two attention heads, and a local 24-token tokenizer. It runs on CPU with eager attention and FP32 weights. The model and tokenizer are generated locally; no pretrained model download is required.

The integration suite exercises the pinned official `jlens` implementation, including forward hooks, activation gradients, Jacobian accumulation and serialization. It checks:

- Actual fit and reload, and exact CLI readout token/probability agreement with a direct upstream call.
- Logit-lens baseline and final-model readout in the same result.
- Unchanged saved model weights after fitting.
- Checkpoint resume equality; rejection of changed fitting inputs/settings.
- Disjoint shard merging against a single full-corpus fit.
- JSON/JSONL batch output, stdin input, atomic result files and overwrite protection.
- HTML export, prompt/title escaping and lack of external scripts.
- Invalid positions/layers, mismatched widths, non-finite lens matrices and all-short corpora.

Unit tests cover index parsing, corpus validation, dependency-free CLI discovery and the structured error contract. The retained legacy suite additionally covers a real two-layer Llama and numerical J++ readouts. New J++ tests compare the unified batch output against direct upstream logits and the retained reader, including full-vocabulary and filtered tracked ranks, one model load per batch, all-or-nothing output, local download/inspect and offline HTML. The GitHub Actions workflow is configured to run the suite on Python 3.10 and 3.12 and build the wheel/sdist.

This validates the wrapper's behavior and the small-model integration. It does **not** validate scientific lens quality, large-model throughput, GPU/MPS execution, mixed precision or sharded model loading. The documented large-model/pre-fitted lens examples require the corresponding external assets and hardware.

## Historical source-repository run: 2026-11-07

Environment: Python 3.12.14, PyTorch 2.15.0+cpu, Transformers 5.22.2, upstream `jlens` 0.1.0 at the pinned commit.

```text
ruff check: All checks passed!
ruff format --check: passed
pytest -q: 28 passed in 58.32s
python -m build: wheel and source distribution built successfully
installed wheel: j-lens --version, jlens schema and j-lens doctor passed
HTML DOM smoke test: 15 clickable cells; selected cell displays 10 vocabulary entries
```

Source distribution contents (including test fixtures and documentation) and wheel contents were checked. The HTML JavaScript was executed in jsdom; a full browser layout check was not completed because the available Chromium download was invalid. No CUDA device was used. Hosted GitHub Actions results are separate from this local run.

## Consolidation validation: 2026-11-10

Python 3.12.14, PyTorch 2.15.0+cpu, Transformers 5.22.2, pinned upstream commit.
The complete suite passed 45 tests; a subsequently added CLI compatibility test
also passed (25 agent CLI tests total). Ruff lint/format and dependency checks
passed. Wheel and sdist built successfully with scripts, examples, tests and
migration documentation included. The installed wheel was checked in a fresh
environment without engine dependencies: all four entry points, both schemas,
both doctor reports and both legacy preset entry points passed.

No GPU execution was performed for this consolidation. Existing GPU scripts and
historical results are retained separately. Hosted CI runs the complete suite
on Python 3.10 and 3.12 for the final commit.
