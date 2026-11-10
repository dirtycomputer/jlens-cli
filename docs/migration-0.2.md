# Consolidation into jlens-cli 0.2

`dirtycomputer/jlens-cli` is the maintained repository for both J-lens and the
compatible J++ reader. The agent-oriented functionality from `j-lens-cli` at
`de94a22b519742236aa90e3bfa7162228a0cbc7c` is incorporated here. The consolidation
commit includes that repository as a merge parent, preserving its original
commits, documentation and attribution even if the old GitHub repository is
deleted. Use a **merge commit**, rather than squash/rebase, when merging this PR
to retain those original commit identities on main.

## Included capabilities

| Capability | Entry points |
| --- | --- |
| J-lens fitting, safe resume, shard selection, provenance-aware merge | `jlens fit`, `jlens merge` |
| J-lens and J++ batch readouts, JSON/JSONL, token ranks | `jlens apply`, `jpplens apply` |
| Versioned success/error envelopes, runtime checks | Both CLIs: `schema`, `doctor` |
| Offline interactive HTML without a CDN download | Both CLIs: `visualize` |
| Existing model/lens presets | Both CLIs: `presets`, `--preset` |
| Published J++ checkpoint download, identity validation and filtering | `jpplens download`, `inspect`, `apply` |
| Original upstream HTML, 0.1 JSON and checkpoint behavior | `jlens render`, both CLIs: `--legacy ...` |
| Existing GPU benchmarks and reports | `scripts/`, using the retained legacy API |

J++ fitting, activation intervention, parameter saliency and automatic RSI
training are not implemented by this migration.

## Installation

Both former distributions used the `jlens_cli` Python package and `jlens` /
`j-lens` console scripts. Use one distribution per virtual environment. To
replace an environment where both were installed, uninstall both distributions,
then install this repository again:

```bash
python -m pip uninstall j-lens-cli jlens-cli
python -m pip install -e '.[engine]'
jlens --version
jlens doctor
jpplens schema
```

Run these commands from the maintained `jlens-cli` checkout. A fresh virtual
environment is also supported. Install the appropriate PyTorch CPU/CUDA build
first; add `sharded` alongside `engine` to use `--device-map auto`.

## Output and argument compatibility

The default 0.2 JSON response is:

```json
{"schema_version":"1.0","status":"ok","command":"apply","data":{"results":[]}}
```

Single-prompt results are in `data.results[0]`. JSONL output uses one envelope
per sample, with `data.result`. `--output FILE` writes the complete response
atomically and prints a summary to stdout. Logs and structured errors go to
stderr. All samples must succeed before batch output is published.

The old flags `--model-revision`, `--lens-file`, `--all-positions`, `--logit-lens`
and `--format text` are accepted by the new readout CLI. Their equivalents are
`--revision`, `--lens-filename`, `--positions all`, the baseline-only readout,
and plain text respectively. `--baseline` includes both transported and
untransported readouts. J++ filtering applies only to transported readout top-k;
probabilities and tracked `rank` use the entire vocabulary. `filtered_rank` is
reported separately, and is null for excluded tokens or unfiltered readouts.

For an unchanged 0.1 script, prefix its existing command with `--legacy`:

```bash
jlens --legacy apply --model /models/model --lens /lenses/lens.pt --prompt 'hello'
jpplens --legacy apply --preset qwen3.5-9b --prompt 'hello'
```

`--legacy` must be the first argument. It preserves the old raw JSON, exit codes,
single-device loading and fitting behavior. The historical `jlens render`
command remains available, including its D3 download; prefer `visualize` for
fully offline generation. Historical command documentation is retained in
[English](legacy-cli.md) and [Chinese](legacy-cli.zh-CN.md).

## Lens/checkpoint files

Existing final J-lens `.pt` files and published merged J++ checkpoints remain
readable. New J-lens fits save `.pt.meta.json` provenance sidecars. Old final
lenses lacking sidecars are structurally checked, but their model identity
cannot be verified from dimensions alone.

The 0.1 `.cli.json` resumable checkpoint format remains usable through
`jlens --legacy fit ... --resume`. New fitting jobs use the stronger
`.meta.json` fingerprint and reject unverified old checkpoints. Fit defaults to
resume when a matching checkpoint exists; use `--no-resume --force` for an
intentional restart. New final lenses default to FP16 storage; specify
`--save-dtype float32` when full estimator precision is required. Legacy fitting
retains its FP32 storage behavior.

Preserve metadata sidecars with their lens/checkpoint. Metadata fingerprints
cover configuration and inputs, not full model-weight checksums.

## 删除旧仓库前

迁移覆盖原仓库的代码、测试、文档、示例和提交历史。确认迁移 PR 以 merge commit
合入 `dirtycomputer/jlens-cli` 的 main、CI 通过后，即可删除 `j-lens-cli`。
安装地址统一使用 `https://github.com/dirtycomputer/jlens-cli`。
