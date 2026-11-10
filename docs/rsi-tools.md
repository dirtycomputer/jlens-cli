# RSI tool interface (0.3)

The recommended boundary is **observations → bounded trial candidates → an
external trainer and independent gates**. J-lens/J++ supply evidence; pilot
training and held-out behavior determine which update is accepted. This package
does not train, select a winner, update a deployed model, or learn an outer-loop
policy. Component projections remain experimental associations.

## Three layers

```text
RSI controller / experiment ledger
  ├─ harness, memory and data updates
  ├─ choose samples, concepts, positions, budgets and trial arms
  └─ external SFT / GRPO / CRT trainer and behavioral gates
                 ↕ jlens-tools/1
LensTool (Python) / jlens tool (one request) / jlens serve (JSONL)
                 ↕
LensSession: one resident immutable checkpoint + tokenizer + J-lens or J++
```

`jlens` / `j-lens` default to J-lens. `jpplens` / `j-lens++` default to J++.
The session config can explicitly select `backend`. Both use the same methods,
request schema and artifact format. `apply`, `fit`, `merge`, `visualize`, presets,
the historical benchmark scripts and `--legacy` remain available.

## Start a session

Copy [the config example](../examples/rsi-session.json), set local/Hugging Face
model and lens paths, then run:

```bash
jlens serve --config examples/rsi-session.json
# Or send one JSON file to the identical protocol:
jlens tool --config examples/rsi-session.json --request examples/rsi-readout.json
jlens schema
```

`serve` reads one request per stdin line and writes one reply per stdout line,
flushing after each reply. Logs go to stderr. It is a local sequential worker,
not an HTTP service or concurrent model server. The first model-backed method
loads the model, lens and content identities; later requests reuse them. Use
`close` or EOF to end the worker. Restart with a new immutable checkpoint after
training. Do not wrap a live training actor with `jlens.from_hf`.

```json
{"id":"read-1","method":"readout","params":{"samples":[{"id":"case-1","input_ids":[101,202,303],"positions":[-1]}],"layers":"4:20:4","concepts":{"need":["honest"," honest"],"wrong":["trick"]},"baseline":true}}
```

The token IDs/layers above illustrate the contract; use IDs from the actual
rollout and layers present in your lens. Subsequent model-backed requests can
include `manifest_id` at the request root; a mismatch produces `stale_manifest`.
Use [the Python process adapter](../examples/resident_tool.py) to keep a worker
open, or import `LensTool` from `jlens_cli.tools` directly.

```python
from jlens_cli.tools import LensTool

tool = LensTool(config)
reply = tool.handle({"id": 1, "method": "capabilities"})
```

`capabilities` returns method descriptions, JSON input schemas, actual configured
limits and explicitly unsupported operations, without loading model weights.
Malformed requests fail individually; the worker continues. A request produces
one complete reply. It does not publish partially successful sample results.

## Methods and artifacts

| Method | Result / use |
| --- | --- |
| `input.prepare` | Exact IDs, template handling, original/truncated lengths, selected source positions and the next-token positions those logits predict |
| `concepts.resolve` | Single-token alias IDs; multi-token aliases remain explicitly unsupported/partial |
| `readout` | Transported, optional logit baseline and final-model readouts from **one forward per sample**; full-vocabulary tracked ranks, separate J++ filtered ranks |
| `concepts.trajectory` | Descriptive first appearance/dropout curves at sampled layers; missing/partial coverage gives `unknown` |
| `components.list` | Actual supported residual-output module paths, layers, dimensions and LoRA parameter counts per rank |
| `components.score` | Failure-set projection minus background-set projection, sample counts and an unpaired standard-error estimate |
| `checkpoint.compare` | Paired fixed-token replay; rejects changed inputs/tokenizer/specification and implicit lens-coordinate changes |
| `audit.scan` | Concept-family occurrence rates and evidence sample IDs; no automatic release verdict |
| `lens.health` | Cheap labeled-canary coverage, or comparison with an independently fitted content-bound reference; insufficient evidence gives `unknown` |
| `trials.propose` | Budgeted PEFT configs for contrastive, seeded random and late-MLP arms, plus supervision scope; every candidate is unvalidated |
| `close` | End the resident session |

All artifact-only methods can run without a session config or ML dependencies.
Pass a saved readout as a JSON object in `observation`, `before`/`after`, or
`reference`/`candidate`; the complete `readout` reply is also accepted. A caller
can therefore reuse one numerical observation for trajectory, audit, comparison
and health analysis. Artifact requests do not read arbitrary filesystem paths.

Every readout carries `manifest_id`, `artifact_id`, a model/tokenizer/lens
manifest, exact token inputs, position mappings, analysis settings and metrics.
Local model/config/tokenizer files are content hashed once per session; remote
models use the resolved Hub commit. Tokenizer/template content and the actual
lens file are fingerprinted. An unavailable immutable identity is marked
`unverified`. `artifact_id` covers semantic content, excluding volatile timing
metrics. Hashes detect accidental changes, not malicious provenance forgery;
store the original artifacts in the caller's immutable ledger.

## Exact input and positions

Each sample contains exactly one of `input_ids`, `text`, or `messages`. Prefer
rollout `input_ids`: no decoding, re-tokenization, extra BOS or template is
inserted. Optional `attention_mask` and `position_ids` are passed through.
`messages` explicitly applies the configured tokenizer's chat template once;
`text` is raw text. Tokenizer/template choice is part of the manifest.

Source position `p` describes the residual after consuming input token `p`;
the final-model logits predict token `p+1`. To diagnose an incorrect token,
observe the preceding position. `original_positions` and
`predicts_original_positions` preserve source coordinates after explicit
truncation. A truncated window is a different input; it does not preserve the
omitted context's computation or automatically preserve rotary positions.
Supply original `position_ids` when appropriate for the model.

Over-budget inputs fail with `input_truncated`. Set `truncation` to `left` or
`right` deliberately to accept a window; the result reports lengths and offset.
This fail-fast default also applies to the modern `apply`/`visualize` commands
(`--truncation left|right`). Legacy commands retain their old truncation behavior.

## Components and candidate updates

The implemented score is:

```text
direction_l = unit((mean(U[need]) - beta * mean(U[wrong])) @ J_l)
score_m = mean_failure(dot(component_write_m, direction_l))
          - background_weight * mean_reference(dot(component_write_m, direction_l))
```

Only actual residual-output projections with recognized module names and widths
are supported (`o_proj`, `down_proj`, and GPT-2 `c_proj`). The method scores whole
modules, including their output biases. It does not interpret Q/K/V projections
as residual writes, identify broadcast heads, differentiate parameters, or
implement head/element masks. Final-norm nonlinearities, model-specific residual
scales, compensation and redundancy can invalidate a selection hypothesis.
Validate it through interventions or pilot training in the external framework.

Provide explicit `modules` from `components.list`, `failure_samples`,
`reference_samples`, and fully supported `need`/optional `wrong` concepts.
If one side has fewer than two samples, the standard-error estimate is null.
It is an unpaired descriptive estimate, not a calibrated confidence interval;
correlated rollouts require group-aware statistics in the experiment runner.

`trials.propose` needs a component report, explicit `allowed_modules`,
`max_trainable_parameters`, rank, algorithm (`sft`, `grpo`, `crt`) and an immutable
`training_data_id`. It emits distinct control arms when the budget permits;
counts may differ, so match actual trainable parameters/steps/data in the runner.
Gradient-probe and all-linear baselines belong to that runner and are not faked
by the CLI. Default supervision is reflection-only for CRT and response-only
otherwise. `selected_positions` accepts explicit loss-token positions per sample;
the trainer must verify bounds and construct the actual loss mask.

The whitelist supplied to this read-only tool is a proposal constraint, not an
authorization boundary. The trainer must independently enforce allowed modules,
parameter/step budgets, data identity, loss scope and acceptance gates. Ledger
records should connect observations → candidates/predictions → pilot costs →
target/retention/independent-judge results → accepted/rejected outcome. Learning
a selector from those records is an outer-loop experiment, not a property of a
projection score.

## Health and checkpoint comparison

For frequent, inexpensive checks, use `lens.health` with `observation`, labeled
`canaries` (`sample_id`, `expected_token_ids`), `rank_threshold`, explicit
`min_pass_rate` and `min_samples`. Expected token IDs are alternatives, not a
multi-token sequence. They must be tracked in transported readouts. Missing
tokens or insufficient coverage yield `unknown`. Labels/probes should be held
out from the update loop. No research dataset is downloaded or bundled; adapt
your own probes or upstream evaluation tasks to this format.

For an occasional calibration check, supply `reference` and `candidate` readouts
on identical current weights/inputs, using a separately fitted reference lens,
plus explicit `min_overlap` and `min_cells`. Only transported cells contribute
to health agreement: matching final-model predictions cannot inflate it.
`consistent` means agreement/coverage on these probes, never behavioral safety.

`checkpoint.compare` uses fixed replay. On-policy outputs from two policies are
different inputs and are rejected for paired comparison. Keep those evaluations
separate. Changing the lens requires `allow_lens_change: true`; that output is
explanatory and mixes coordinate changes with model changes.

## Frequency and resource use

| Outer-loop stage | Trigger / tool use |
| --- | --- |
| Fast harness/memory | Per-task feedback; collect failure groups and promising supervision positions without loading a lens on every generation token |
| Adapter trials | Enough informative failed/near-miss groups, harness gains stalled, and trial budget available; reuse one session per immutable candidate, run bounded pilots and controls |
| Consolidation | Several accepted adapters, measurable consolidation benefit, or an operational deadline; replay/retention evaluation, full gates and occasional lens recalibration |

These are event triggers, not universal hourly/daily constants. A legal/medical
rubric pipeline generally has a different feedback budget from high-volume code
completion. Calibrate thresholds and cadence with your ledger.

`metrics.wall_seconds` includes lazy load when it occurs;
`model_loaded_this_request` separates startup from reuse. Readout metrics include
resident elapsed time, forward count, token count and session startup time.
The startup time includes content hashing and may be significant for a large
local checkpoint. Do not sum `session_load_seconds` across replies: it describes
the same one-time startup. Use the retained GPU benchmark/NVML tooling for
utilization, VRAM and measured GPU-hours; CPU wall time is not a GPU benchmark.

`max_tokens`, `max_samples`, `max_readout_cells`, `max_tracked_tokens` and
`max_request_bytes` bound each operation. The optional deadline covers resident work after startup and is checked
between samples, not a hard GPU interrupt. Avoid returning all vocabulary logits
or all positions when a few critical positions suffice. Many tracked concepts
share a sorted-vocabulary rank computation. J++ readout masks are cached.

Use canaries and saved-artifact analyses frequently; reserve full refits for
consolidation or demonstrated degradation. J++ fitting is still external. Do not
make held-out lens audit values a reward/loss target. Use independent behavioral
and retention gates to decide whether a trial is accepted.

## Compatibility and regression fixes

0.3 retains the 0.2 envelope and adds a separate `jlens-tools/1` protocol. Modern
fitting identities now include local weight/tokenizer content; old 0.2 resume
fingerprints must be restarted or resumed with the matching old version.
Existing final lenses without content metadata remain structurally readable;
their calibration identity is unknown. An intentional changed-model diagnostic
requires `allow_model_mismatch`; it does not certify lens validity.

Output guards resolve directory lens aliases and protect metadata/model files.
Recursive merges check original leaf provenance and estimator settings, so
`merge(merge(A,B),A)` is rejected. Arbitrary partially overlapping corpora with
different subset fingerprints still require caller-side deduplication.
Pure logit-only readout can use unfitted layers. J++ presets/downloads and the
historical measurement API are preserved.
