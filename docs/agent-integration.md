# Agent integration

Use this CLI as a local process tool. It does not require MCP, an HTTP server, or a particular agent framework. See `examples/agent_tool.py` for an argument-array adapter.

1. Run `j-lens schema` for the contract and `j-lens doctor` for dependency/device availability.
2. Provide the model identifier and a matching pre-fitted lens, or run a separate fit job.
3. Call `j-lens apply` and parse the success envelope from stdout.
4. On failure, use the exit code and parse the final stderr line as the error envelope.

## Request example

```bash
j-lens apply --model /models/checkpoint --lens /lenses/lens.pt \
  --prompt-file - --positions=-3,-1 --layers 0:12:2 \
  --top-k 10 --track-token-id 1234 --baseline
```

The caller writes the raw prompt to stdin. Layers/positions refer to the encoded prompt, not characters or generated tokens. For multiple requests against the same model/lens, use `--input requests.jsonl` so the model is loaded only once. Each standalone invocation otherwise reloads the model. For a resident sequential worker, use `serve` and the versioned protocol in [RSI tools](rsi-tools.md). Ordinary `apply` remains a single batch invocation; concurrent model serving is not implemented.

## Result structure

```text
schema_version: "1.0"
status: "ok"
command: "apply"
data:
  model, revision, resolved_model_revision, tokenizer, dtype, n_layers, d_model
  lens, lens_filename, lens_revision, lens_info, lens_metadata
  max_seq_len, top_k
  results[]:
    id, prompt, num_tokens
    tokens[]: position, token_id, token
    positions[]: absolute selected input positions
    readouts[]:
      kind: "jacobian" | "jpp" | "logit" | "model"
      layer: zero-based output block index
      positions[]:
        position, input_token_id, input_token
        top_tokens[]: token_id, token, logit, probability
        tracked_tokens[]: token_id, token, rank, filtered_rank, logit, probability
```

For `--output-format jsonl`, each line has the same envelope and run metadata but `data.result` contains one sample instead of `data.results`. Output is written after all samples succeed, so a failed batch has no partially successful result stream. The program keeps compact readouts in memory; upstream logits for the selected layers/positions still require memory proportional to vocabulary size.

With `--output FILE`, stdout contains a summary (`output`, `output_format`, `num_prompts`), while the file contains the full response. Existing outputs require `--force`.

## Error codes

| Code | Typical cause | Response |
| --- | --- | --- |
| `usage_error` | Missing or invalid command-line arguments | Check subcommand help |
| `invalid_input` | Invalid positions, bad corpus, short prompts, incompatible shapes | Correct inputs |
| `output_exists` | Refusing to overwrite an existing output | Pick a new path or intentionally use `--force` |
| `checkpoint_mismatch` | Resume inputs/settings changed, or metadata absent | Use the original inputs or start a new fit |
| `incompatible_lens` | Model width, layer bounds, known identity or estimator mismatch | Select matching artifacts |
| `missing_dependency` | ML runtime unavailable | Install package dependencies |
| `nonfinite_logits` | NaN/Inf model outputs | Inspect weights, attention implementation and dtype |
| `runtime_error` | Other runtime failures, including resource exhaustion | Inspect stderr diagnostics |
| `interrupted` | User/process interruption | Resume a configured fit checkpoint if applicable |

`--debug` is intended for development and emits normal tracebacks instead of the stable error contract. Avoid it in machine integrations. Never put secret credentials in CLI arguments; use the standard Hugging Face authentication/cache mechanism.

## Use in RSI experiments

A useful observation is a pattern in readouts across layers/positions, including a comparison against the untransported baseline and the model's final logits. A candidate intervention still needs an independent evaluation. This tool does not output parameter gradients, recommend which weights to update, or prove that a decoded concept causally determines a decision. Fitting computes activation Jacobians while the model parameters remain frozen.

## J++ and compatibility

Use `jpplens` instead of `jlens` for published merged J++ checkpoints. It supports the same batch/readout protocol, `schema`, `doctor`, `visualize`, plus the existing `presets`, `download`, `--paper-layers` and `--model-identity` options. J++ fitting remains out of scope.

`rank` always counts the full vocabulary; `filtered_rank` counts eligible alphanumeric tokens only and is null for excluded tokens or unfiltered readouts. Readout filtering changes the top-k ranking, never the probability denominator. Final model predictions and logit baselines are unfiltered.

The 0.1 JSON format and checkpoint behavior remain available with `jlens --legacy ...` / `jpplens --legacy ...`. The `--legacy` switch must be first. See [migration](migration-0.2.md).
