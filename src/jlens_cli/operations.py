"""Fit, inspect, merge and read out a lens using the upstream implementation."""

from __future__ import annotations

import json
from pathlib import Path

from jlens_cli import UPSTREAM_COMMIT, __version__
from jlens_cli.io import (
    CLIError,
    atomic_lens,
    digest,
    ensure_output,
    indices,
    read_metadata,
    read_records,
    read_text,
    write_json,
)
from jlens_cli.runtime import dependencies, lens_info, load_lens, load_model, validate_lens


def model_identity(args, model) -> dict:
    return {
        "model": args.model,
        "revision": args.revision,
        "tokenizer": args.tokenizer or args.model,
        "tokenizer_revision": args.tokenizer_revision if args.tokenizer else args.revision,
        "resolved_model_revision": getattr(model._hf_model.config, "_commit_hash", None),
        "dtype": str(next(model._hf_model.parameters()).dtype),
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "force_bos": not args.no_force_bos,
        "attn_implementation": args.attn_implementation,
    }


def fit(args) -> dict:
    records = read_records(args.prompts, text_field=args.text_field, input_format=args.input_format)
    if args.limit:
        records = records[: args.limit]
    records = records[args.shard_index :: args.num_shards]
    if not records:
        raise CLIError("The selected shard contains no prompts.")
    ensure_output(args.output, force=args.force)
    ensure_output(f"{args.output}.meta.json", force=args.force)
    model = load_model(args)
    torch, jlens = dependencies()
    layers = indices(args.layers, model.n_layers, default=list(range(model.n_layers - 1)))
    if any(layer >= model.n_layers - 1 for layer in layers):
        raise CLIError(
            "Fit source layers must precede the final layer; omit --layers for all eligible layers."
        )
    prompts = [record["text"] for record in records]
    identity = model_identity(args, model)
    configuration = {
        **identity,
        "layers": sorted(layers),
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "dim_batch": args.dim_batch,
        "prompts_sha256": digest(prompts),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    fingerprint = digest(configuration)
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        sidecar = Path(f"{args.checkpoint}.meta.json")
        if checkpoint.exists() and args.resume:
            if not sidecar.is_file():
                raise CLIError(
                    "Cannot verify this checkpoint's inputs. Choose a new checkpoint, or "
                    "use --no-resume --force to replace it.",
                    code="checkpoint_mismatch",
                )
            previous = json.loads(sidecar.read_text(encoding="utf-8"))
            if previous.get("fingerprint") != fingerprint:
                raise CLIError(
                    "Checkpoint model, prompts or fitting settings differ. Choose a new "
                    "checkpoint, or use --no-resume --force.",
                    code="checkpoint_mismatch",
                )
        elif checkpoint.exists() and not args.force:
            raise CLIError(
                "Replacing a checkpoint requires --no-resume --force.", code="output_exists"
            )
        elif checkpoint.exists() and not args.resume:
            # Never pair an old checkpoint with a new fingerprint after a failed restart.
            checkpoint.unlink()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        write_json(sidecar, {"fingerprint": fingerprint, "configuration": configuration})
    lens = jlens.fit(
        model,
        prompts,
        source_layers=layers,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every or None,
        resume=args.resume,
    )
    validate_lens(lens, model)
    atomic_lens(lens, args.output, dtype=getattr(torch, args.save_dtype))
    metadata = {
        **identity,
        **lens_info(lens),
        "cli_version": __version__,
        "upstream_commit": UPSTREAM_COMMIT,
        "fit": configuration,
        "fingerprint": fingerprint,
        "input_prompts": len(prompts),
        "skipped_prompts": len(prompts) - lens.n_prompts,
        "save_dtype": args.save_dtype,
    }
    write_json(f"{args.output}.meta.json", metadata)
    return {"output": args.output, "metadata_path": f"{args.output}.meta.json", **metadata}


def prompt_records(args) -> list[dict]:
    if args.input:
        return read_records(args.input, text_field=args.text_field, input_format=args.input_format)
    prompt = args.prompt if args.prompt is not None else read_text(args.prompt_file)
    if not prompt.strip():
        raise CLIError("Prompt must not be empty.")
    return [{"id": 0, "text": prompt}]


def rows(logits, positions, token_ids, tokenizer, top_k: int, tracked_ids: list[int]) -> list[dict]:
    torch, _ = dependencies()
    if not torch.isfinite(logits).all().item():
        raise CLIError(
            "Non-finite logits encountered; try --dtype float32.", code="nonfinite_logits"
        )
    vocab_size = logits.shape[-1]
    if any(token_id < 0 or token_id >= vocab_size for token_id in tracked_ids):
        raise CLIError(f"Tracked token IDs must be in [0, {vocab_size}).")
    log_probs = logits.float().log_softmax(-1)
    values, ids = logits.topk(min(top_k, vocab_size), dim=-1)
    top_probs = log_probs.gather(-1, ids).exp().tolist()
    results = []
    for index, position in enumerate(positions):
        top = [
            {
                "token_id": token_id,
                "token": tokenizer.decode([token_id]),
                "logit": logit,
                "probability": probability,
            }
            for token_id, logit, probability in zip(
                ids[index].tolist(), values[index].tolist(), top_probs[index], strict=True
            )
        ]
        tracked = [
            {
                "token_id": token_id,
                "token": tokenizer.decode([token_id]),
                "rank": int((logits[index] > logits[index, token_id]).sum().item()) + 1,
                "probability": log_probs[index, token_id].exp().item(),
            }
            for token_id in tracked_ids
        ]
        results.append(
            {
                "position": position,
                "input_token_id": token_ids[position],
                "input_token": tokenizer.decode([token_ids[position]]),
                "top_tokens": top,
                "tracked_tokens": tracked,
            }
        )
    return results


def apply(args) -> dict:
    records = prompt_records(args)
    if args.command == "visualize" and len(records) != 1:
        raise CLIError("visualize requires exactly one prompt. Use apply for a batch.")
    ensure_output(args.output, force=args.force)
    model = load_model(args)
    lens, metadata = load_lens(args, model)
    layers = indices(args.layers, model.n_layers, default=lens.source_layers)
    unknown = set(layers) - set(lens.source_layers)
    if unknown:
        raise CLIError(
            f"Layers {sorted(unknown)} were not fitted; fitted layers: {lens.source_layers}."
        )
    results = []
    for record in records:
        prompt = record["text"]
        ids = model.encode(prompt, max_length=args.max_seq_len)[0].tolist()
        if not ids:
            raise CLIError(f"Prompt {record['id']!r} encodes to an empty sequence.")
        positions = indices(args.positions, len(ids))
        logits, model_logits, input_ids = lens.apply(
            model,
            prompt,
            layers=layers,
            positions=positions,
            max_seq_len=args.max_seq_len,
        )
        if input_ids[0].tolist() != ids:
            raise CLIError(
                "Tokenizer produced inconsistent encodings.", code="runtime_error", exit_code=1
            )
        readouts = [
            {
                "kind": "jacobian",
                "layer": layer,
                "positions": rows(
                    logits[layer], positions, ids, model.tokenizer, args.top_k, args.track_token_id
                ),
            }
            for layer in layers
        ]
        if args.baseline:
            baseline, _, _ = lens.apply(
                model,
                prompt,
                layers=layers,
                positions=positions,
                max_seq_len=args.max_seq_len,
                use_jacobian=False,
            )
            readouts.extend(
                {
                    "kind": "logit",
                    "layer": layer,
                    "positions": rows(
                        baseline[layer],
                        positions,
                        ids,
                        model.tokenizer,
                        args.top_k,
                        args.track_token_id,
                    ),
                }
                for layer in layers
            )
        readouts.append(
            {
                "kind": "model",
                "layer": model.n_layers - 1,
                "positions": rows(
                    model_logits, positions, ids, model.tokenizer, args.top_k, args.track_token_id
                ),
            }
        )
        results.append(
            {
                "id": record["id"],
                "prompt": prompt,
                "num_tokens": len(ids),
                "tokens": [
                    {
                        "position": i,
                        "token_id": token_id,
                        "token": model.tokenizer.decode([token_id]),
                    }
                    for i, token_id in enumerate(ids)
                ],
                "positions": positions,
                "readouts": readouts,
            }
        )
    return {
        **model_identity(args, model),
        "lens": args.lens,
        "lens_revision": args.lens_revision,
        "lens_filename": args.lens_filename,
        "lens_info": lens_info(lens),
        "lens_metadata": metadata,
        "max_seq_len": args.max_seq_len,
        "top_k": args.top_k,
        "results": results,
    }


def inspect(args) -> dict:
    lens, metadata = load_lens(args)
    return {
        "lens": args.lens,
        **lens_info(lens),
        "metadata": metadata,
        "matrices": [
            {
                "layer": layer,
                "shape": list(matrix.shape),
                "dtype_in_memory": str(matrix.dtype),
                "frobenius_norm": matrix.norm().item(),
            }
            for layer, matrix in sorted(lens.jacobians.items())
        ],
    }


def merge(args) -> dict:
    torch, jlens = dependencies()
    ensure_output(args.output, force=args.force)
    ensure_output(f"{args.output}.meta.json", force=args.force)
    lenses, metadata = [], []
    if len({str(Path(path).resolve()) for path in args.lenses}) != len(args.lenses):
        raise CLIError("The same lens file cannot be merged more than once.")
    for path in args.lenses:
        lens = jlens.JacobianLens.load(path)
        validate_lens(lens)
        lenses.append(lens)
        metadata.append(read_metadata(path))
    identities = [item.get("model") for item in metadata if item and item.get("model")]
    revisions = [item.get("revision") for item in metadata if item and item.get("model")]
    if len(set(identities)) > 1 or len(set(revisions)) > 1:
        raise CLIError(
            "Cannot merge lenses with different model identities/revisions.",
            code="incompatible_lens",
        )
    configurations = [item.get("fit") for item in metadata if item and item.get("fit")]
    comparable = [
        {k: v for k, v in item.items() if k not in ("prompts_sha256", "shard_index", "num_shards")}
        for item in configurations
    ]
    if comparable and any(item != comparable[0] for item in comparable[1:]):
        raise CLIError(
            "Cannot merge lenses fitted with different estimator settings.",
            code="incompatible_lens",
        )
    hashes = [item["prompts_sha256"] for item in configurations]
    if len(set(hashes)) < len(hashes):
        raise CLIError("Input metadata identifies identical prompt subsets; use disjoint shards.")
    lens = jlens.JacobianLens.merge(lenses)
    validate_lens(lens)
    atomic_lens(lens, args.output, dtype=getattr(torch, args.save_dtype))
    all_identified = len(identities) == len(lenses)
    result = {
        "output": args.output,
        **lens_info(lens),
        "model": identities[0] if identities and all_identified else None,
        "revision": revisions[0] if revisions and all_identified else None,
        "upstream_commit": UPSTREAM_COMMIT,
        "cli_version": __version__,
        "inputs": args.lenses,
        "input_metadata": metadata,
        "input_prompt_counts": [item.n_prompts for item in lenses],
        "save_dtype": args.save_dtype,
        "provenance_complete": all_identified,
    }
    write_json(f"{args.output}.meta.json", result)
    return result
