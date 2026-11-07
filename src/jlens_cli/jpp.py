"""Compatible reader for published J++ merged Jacobians, not a training implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, cli

LENS_REVISION = "237dab6ea0e611d964bf407f13ba1655f1a6ae43"
PRESETS = {
    "qwen3.5-9b": {
        "model": "Qwen/Qwen3.5-9B",
        "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "lens_file": "qwen3.5-9b/lens.pt",
        "paper_layers": [4, 8, 12, 16, 20, 24, 28],
    },
    "qwen3.6-27b": {
        "model": "Qwen/Qwen3.6-27B",
        "model_revision": "8f33b98937c6031bb49e713dee3e8e185aafd26f",
        "lens_file": "qwen3.6-27b/lens.pt",
        "paper_layers": [8, 16, 24, 32, 40, 48, 56],
    },
}
for _preset in PRESETS.values():
    _preset.update(lens="koayon/jpp-lenses", lens_revision=LENS_REVISION)


def parser():
    root = argparse.ArgumentParser(
        prog="jpplens", description="Unofficial compatible CLI for published J++ checkpoints."
    )
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    root.add_argument("--debug", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("presets", help="list revision-pinned presets")
    for name in ("download", "inspect", "apply"):
        sub = commands.add_parser(name)
        cli.model_args(sub)
        sub.add_argument("--preset", choices=PRESETS)
        sub.add_argument("--lens")
        sub.add_argument("--lens-file")
        sub.add_argument("--lens-revision")
        sub.add_argument("--cache-dir", type=Path)
        cli.output_arg(sub)
        if name == "download":
            sub.add_argument("--include-model", action="store_true")
        if name == "apply":
            cli.prompt_args(sub)
            layers = sub.add_mutually_exclusive_group()
            layers.add_argument("--layers", type=cli.indices)
            layers.add_argument("--paper-layers", action="store_true")
            sub.add_argument("--positions", type=cli.indices, default=[-1])
            sub.add_argument("--all-positions", action="store_true")
            sub.add_argument("--top-k", type=cli.positive, default=10)
            sub.add_argument("--no-readout-filter", action="store_true")
            sub.add_argument("--format", choices=["json", "text"], default="json")
            sub.add_argument(
                "--model-identity",
                help="original HF ID for a local model copy; must match checkpoint metadata",
            )
    return root


def resolve(args):
    preset = PRESETS.get(args.preset, {})
    # An explicit custom model must not inherit a different model's revision.
    custom_model = args.model is not None and args.model != preset.get("model")
    for key in ("model", "model_revision", "lens", "lens_file", "lens_revision"):
        if key == "model_revision" and custom_model:
            continue
        if getattr(args, key) is None:
            setattr(args, key, preset.get(key))
    args.lens_file = args.lens_file or "lens.pt"
    if getattr(args, "paper_layers", False):
        if not preset:
            raise ValueError("--paper-layers requires --preset")
        args.layers = preset["paper_layers"]


def lens_path(args):
    if not args.lens:
        raise ValueError("--lens or --preset is required")
    path = Path(args.lens)
    if path.is_file():
        return path
    if path.is_dir():
        return path / args.lens_file
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            args.lens,
            args.lens_file,
            revision=args.lens_revision,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    )


def load_checkpoint(path):
    jlens, torch = cli.engine()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("expected a J++ checkpoint dictionary")
    config = checkpoint.get("config", {})
    params = checkpoint.get("parameters", {})
    if not isinstance(config, dict) or not isinstance(params, dict):
        raise ValueError("invalid J++ config/parameters")
    matrices = params.get("jacobians", {})
    sources = checkpoint.get("source_layers", [])
    if (
        not isinstance(matrices, dict)
        or not matrices
        or not isinstance(sources, list)
        or any(type(k) is not int for k in sources)
        or any(type(k) is not int for k in matrices)
        or len(set(sources)) != len(sources)
        or set(sources) != set(matrices)
    ):
        raise ValueError("source_layers must match integer Jacobian keys")
    if set(params) != {"jacobians"}:
        raise ValueError("only merged Jacobian checkpoints are supported")
    if config.get("source_layers", sources) != sources:
        raise ValueError("config source_layers disagree with checkpoint")
    if (
        config.get("lens_type") != "jacobian"
        or config.get("relative_end_transport_layer") != -1
        or not isinstance(config.get("hf_model_name"), str)
    ):
        raise ValueError("expected a model identity and final-block Jacobian transport")
    width = config.get("d_model")
    count = config.get("num_prompts_trained_on")
    if type(width) is not int or type(count) is not int or width <= 0 or count <= 0:
        raise ValueError("invalid checkpoint width/prompt count")
    for layer, matrix in matrices.items():
        if (
            layer < 0
            or not isinstance(matrix, torch.Tensor)
            or matrix.dtype != torch.float32
            or tuple(matrix.shape) != (width, width)
            or not matrix.isfinite().all()
        ):
            raise ValueError(f"invalid/non-finite FP32 Jacobian at layer {layer}")
    lens = jlens.JacobianLens(matrices, n_prompts=count, d_model=width)
    return lens, config


def validate_model(lens, config, model, identity):
    if identity != config["hf_model_name"]:
        raise ValueError(
            "model identity does not match J++ checkpoint; "
            "use --model-identity for an identified local copy"
        )
    if lens.d_model != model.d_model or max(lens.source_layers) >= model.n_layers - 1:
        raise ValueError("checkpoint dimensions/source layers do not match model")


def readout_mask(tokenizer, vocab_size):
    """Unicode letters/digits; filter ranking only, retaining full-vocabulary probabilities."""
    _, torch = cli.engine()
    return torch.tensor(
        [any(c.isalnum() for c in tokenizer.decode([i])) for i in range(vocab_size)],
        dtype=torch.bool,
    )


def top_tokens(logits, tokenizer, k, mask=None):
    if not logits.isfinite().all():
        raise ValueError("non-finite readout logits")
    probabilities = logits.float().softmax(dim=-1)
    ranking = probabilities
    if mask is not None:
        mask = mask.to(probabilities.device)
        ranking = probabilities.masked_fill(~mask, -1)
        k = min(k, int(mask.sum()))
    k = min(k, logits.shape[-1])
    _, ids = ranking.topk(k, dim=-1)
    values = probabilities.gather(-1, ids)
    return [
        [
            {"token_id": i, "token": tokenizer.decode([i]), "probability": p}
            for i, p in zip(row_ids, row_values, strict=True)
        ]
        for row_ids, row_values in zip(ids.tolist(), values.tolist(), strict=True)
    ]


def infer(args, model, lens, config, prompt, mask=None):
    validate_model(lens, config, model, args.model_identity or args.model)
    ids = model.encode(prompt, max_length=args.max_seq_len)[0].tolist()
    positions = list(range(len(ids))) if args.all_positions else args.positions
    if not ids or any(p < -len(ids) or p >= len(ids) for p in positions):
        raise ValueError("positions must be within the encoded sequence")
    positions = [p % len(ids) for p in positions]
    if len(set(positions)) != len(positions):
        raise ValueError("positions refer to the same token")
    logits, final, _ = lens.apply(
        model,
        prompt,
        layers=args.layers,
        positions=positions,
        max_seq_len=args.max_seq_len,
    )
    if not args.no_readout_filter and mask is None:
        mask = readout_mask(model.tokenizer, final.shape[-1])
    if args.no_readout_filter:
        mask = None
    return {
        "schema_version": 1,
        "method": "j++-compatible-merged-checkpoint",
        "model": args.model,
        "model_revision": args.model_revision,
        "lens": args.lens,
        "lens_revision": args.lens_revision,
        "readout_filter": mask is not None,
        "probability_basis": "full-vocabulary",
        "input_token_ids": ids,
        "positions": positions,
        "layers": {
            str(k): top_tokens(v, model.tokenizer, args.top_k, mask)
            for k, v in sorted(logits.items())
        },
        # Actual model predictions remain unfiltered.
        "model_output": top_tokens(final, model.tokenizer, args.top_k),
    }


def run(args):
    if args.command == "presets":
        print(json.dumps(PRESETS, indent=2))
        return
    resolve(args)
    cli.prepare_output(args.output, args.force)
    prompt = cli.read_prompt(args) if args.command == "apply" else None
    path = lens_path(args)
    if args.output and args.output.resolve() == path.resolve():
        raise ValueError("output must differ from checkpoint")
    if args.command == "download":
        result = {
            "lens_path": str(path),
            "lens_bytes": path.stat().st_size,
            "lens_revision": args.lens_revision,
        }
        if args.include_model:
            if not args.model:
                raise ValueError("--include-model requires --model or --preset")
            from huggingface_hub import snapshot_download

            result["model_path"] = snapshot_download(
                args.model,
                revision=args.model_revision,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
            )
            result["model_revision"] = args.model_revision
        cli.emit(result, args)
        return
    lens, config = load_checkpoint(path)
    if args.command == "inspect":
        cli.emit(
            {
                "schema_version": 1,
                "path": str(path),
                "bytes": path.stat().st_size,
                "config": config,
                "source_layers": lens.source_layers,
                "matrix_dtype": "float32",
                "d_model": lens.d_model,
            },
            args,
        )
        return
    if not args.model:
        args.model = config["hf_model_name"]
    model = cli.load_model(args)
    validate_model(lens, config, model, args.model_identity or args.model)
    # Keep selected matrices resident; transfer and validation are outside inference.
    selected = args.layers if args.layers is not None else lens.source_layers
    if not selected or set(selected) - set(lens.source_layers):
        raise ValueError("requested layers are not in checkpoint source_layers")
    for layer in selected:
        lens.jacobians[layer] = lens.jacobians[layer].to(model.input_device)
    result = infer(args, model, lens, config, prompt)
    if args.format == "text":
        lines = []
        for offset, position in enumerate(result["positions"]):
            lines.append(f"Position {position}")
            for layer, rows in [*result["layers"].items(), ("model", result["model_output"])]:
                tokens = ", ".join(f"{v['token']!r} ({v['probability']:.4f})" for v in rows[offset])
                lines.append(f"  {layer}: {tokens}")
        cli.emit("\n".join(lines), args)
    else:
        cli.emit(result, args)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError, KeyError, ImportError, TypeError) as exc:
        if args.debug:
            raise
        print(f"jpplens: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
