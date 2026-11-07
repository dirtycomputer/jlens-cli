"""Lazy imports and the Hugging Face / upstream jlens boundary."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from jlens_cli import UPSTREAM_COMMIT
from jlens_cli.io import CLIError, read_metadata


def dependencies():
    try:
        import jlens
        import torch
    except ImportError as exc:
        raise CLIError(
            "Runtime dependencies are unavailable. Install j-lens-cli with its dependencies "
            "(see README). Original error: " + str(exc),
            code="missing_dependency",
            exit_code=1,
        ) from exc
    return torch, jlens


def load_model(args):
    torch, jlens = dependencies()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise CLIError(str(exc), code="missing_dependency", exit_code=1) from exc
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if args.device_map and args.device != "auto":
        raise CLIError("Use either --device or --device-map, not both.")
    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
    shared = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    kwargs = dict(
        shared, revision=args.revision, dtype=dtype, attn_implementation=args.attn_implementation
    )
    if args.device_map:
        kwargs["device_map"] = args.device_map
    hf = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if not args.device_map:
        hf.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        revision=args.tokenizer_revision if args.tokenizer else args.revision,
        **shared,
    )
    return jlens.from_hf(hf, tokenizer, force_bos=not args.no_force_bos)


def validate_lens(lens, model=None):
    torch, _ = dependencies()
    if not lens.source_layers or lens.n_prompts <= 0 or lens.d_model <= 0:
        raise CLIError(
            "Lens must have fitted layers, a positive width and a positive prompt count."
        )
    for layer, matrix in lens.jacobians.items():
        if (
            not isinstance(layer, int)
            or layer < 0
            or tuple(matrix.shape) != (lens.d_model, lens.d_model)
        ):
            raise CLIError(f"Invalid lens matrix at layer {layer}.")
        if not torch.isfinite(matrix).all().item():
            raise CLIError(f"Lens matrix at layer {layer} contains NaN or infinity.")
    if model is not None:
        if lens.d_model != model.d_model:
            raise CLIError(
                f"Lens width {lens.d_model} differs from model width {model.d_model}.",
                code="incompatible_lens",
            )
        if any(layer >= model.n_layers for layer in lens.source_layers):
            raise CLIError("Lens contains layers outside this model.", code="incompatible_lens")


def load_lens(args, model=None):
    _, jlens = dependencies()
    if getattr(args, "local_files_only", False) and not Path(args.lens).exists():
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            args.lens,
            filename=args.lens_filename,
            revision=args.lens_revision,
            local_files_only=True,
        )
        lens = jlens.JacobianLens.load(path)
    else:
        lens = jlens.JacobianLens.from_pretrained(
            args.lens,
            filename=args.lens_filename,
            revision=args.lens_revision,
        )
    validate_lens(lens, model)
    metadata = read_metadata(args.lens)
    if model is not None and metadata:
        source = metadata.get("model")
        revision = metadata.get("revision")
        if (
            source
            and (source != args.model or revision != args.revision)
            and not args.allow_model_mismatch
        ):
            raise CLIError(
                f"Lens metadata identifies {source!r} at revision {revision!r}; requested "
                f"{args.model!r} at {args.revision!r}. Use the matching model, or explicitly "
                "--allow-model-mismatch if these identify the same weights.",
                code="incompatible_lens",
            )
    return lens, metadata


def lens_info(lens) -> dict:
    return {
        "d_model": lens.d_model,
        "source_layers": lens.source_layers,
        "n_prompts": lens.n_prompts,
    }


def doctor() -> dict:
    versions = {}
    for name in ("j-lens-cli", "jlens", "torch", "transformers", "huggingface_hub", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {
        "packages": versions,
        "expected_upstream_commit": UPSTREAM_COMMIT,
        "cuda_available": False,
        "mps_available": False,
    }
    try:
        import torch

        result.update(
            cuda_available=torch.cuda.is_available(),
            mps_available=torch.backends.mps.is_available(),
        )
    except (ImportError, OSError) as exc:
        result["torch_error"] = str(exc)
    try:
        import jlens

        result["upstream_importable"] = all(
            hasattr(jlens, name) for name in ("fit", "from_hf", "JacobianLens")
        )
    except (ImportError, OSError) as exc:
        result.update(upstream_importable=False, upstream_error=str(exc))
    result["runtime_ready"] = bool(result["upstream_importable"] and versions["transformers"])
    return result
