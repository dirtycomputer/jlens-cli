"""CLI boundary: lazy imports, validated input, and stable JSON output."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import __version__

UPSTREAM_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
PRESETS = {
    "qwen3.5-4b": {
        "model": "Qwen/Qwen3.5-4B",
        "lens": "neuronpedia/jacobian-lens",
        "lens_revision": "qwen-n1000",
        "lens_file": "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    },
    "qwen3.6-27b": {
        "model": "Qwen/Qwen3.6-27B",
        "lens": "neuronpedia/jacobian-lens",
        "lens_revision": "qwen-n1000",
        "lens_file": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    },
}


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def indices(value):
    try:
        result = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("duplicate indices are not allowed")
    return result


def engine():
    try:
        import jlens
        import torch
    except ImportError as exc:
        raise ValueError(
            "runtime dependencies are missing; install with: pip install '.[engine]' "
            "(from the repository directory)"
        ) from exc
    return jlens, torch


def output_arg(parser, required=False):
    parser.add_argument("-o", "--output", type=Path, required=required)
    parser.add_argument("--force", action="store_true", help="replace an existing output")


def model_args(parser):
    parser.add_argument("--model", help="Hugging Face model ID or local model directory")
    parser.add_argument("--model-revision", help="model and tokenizer revision")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="explicitly allow Hugging Face repository Python code",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="load model/tokenizer from local files/cache only",
    )


def lens_args(parser):
    parser.add_argument("--preset", choices=PRESETS)
    parser.add_argument("--lens", help="local .pt, local directory, or Hugging Face repo ID")
    parser.add_argument("--lens-file", help="filename within lens directory/repository")
    parser.add_argument("--lens-revision", help="lens repository revision")


def prompt_args(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="raw model input; chat templates are not added")
    group.add_argument("--prompt-file", type=Path, help="UTF-8 text file, or - for stdin")
    parser.add_argument("--max-seq-len", type=positive, default=512)


def parser():
    root = argparse.ArgumentParser(
        prog="jlens", description="Unofficial CLI for Anthropic's Jacobian lens."
    )
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    root.add_argument("--debug", action="store_true", help="show tracebacks on failure")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("presets", help="list model/lens presets as JSON")
    apply = commands.add_parser("apply", help="decode internal activations into top tokens")
    model_args(apply)
    lens_args(apply)
    prompt_args(apply)
    apply.add_argument("--layers", type=indices, help="comma-separated layer indices")
    apply.add_argument(
        "--positions",
        type=indices,
        default=[-1],
        help="comma-separated token positions (default: -1); use = for negatives",
    )
    apply.add_argument("--all-positions", action="store_true")
    apply.add_argument("--top-k", type=positive, default=5)
    apply.add_argument(
        "--logit-lens", action="store_true", help="use the untransported logit-lens baseline"
    )
    apply.add_argument("--format", choices=["json", "text"], default="json")
    output_arg(apply)
    render = commands.add_parser("render", help="create a self-contained interactive HTML slice")
    model_args(render)
    lens_args(render)
    prompt_args(render)
    render.add_argument("--layer-stride", type=positive, default=2)
    render.add_argument("--last-n-tokens", type=positive)
    render.add_argument("--top-k", type=positive, default=10)
    render.add_argument("--max-tracked", type=positive, default=100)
    render.add_argument("--mask-display", action="store_true")
    render.add_argument("--title", default="Jacobian lens")
    output_arg(render, required=True)
    fit = commands.add_parser("fit", help="fit a lens using a local text or JSONL corpus")
    model_args(fit)
    fit.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help=".txt: one prompt per nonempty line; .jsonl: objects with text",
    )
    fit.add_argument("--limit", type=positive)
    fit.add_argument("--layers", type=indices)
    fit.add_argument("--dim-batch", type=positive, default=8)
    fit.add_argument("--max-seq-len", type=positive, default=128)
    fit.add_argument("--skip-first", type=nonnegative, default=16)
    fit.add_argument("--checkpoint", type=Path)
    fit.add_argument("--checkpoint-every", type=positive, default=1)
    fit.add_argument("--resume", action="store_true")
    output_arg(fit, required=True)
    inspect = commands.add_parser("inspect", help="show lens metadata without loading a model")
    lens_args(inspect)
    output_arg(inspect)
    merge = commands.add_parser("merge", help="merge lenses weighted by prompt count")
    merge.add_argument("lenses", nargs="+", type=Path)
    output_arg(merge, required=True)
    return root


def prepare_output(path, force=False):
    if path is None:
        return
    if path.exists() and not force:
        raise ValueError(f"output already exists: {path}; use --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)


def emit(value, args):
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    )
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Saved {args.output}", file=sys.stderr)
    else:
        print(text)


def resolve(args):
    preset = PRESETS.get(getattr(args, "preset", None), {})
    for key in ("model", "lens", "lens_file", "lens_revision"):
        if not getattr(args, key, None):
            setattr(args, key, preset.get(key))
    args.lens_file = args.lens_file or "lens.pt"


def read_prompt(args):
    if args.prompt is not None:
        text = args.prompt
    elif str(args.prompt_file) == "-":
        text = sys.stdin.read()
    else:
        text = args.prompt_file.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("prompt must not be empty")
    return text


def read_corpus(path, limit=None):
    prompts = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if path.suffix.lower() == ".jsonl":
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
                text = obj.get("text") if isinstance(obj, dict) else None
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{path}:{number}: expected a nonempty 'text' string")
            else:
                text = line.rstrip("\r\n")
            prompts.append(text)
            if limit and len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError("corpus contains no prompts")
    return prompts


def load_model(args):
    if not args.model:
        raise ValueError("--model or --preset is required")
    jlens, torch = engine()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    dtype = args.dtype
    if dtype == "auto":
        # float32 is the conservative default for fitting and CPU execution.
        dtype = (
            "bfloat16"
            if device.startswith("cuda") and torch.cuda.is_bf16_supported()
            else "float32"
        )
    options = dict(
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    print(f"Loading {args.model} on {device} ({dtype})", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, dtype), attn_implementation="eager", **options
    ).to(device)
    return jlens.from_hf(hf_model, tokenizer)


def load_lens(args):
    if not args.lens:
        raise ValueError("--lens or --preset is required")
    jlens, _ = engine()
    if getattr(args, "local_files_only", False) and not Path(args.lens).exists():
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            args.lens, args.lens_file, revision=args.lens_revision, local_files_only=True
        )
        return jlens.JacobianLens.load(path)
    return jlens.JacobianLens.from_pretrained(
        args.lens, filename=args.lens_file, revision=args.lens_revision
    )


def check_lens(lens, model=None):
    if not lens.source_layers or lens.n_prompts <= 0 or lens.d_model <= 0:
        raise ValueError("lens must contain fitted layers and a positive prompt count/width")
    for layer, matrix in lens.jacobians.items():
        if layer < 0 or tuple(matrix.shape) != (lens.d_model, lens.d_model):
            raise ValueError(f"invalid Jacobian shape or layer index: {layer}")
        if not matrix.isfinite().all():
            raise ValueError(f"non-finite Jacobian at layer {layer}")
    if model and (lens.d_model != model.d_model or max(lens.source_layers) >= model.n_layers):
        raise ValueError("lens dimensions/layers do not match the selected model")


def top_tokens(logits, tokenizer, k):
    probabilities = logits.softmax(dim=-1)
    values, ids = probabilities.topk(min(k, logits.shape[-1]), dim=-1)
    return [
        [
            {
                "token_id": token_id,
                "token": tokenizer.decode([token_id]),
                "probability": probability,
            }
            for token_id, probability in zip(row_ids, row_values, strict=True)
        ]
        for row_ids, row_values in zip(ids.tolist(), values.tolist(), strict=True)
    ]


def apply_command(args, model, lens, prompt):
    ids = model.encode(prompt, max_length=args.max_seq_len)[0].tolist()
    positions = list(range(len(ids))) if args.all_positions else args.positions
    if not ids or any(p < -len(ids) or p >= len(ids) for p in positions):
        raise ValueError(f"positions must be within [{-len(ids)}, {len(ids) - 1}]")
    positions = [p % len(ids) for p in positions]
    if len(set(positions)) != len(positions):
        raise ValueError("positions refer to the same token after resolving negative indices")
    logits, final, _ = lens.apply(
        model,
        prompt,
        layers=args.layers,
        positions=positions,
        max_seq_len=args.max_seq_len,
        use_jacobian=not args.logit_lens,
    )
    result = {
        "schema_version": 1,
        "model": args.model,
        "lens": args.lens,
        "method": "logit-lens" if args.logit_lens else "jacobian-lens",
        "input_token_ids": ids,
        "positions": positions,
        "input_tokens": [model.tokenizer.decode([ids[p]]) for p in positions],
        "layers": {
            str(layer): top_tokens(values, model.tokenizer, args.top_k)
            for layer, values in sorted(logits.items())
        },
        "model_output": top_tokens(final, model.tokenizer, args.top_k),
    }
    if args.format == "text":
        lines = []
        for offset, position in enumerate(positions):
            lines.append(f"Position {position}: {result['input_tokens'][offset]!r}")
            for label, rows in [*result["layers"].items(), ("model", result["model_output"])]:
                tokens = ", ".join(f"{v['token']!r} ({v['probability']:.4f})" for v in rows[offset])
                lines.append(f"  {label:>6}: {tokens}")
        emit("\n".join(lines), args)
    else:
        emit(result, args)


def fit_command(args):
    if not args.model:
        raise ValueError("--model is required")
    if args.max_seq_len <= args.skip_first + 1:
        raise ValueError("--max-seq-len must exceed --skip-first + 1")
    prompts = read_corpus(args.prompts, args.limit)
    signature = {
        "model": args.model,
        "model_revision": args.model_revision,
        "layers": args.layers,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "dtype": args.dtype,
        "upstream_commit": UPSTREAM_COMMIT,
        "prompts_sha256": hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    checkpoint = args.checkpoint
    if args.resume and (checkpoint is None or not checkpoint.is_file()):
        raise ValueError("--resume requires an existing --checkpoint")
    if checkpoint:
        if checkpoint.resolve() == args.output.resolve():
            raise ValueError("--checkpoint and --output must be different files")
        metadata = checkpoint.with_name(checkpoint.name + ".cli.json")
        if args.resume:
            if not metadata.exists() or json.loads(metadata.read_text()) != signature:
                raise ValueError("checkpoint metadata does not match model, corpus, or settings")
        else:
            prepare_output(checkpoint, False)
            prepare_output(metadata, False)
    jlens, torch = engine()
    jlens.configure_logging()
    model = load_model(args)
    if checkpoint and not args.resume:
        metadata.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    lens = jlens.fit(
        model,
        prompts=prompts,
        source_layers=args.layers,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=str(checkpoint) if checkpoint else None,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    check_lens(lens, model)
    lens.save(str(args.output), dtype=torch.float32)
    print(f"Saved {args.output} ({lens.n_prompts} prompts)", file=sys.stderr)


def run(args):
    if args.command == "presets":
        print(json.dumps(PRESETS, ensure_ascii=False, indent=2))
        return
    prepare_output(args.output, args.force)
    if args.command == "fit":
        fit_command(args)
        return
    if args.command == "merge":
        if args.output.resolve() in {path.resolve() for path in args.lenses}:
            raise ValueError("merge output must be different from its input files")
        jlens, torch = engine()
        lenses = [jlens.JacobianLens.load(str(path)) for path in args.lenses]
        for lens in lenses:
            check_lens(lens)
        merged = jlens.JacobianLens.merge(lenses)
        check_lens(merged)
        merged.save(str(args.output), dtype=torch.float32)
        print(f"Saved {args.output} ({merged.n_prompts} prompts)", file=sys.stderr)
        return
    resolve(args)
    if args.command == "inspect":
        lens = load_lens(args)
        check_lens(lens)
        emit(
            {
                "schema_version": 1,
                "d_model": lens.d_model,
                "n_prompts": lens.n_prompts,
                "source_layers": lens.source_layers,
                "matrix_shapes": {str(k): list(v.shape) for k, v in lens.jacobians.items()},
            },
            args,
        )
        return
    prompt = read_prompt(args)
    if not args.model:
        raise ValueError("--model or --preset is required")
    lens = load_lens(args)
    model = load_model(args)
    check_lens(lens, model)
    if args.command == "apply":
        apply_command(args, model, lens, prompt)
    else:
        from jlens.vis import build_page, compute_slice

        data = compute_slice(
            model,
            lens,
            prompt,
            top_n=min(args.top_k, model._lm_head.out_features),
            layer_stride=args.layer_stride,
            last_n_tokens=args.last_n_tokens,
            max_seq_len=args.max_seq_len,
            mask_display=args.mask_display,
            max_tracked=args.max_tracked,
        )
        page, _, _ = build_page(
            data, prompt, title=args.title, description=f"Model: {args.model}", mode="embed"
        )
        emit(page, args)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        if args.debug:
            raise
        print(f"jlens: error: {exc}", file=sys.stderr)
        return 1
    return 0
