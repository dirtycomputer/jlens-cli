"""Public command-line interface. Stdout is reserved for results."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

from jlens_cli import SCHEMA_VERSION, UPSTREAM_COMMIT, __version__
from jlens_cli.io import CLIError, atomic_text, ensure_output


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise CLIError(message, code="usage_error")


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def model_options(parser):
    parser.add_argument("--model", help="Hugging Face model ID or local directory")
    parser.add_argument(
        "--revision", "--model-revision", help="Model revision; prefer an immutable commit"
    )
    parser.add_argument("--preset", help="Model/lens preset; see presets")
    parser.add_argument("--tokenizer", help="Optional separate tokenizer ID/directory")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--device-map", choices=["auto"], help="Requires the [sharded] extra")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument(
        "--attn-implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"]
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Opt in to executing model repository code"
    )
    parser.add_argument(
        "--no-force-bos", action="store_true", help="Disable upstream's BOS adjustment"
    )


def lens_options(parser):
    parser.add_argument("--lens", help="Local .pt file, directory or Hugging Face lens repo")
    parser.add_argument("--lens-filename", "--lens-file")
    parser.add_argument("--lens-revision")
    parser.add_argument("--cache-dir", type=Path)


def input_options(parser):
    parser.add_argument("--input-format", choices=["auto", "text", "json", "jsonl"], default="auto")
    parser.add_argument("--text-field", default="text", help="Text field in JSON/JSONL objects")


def output_options(parser, required=False):
    parser.add_argument("--output", "-o", required=required, help="Result file; otherwise stdout")
    parser.add_argument("--force", action="store_true", help="Replace existing output files")


def parser_for_cli(backend="jacobian"):
    parser = Parser(
        prog="jpplens" if backend == "jpp" else "jlens",
        description="Unified J-lens / J++ CLI.",
        epilog="Use --legacy before the command to retain the 0.1 parameters and JSON format.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--verbose", action="store_true", help="Debug logs to stderr (before subcommand)"
    )
    parser.add_argument("--debug", action="store_true", help="Raise exceptions for debugging")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("presets", help="List model/lens presets")
    sub.add_parser("doctor", help="Check installed packages and compute availability")
    sub.add_parser("schema", help="Print CLI contract for agent/tool integration")
    if backend != "jpp":
        fit = sub.add_parser("fit", help="Fit a lens on a text/JSON/JSONL corpus")
        model_options(fit)
        input_options(fit)
        output_options(fit, required=True)
        fit.add_argument(
            "--prompts",
            required=True,
            help="Corpus path or - for stdin; text means one prompt per line",
        )
        fit.add_argument(
            "--layers",
            help="Zero-based source layers, e.g. 0,4,8 or 0:16:2; default excludes final layer",
        )
        fit.add_argument("--max-seq-len", type=positive, default=128)
        fit.add_argument("--skip-first", type=nonnegative, default=16)
        fit.add_argument("--dim-batch", type=positive, default=8)
        fit.add_argument(
            "--checkpoint", help="Resumable upstream checkpoint; distinct from the final lens"
        )
        fit.add_argument(
            "--checkpoint-every", type=nonnegative, default=1, help="0 saves only at end"
        )
        fit.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
        fit.add_argument(
            "--limit", type=positive, help="Use only the first N input records, before sharding"
        )
        fit.add_argument("--num-shards", type=positive, default=1)
        fit.add_argument("--shard-index", type=nonnegative, default=0)
        fit.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    for command in ("apply", "visualize"):
        apply = sub.add_parser(
            command,
            help="Read layer/token representations"
            if command == "apply"
            else "Export an interactive, self-contained HTML report",
        )
        model_options(apply)
        lens_options(apply)
        input_options(apply)
        output_options(apply, required=command == "visualize")
        source = apply.add_mutually_exclusive_group(required=True)
        source.add_argument("--prompt", help="One raw text prompt (no chat template added)")
        source.add_argument("--prompt-file", help="Entire UTF-8 file as one prompt; - for stdin")
        source.add_argument("--input", help="Batch corpus; model loaded once")
        apply.add_argument(
            "--layers", help="Zero-based layer indices or Python slices; default: all fitted layers"
        )
        apply.add_argument(
            "--positions",
            default="all" if command == "visualize" else "-1",
            help="Zero-based token indices, Python slices or all; negative indices allowed",
        )
        apply.add_argument("--all-positions", action="store_true", help="Alias for --positions all")
        apply.add_argument("--top-k", type=positive, default=10)
        apply.add_argument("--max-seq-len", type=positive, default=512)
        apply.add_argument(
            "--track-token-id",
            type=nonnegative,
            action="append",
            default=[],
            help="Track a vocabulary token's full-vocabulary rank; repeatable",
        )
        apply.add_argument(
            "--baseline", action="store_true", help="Also compute the vanilla logit lens"
        )
        apply.add_argument(
            "--logit-lens", action="store_true", help="Read only the untransported baseline"
        )
        apply.add_argument("--format", choices=["json", "text"], default="json")
        if backend == "jpp":
            apply.add_argument("--paper-layers", action="store_true")
            apply.add_argument("--no-readout-filter", action="store_true")
            apply.add_argument(
                "--model-identity", help="Original HF ID of an identified local model copy"
            )
        apply.add_argument(
            "--allow-model-mismatch",
            action="store_true",
            help="Explicitly override differing local model ID/revision metadata",
        )
        if command == "apply":
            apply.add_argument("--output-format", choices=["json", "jsonl"], default="json")
        else:
            apply.add_argument("--title", default="Jacobian Lens")
    inspect = sub.add_parser(
        "inspect", help="Inspect lens metadata and matrix dimensions without loading a model"
    )
    lens_options(inspect)
    inspect.add_argument("--preset")
    inspect.add_argument("--local-files-only", action="store_true")
    output_options(inspect)
    if backend != "jpp":
        merge = sub.add_parser(
            "merge", help="Combine disjoint fitted shards by prompt-count weighted mean"
        )
        merge.add_argument("lenses", nargs="+", help="Local lens .pt files (not fit checkpoints)")
        output_options(merge, required=True)
        merge.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    if backend == "jpp":
        download = sub.add_parser(
            "download", help="Download J++ checkpoint and optionally its model"
        )
        model_options(download)
        lens_options(download)
        output_options(download)
        download.add_argument("--include-model", action="store_true")
    return parser


def schema(backend="jacobian") -> dict:
    return {
        "cli_version": __version__,
        "upstream_commit": UPSTREAM_COMMIT,
        "backend": backend,
        "commands": (["download"] if backend == "jpp" else ["fit", "merge"])
        + ["apply", "visualize", "inspect", "presets", "doctor", "schema"],
        "success": {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "command": "<command>",
            "data": "<object>",
        },
        "error": {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "command": "<command or null>",
            "error": {"code": "<stable code>", "message": "<description>"},
        },
        "streams": {
            "stdout": "JSON result; JSONL with apply --output-format jsonl",
            "stderr": "logs and JSON errors",
        },
        "exit_codes": {
            "0": "success",
            "1": "runtime or missing dependency",
            "2": "invalid usage/input",
            "130": "interrupted",
        },
        "readout_kinds": ["jpp" if backend == "jpp" else "jacobian", "logit", "model"],
        "legacy": "Use --legacy before the command for the 0.1 CLI and output schema.",
        "coordinates": "0-based positions after BOS/truncation; layer output residuals",
        "probabilities": "softmax over the entire vocabulary, not renormalized top-k",
        "tracked_rank": "1 + count(logit > tracked logit); ties share a rank",
        "batch_input": {
            "format": "JSONL",
            "record": {"id": "optional string/integer", "text": "raw prompt"},
        },
        "jsonl_output": "One envelope per prompt; data.result holds the sample and run metadata",
        "help": "j-lens <command> --help",
    }


def resolve(args, backend):
    args.backend = backend
    if args.command in ("presets", "schema", "doctor", "merge"):
        return
    from . import jpp, legacy

    presets = jpp.PRESETS if backend == "jpp" else legacy.PRESETS
    name = getattr(args, "preset", None)
    if name and name not in presets:
        raise CLIError(f"Unknown preset {name!r}; run presets.")
    preset = presets.get(name, {})
    custom_model = (
        args.model is not None and args.model != preset.get("model")
        if hasattr(args, "model")
        else False
    )
    for attr, key in (
        ("model", "model"),
        ("revision", "model_revision"),
        ("lens", "lens"),
        ("lens_filename", "lens_file"),
        ("lens_revision", "lens_revision"),
    ):
        if hasattr(args, attr) and getattr(args, attr) is None:
            if attr != "revision" or not custom_model:
                setattr(args, attr, preset.get(key))
    if hasattr(args, "lens_filename"):
        args.lens_filename = args.lens_filename or "lens.pt"
        args.lens_file = args.lens_filename  # Published J++ checkpoint resolver.
    if getattr(args, "paper_layers", False):
        if not preset:
            raise CLIError("--paper-layers requires --preset.")
        if args.layers is not None:
            raise CLIError("Use either --layers or --paper-layers.")
        args.layers = ",".join(map(str, preset["paper_layers"]))
    if getattr(args, "all_positions", False):
        args.positions = "all"


def validate_args(args):
    if args.command in ("apply", "visualize"):
        if args.baseline and args.logit_lens:
            raise CLIError("Use either --baseline or --logit-lens.")
        if args.command == "apply" and args.format == "text" and args.output_format == "jsonl":
            raise CLIError("--format text cannot be combined with --output-format jsonl.")
    if args.command in ("apply", "visualize", "fit") and not args.model and args.backend != "jpp":
        raise CLIError("--model or --preset is required.", code="usage_error")
    if args.command in ("apply", "visualize", "inspect", "download") and not args.lens:
        raise CLIError("--lens or --preset is required.", code="usage_error")
    if args.command == "fit":
        if args.shard_index >= args.num_shards:
            raise CLIError("--shard-index must be smaller than --num-shards.")
        if args.max_seq_len <= args.skip_first + 1:
            raise CLIError("--max-seq-len must exceed --skip-first + 1.")
    if getattr(args, "tokenizer_revision", None) and not args.tokenizer:
        raise CLIError("--tokenizer-revision requires --tokenizer; otherwise --revision is shared.")
    if args.command in ("fit", "merge", "visualize") and args.output == "-":
        raise CLIError("This command requires a file path for --output.")
    inputs = [getattr(args, name, None) for name in ("prompts", "input", "prompt_file", "lens")]
    inputs.extend(getattr(args, "lenses", []))
    outputs = [getattr(args, "output", None), getattr(args, "checkpoint", None)]
    output_paths = [Path(path).resolve() for path in outputs if path and path != "-"]
    if args.command in ("fit", "merge"):
        output_paths += [Path(str(path) + ".meta.json") for path in output_paths]
    if len(set(output_paths)) != len(output_paths):
        raise CLIError("Output and checkpoint paths must be distinct, including metadata sidecars.")
    if set(output_paths) & {Path(path).resolve() for path in inputs if path and path != "-"}:
        raise CLIError("Output/checkpoint paths must not overwrite input files.")


def envelope(command: str, data: dict) -> dict:
    return {"schema_version": SCHEMA_VERSION, "status": "ok", "command": command, "data": data}


def dispatch(args) -> dict:
    if args.command == "schema":
        return schema(args.backend)
    if args.command == "presets":
        from . import jpp, legacy

        return jpp.PRESETS if args.backend == "jpp" else legacy.PRESETS
    if args.command == "doctor":
        from jlens_cli.runtime import doctor

        return doctor()
    from jlens_cli import operations

    if args.backend == "jpp" and args.command in ("inspect", "download"):
        return operations.jpp_artifact(args)
    if args.command == "visualize":
        from jlens_cli.report import render

        data = operations.apply(args)
        atomic_text(args.output, render(data, title=args.title))
        return {
            "output": args.output,
            "format": "html",
            "self_contained": True,
            "model": args.model,
            "num_tokens": data["results"][0]["num_tokens"],
        }
    return getattr(operations, args.command)(args)


def main(argv=None, *, backend="jacobian") -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--legacy":
        from . import jpp, legacy

        return (jpp.legacy_main if backend == "jpp" else legacy.main)(argv[1:])
    # Keep the upstream visualization and all its historical options available.
    if backend == "jacobian" and argv and argv[0] == "render":
        from .legacy import main as legacy_main

        return legacy_main(argv)
    args = None
    try:
        args = parser_for_cli(backend).parse_args(argv)
        resolve(args, backend)
        validate_args(args)
        ensure_output(getattr(args, "output", None), force=getattr(args, "force", False))
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )
        with contextlib.redirect_stdout(sys.stderr):
            result = dispatch(args)
        response = envelope(args.command, result)
        if args.command == "apply" and args.format == "text":
            text = text_readout(result)
        elif args.command == "apply" and args.output_format == "jsonl":
            metadata = {key: value for key, value in result.items() if key != "results"}
            text = "".join(
                json.dumps(
                    envelope("apply", {**metadata, "result": item}),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
                for item in result["results"]
            )
        else:
            text = json.dumps(response, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if args.command in ("apply", "inspect", "download") and args.output and args.output != "-":
            atomic_text(args.output, text)
            print(
                json.dumps(
                    envelope(
                        args.command,
                        {
                            "output": args.output,
                            "output_format": getattr(args, "output_format", "json"),
                            **(
                                {"num_prompts": len(result["results"])}
                                if "results" in result
                                else {}
                            ),
                        },
                    )
                )
            )
        else:
            sys.stdout.write(text)
        return 0
    except KeyboardInterrupt:
        error = CLIError(
            "Interrupted; a fit checkpoint can be resumed if one was configured.",
            code="interrupted",
            exit_code=130,
        )
    except Exception as exc:
        if args and args.debug:
            raise
        error = (
            exc
            if isinstance(exc, CLIError)
            else CLIError(
                str(exc),
                code="invalid_input"
                if isinstance(exc, (ValueError, FileNotFoundError))
                else "runtime_error",
                exit_code=2 if isinstance(exc, (ValueError, FileNotFoundError)) else 1,
            )
        )
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "command": args.command if args else None,
                "error": {"code": error.code, "message": str(error)},
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    return error.exit_code


def text_readout(data):
    lines = []
    for sample in data["results"]:
        lines.append(f"Sample {sample['id']}")
        for offset, position in enumerate(sample["positions"]):
            lines.append(f"Position {position}")
            for readout in sample["readouts"]:
                tokens = ", ".join(
                    f"{t['token']!r} ({t['probability']:.4f})"
                    for t in readout["positions"][offset]["top_tokens"]
                )
                lines.append(f"  {readout['kind']} / {readout['layer']}: {tokens}")
    return "\n".join(lines) + "\n"
