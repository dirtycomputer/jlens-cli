"""Public command-line interface. Stdout is reserved for results."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

from jlens_cli import SCHEMA_VERSION, UPSTREAM_COMMIT, __version__
from jlens_cli.io import CLIError, atomic_text


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
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local directory")
    parser.add_argument("--revision", help="Model revision; prefer an immutable commit")
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
    parser.add_argument(
        "--lens", required=True, help="Local .pt file, directory or Hugging Face lens repo"
    )
    parser.add_argument("--lens-filename", default="lens.pt")
    parser.add_argument("--lens-revision")


def input_options(parser):
    parser.add_argument("--input-format", choices=["auto", "text", "json", "jsonl"], default="auto")
    parser.add_argument("--text-field", default="text", help="Text field in JSON/JSONL objects")


def output_options(parser, required=False):
    parser.add_argument("--output", "-o", required=required, help="Result file; otherwise stdout")
    parser.add_argument("--force", action="store_true", help="Replace existing output files")


def parser_for_cli():
    parser = Parser(prog="j-lens", description="Agent-friendly CLI for Anthropic's Jacobian Lens.")
    parser.add_argument("--version", action="version", version=f"j-lens-cli {__version__}")
    parser.add_argument(
        "--verbose", action="store_true", help="Debug logs to stderr (before subcommand)"
    )
    parser.add_argument("--debug", action="store_true", help="Raise exceptions for debugging")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check installed packages and compute availability")
    sub.add_parser("schema", help="Print CLI contract for agent/tool integration")
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
    fit.add_argument("--checkpoint-every", type=nonnegative, default=1, help="0 saves only at end")
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
    merge = sub.add_parser(
        "merge", help="Combine disjoint fitted shards by prompt-count weighted mean"
    )
    merge.add_argument("lenses", nargs="+", help="Local lens .pt files (not fit checkpoints)")
    output_options(merge, required=True)
    merge.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    return parser


def schema() -> dict:
    return {
        "cli_version": __version__,
        "upstream_commit": UPSTREAM_COMMIT,
        "commands": ["fit", "apply", "visualize", "inspect", "merge", "doctor", "schema"],
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
        "readout_kinds": ["jacobian", "logit", "model"],
        "coordinates": "0-based source positions after tokenization/BOS/truncation; layer output residuals",
        "probabilities": "softmax over the entire vocabulary, not renormalized top-k",
        "tracked_rank": "1 + count(logit > tracked logit); ties share a rank",
        "batch_input": {
            "format": "JSONL",
            "record": {"id": "optional string/integer", "text": "raw prompt"},
        },
        "jsonl_output": "One success envelope per prompt; data.result holds one sample, other data fields carry run metadata",
        "help": "j-lens <command> --help",
    }


def validate_args(args):
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
        return schema()
    if args.command == "doctor":
        from jlens_cli.runtime import doctor

        return doctor()
    from jlens_cli import operations

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


def main(argv=None) -> int:
    args = None
    try:
        args = parser_for_cli().parse_args(argv)
        validate_args(args)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )
        with contextlib.redirect_stdout(sys.stderr):
            result = dispatch(args)
        response = envelope(args.command, result)
        if args.command == "apply" and args.output_format == "jsonl":
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
        if args.command == "apply" and args.output and args.output != "-":
            atomic_text(args.output, text)
            print(
                json.dumps(
                    envelope(
                        "apply",
                        {
                            "output": args.output,
                            "output_format": args.output_format,
                            "num_prompts": len(result["results"]),
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
