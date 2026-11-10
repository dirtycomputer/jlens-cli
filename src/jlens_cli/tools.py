"""Versioned local tool protocol. No training, network server, or agent framework."""

from __future__ import annotations

import contextlib
import json
import sys
import time
from copy import copy

from . import __version__, analysis
from .contracts import REQUIRED, input_schema
from .io import CLIError, atomic_text, ensure_output, read_text

METHODS = {
    "capabilities": {
        "params": [],
        "description": "Discover operations without importing the model runtime.",
    },
    "input.prepare": {
        "params": ["sample"],
        "description": "Resolve exact rollout IDs, source positions and explicit truncation.",
    },
    "concepts.resolve": {
        "params": ["concepts"],
        "description": "Resolve single-token aliases and report unsupported multi-token concepts.",
    },
    "readout": {
        "params": [
            "samples",
            "layers",
            "kind",
            "top_k",
            "baseline",
            "track_token_ids",
            "concepts",
            "readout_filter",
            "deadline_seconds",
        ],
        "description": "One forward per sample; reuse activations for all readout kinds.",
    },
    "concepts.trajectory": {
        "params": ["observation", "rank_threshold"],
        "description": "Describe sampled concept trajectories; no causal classification.",
    },
    "components.list": {
        "params": [],
        "description": "List residual-output projections and LoRA parameter costs.",
    },
    "components.score": {
        "params": [
            "modules",
            "failure_samples",
            "reference_samples",
            "concepts",
            "wrong_weight",
            "background_weight",
        ],
        "description": "Contrast failure and background residual-write projections.",
    },
    "checkpoint.compare": {
        "params": ["before", "after", "allow_lens_change"],
        "description": "Compare fixed replay with input, tokenizer and coordinate checks.",
    },
    "audit.scan": {
        "params": ["observation", "rank_threshold"],
        "description": "Report held-out concept rates; no release verdict.",
    },
    "lens.health": {
        "params": [
            "reference",
            "candidate",
            "min_overlap",
            "min_cells",
            "observation",
            "canaries",
            "rank_threshold",
            "min_pass_rate",
            "min_samples",
        ],
        "description": "Check labeled canaries or independent reference agreement.",
    },
    "trials.propose": {
        "params": [
            "component_report",
            "allowed_modules",
            "algorithm",
            "training_data_id",
            "rank",
            "max_trainable_parameters",
            "seed",
            "loss_scope",
            "loss_positions",
        ],
        "description": "Propose bounded PEFT trial/control arms; never execute a trial.",
    },
    "close": {
        "params": [],
        "description": "Release the resident session and stop the JSONL worker.",
    },
}
MODEL_METHODS = {
    "input.prepare",
    "concepts.resolve",
    "readout",
    "components.list",
    "components.score",
}


def strict_json(text):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant {value} is not supported.")

    return json.loads(text, parse_constant=invalid)


def capabilities(backend="jacobian"):
    return {
        "protocol": "jlens-tools/1",
        "cli_version": __version__,
        "backend": backend,
        "methods": {
            name: {**spec, "input_schema": input_schema(name, spec["params"])}
            for name, spec in METHODS.items()
        },
        "unsupported": [
            "J++ fitting",
            "causal interventions",
            "parameter gradients",
            "head/element masks",
            "training/promotion/rollback",
            "multi-model concurrent serving",
        ],
        "semantics": {
            "ranks": "full vocabulary; J++ filtered_rank is separate",
            "probabilities": "full-vocabulary softmax",
            "positions": "zero-based source residual positions; final logits at p predict p+1",
            "deadline": "cooperative between samples, not preemptive GPU cancellation",
            "frequency": "caller-controlled event triggers; no hard-coded hourly update schedule",
        },
    }


class LensTool:
    """Python entry point; one object owns one immutable effective checkpoint."""

    def __init__(self, config=None, *, backend="jacobian"):
        self.config = config
        self.args = None
        if config is not None:
            from .session import config_args

            args = config_args(config, backend)
            self.args = args
            backend = args.backend
        self.backend = backend
        self.session = None
        self.closed = False

    def handle(self, request):
        started = time.perf_counter()
        request_id, method, loaded = None, None, False
        try:
            if not isinstance(request, dict) or set(request) - {
                "id",
                "method",
                "params",
                "manifest_id",
            }:
                raise CLIError(
                    "Request must contain id, method, params and optionally manifest_id."
                )
            request_id, method = request.get("id"), request.get("method")
            if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
                raise CLIError("id must be a string or integer.")
            if self.closed:
                raise CLIError("The session is closed.", code="session_closed")
            if not isinstance(method, str) or method not in METHODS:
                raise CLIError(
                    "Unsupported tool method; call capabilities.", code="unsupported_method"
                )
            params = request.get("params", {})
            if not isinstance(params, dict) or set(params) - set(METHODS[method]["params"]):
                raise CLIError(f"Invalid parameters for {method}.")
            if not set(REQUIRED.get(method, [])) <= params.keys():
                raise CLIError(f"Missing required parameters for {method}.")
            if method == "readout" and self.args is not None:
                samples = params["samples"]
                if (
                    not isinstance(samples, list)
                    or not samples
                    or len(samples) > self.args.max_samples
                ):
                    raise CLIError("samples exceed the configured budget.", code="budget_exceeded")
            with contextlib.redirect_stdout(sys.stderr):
                if method in MODEL_METHODS:
                    if self.session is None:
                        from .session import LensSession

                        if self.config is None:
                            raise CLIError("This method requires a session config.")
                        self.session = LensSession(self.config, backend=self.backend)
                        loaded = True
                    if (
                        request.get("manifest_id")
                        and request["manifest_id"] != self.session.manifest_id
                    ):
                        raise CLIError(
                            "Request targets a different immutable session manifest.",
                            code="stale_manifest",
                        )
                if method == "capabilities":
                    data = {
                        **capabilities(self.backend),
                        "configured": self.config is not None,
                        "loaded": self.session is not None,
                        "manifest_id": self.session.manifest_id if self.session else None,
                        "limits": {
                            name: getattr(self.args, name)
                            for name in (
                                "max_seq_len",
                                "max_samples",
                                "max_readout_cells",
                                "max_tracked_tokens",
                            )
                        }
                        if self.args
                        else None,
                    }
                elif method == "input.prepare":
                    data = {
                        "manifest_id": self.session.manifest_id,
                        **self.session.prepared(params.get("sample")),
                    }
                elif method == "concepts.resolve":
                    data = {
                        "manifest_id": self.session.manifest_id,
                        "concepts": self.session.concepts(params.get("concepts")),
                    }
                elif method == "readout":
                    data = self.session.readout(params)
                elif method.startswith("components."):
                    from .components import catalog, score

                    data = (
                        catalog(self.session)
                        if method == "components.list"
                        else score(self.session, params)
                    )
                elif method == "concepts.trajectory":
                    data = analysis.trajectory(params)
                elif method == "checkpoint.compare":
                    data = analysis.compare(params)
                elif method == "audit.scan":
                    data = analysis.audit(params)
                elif method == "lens.health":
                    data = analysis.health(params)
                elif method == "trials.propose":
                    data = analysis.propose(params)
                else:
                    self.session = None
                    self.closed = True
                    data = {"closed": True}
            return {
                "protocol": "jlens-tools/1",
                "id": request_id,
                "method": method,
                "status": "ok",
                "data": data,
                "metrics": {
                    "wall_seconds": time.perf_counter() - started,
                    "model_loaded_this_request": loaded,
                },
            }
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, CLIError)
                else CLIError(str(exc), code="runtime_error", exit_code=1)
            )
            return {
                "protocol": "jlens-tools/1",
                "id": request_id,
                "method": method,
                "status": "error",
                "error": {"code": error.code, "message": str(error), "exit_code": error.exit_code},
                "metrics": {
                    "wall_seconds": time.perf_counter() - started,
                    "model_loaded_this_request": loaded,
                },
            }


def emit(value, stream):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def serve(tool, stream, output, *, max_request_bytes=16 * 1024 * 1024):
    while not tool.closed:
        line = stream.readline(max_request_bytes + 1)
        if not line:
            break
        try:
            if len(line.encode("utf-8") if isinstance(line, str) else line) > max_request_bytes:
                while line and not line.endswith("\n" if isinstance(line, str) else b"\n"):
                    line = stream.readline(max_request_bytes + 1)
                raise CLIError("Request exceeds max_request_bytes.", code="budget_exceeded")
            request = strict_json(line)
            response = tool.handle(request)
        except (CLIError, ValueError, UnicodeError) as exc:
            response = {
                "protocol": "jlens-tools/1",
                "id": None,
                "status": "error",
                "error": {
                    "code": exc.code if isinstance(exc, CLIError) else "invalid_json",
                    "message": str(exc),
                    "exit_code": 2,
                },
            }
        emit(response, output)
    tool.session = None
    return 0


def run_cli(args):
    from .artifacts import protect_paths

    config = strict_json(read_text(args.config)) if args.config else None
    tool = LensTool(config, backend=args.backend)
    if args.command == "serve":
        return serve(
            tool,
            getattr(sys.stdin, "buffer", sys.stdin),
            sys.stdout,
            max_request_bytes=args.max_request_bytes,
        )
    protect_paths(args, [p for p in (args.config, args.request) if p and p != "-"])
    if tool.args is not None:
        sources = copy(tool.args)
        sources.output = args.output
        protect_paths(sources)
    ensure_output(args.output, force=args.force)
    response = tool.handle(strict_json(read_text(args.request)))
    if tool.session is not None:
        sources = copy(tool.session.args)
        sources.output = args.output
        protect_paths(sources, [sources.resolved_lens_path])
    text = json.dumps(response, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output and args.output != "-":
        if response["status"] == "ok":
            atomic_text(args.output, text)
        else:
            emit(response, sys.stderr)
    else:
        sys.stdout.write(text)
    return 0 if response["status"] == "ok" else response["error"]["exit_code"]
