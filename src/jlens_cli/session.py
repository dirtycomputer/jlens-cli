"""A read-only, resident numerical session shared by single calls and workers."""

from __future__ import annotations

import math
import time

from .artifacts import file_digest
from .inputs import integer, prepare
from .io import CLIError, digest, indices
from .runtime import dependencies, lens_info, load_lens, load_model


def config_args(config, backend="jacobian"):
    from .cli import parser_for_cli, resolve, validate_args

    allowed = {
        "backend",
        "model",
        "preset",
        "revision",
        "tokenizer",
        "tokenizer_revision",
        "device",
        "device_map",
        "dtype",
        "attn_implementation",
        "local_files_only",
        "trust_remote_code",
        "no_force_bos",
        "lens",
        "lens_filename",
        "lens_revision",
        "cache_dir",
        "model_identity",
        "allow_model_mismatch",
        "max_tokens",
        "max_samples",
        "max_readout_cells",
        "max_tracked_tokens",
    }
    if not isinstance(config, dict):
        raise CLIError("Session config must be an object.")
    if set(config) - allowed:
        raise CLIError(f"Unknown configuration keys: {sorted(set(config) - allowed)}")
    kind = config.get("backend", backend)
    if kind not in ("jacobian", "jpp"):
        raise CLIError("backend must be jacobian or jpp.")
    for key in ("local_files_only", "trust_remote_code", "no_force_bos", "allow_model_mismatch"):
        if key in config and type(config[key]) is not bool:
            raise CLIError(f"{key} must be a boolean.")
    args = parser_for_cli(kind).parse_args(["apply", "--prompt", "session"])
    for key, value in config.items():
        if key not in (
            "backend",
            "max_tokens",
            "max_samples",
            "max_readout_cells",
            "max_tracked_tokens",
        ):
            setattr(args, key, value)
    args.max_seq_len = integer(config.get("max_tokens", 512), "max_tokens", 1)
    args.max_samples = integer(config.get("max_samples", 64), "max_samples", 1)
    args.max_readout_cells = integer(config.get("max_readout_cells", 512), "max_readout_cells", 1)
    args.max_tracked_tokens = integer(
        config.get("max_tracked_tokens", 128), "max_tracked_tokens", 1
    )
    resolve(args, kind)
    validate_args(args)
    return args


def capture(model, prepared, layers):
    torch, _ = dependencies()
    from jlens.hooks import ActivationRecorder

    kwargs = {
        key: torch.tensor([prepared[key]], device=model.input_device)
        for key in ("input_ids", "attention_mask", "position_ids")
        if key in prepared
    }
    with (
        torch.no_grad(),
        ActivationRecorder(model.layers, at=set(layers) | {model.n_layers - 1}) as recorder,
    ):
        model._text_module(**kwargs, use_cache=False)
        return {
            layer: value[0, prepared["positions"]].detach().float()
            for layer, value in recorder.activations.items()
        }


def forward_readouts(model, lens, prepared, layers, *, use_jacobian=True):
    torch, _ = dependencies()
    with torch.no_grad():
        residuals = capture(model, prepared, layers)
        logits = {
            layer: model.unembed(
                lens.transport(residuals[layer], layer) if use_jacobian else residuals[layer]
            )
            .float()
            .cpu()
            for layer in layers
        }
        final = model.unembed(residuals[model.n_layers - 1]).float().cpu()
    return logits, final


class LensSession:
    def __init__(self, config, *, backend="jacobian"):
        from .operations import model_identity

        started = time.perf_counter()
        self.args = args = config_args(config, backend)
        if args.backend == "jpp":
            from . import jpp

            path = jpp.lens_path(args)
            args.resolved_lens_path = str(path)
            self.lens, self.metadata = jpp.load_checkpoint(path)
            args.model = args.model or self.metadata["hf_model_name"]
            self.model = load_model(args)
            try:
                jpp.validate_model(
                    self.lens, self.metadata, self.model, args.model_identity or args.model
                )
            except ValueError as exc:
                raise CLIError(str(exc), code="incompatible_lens") from exc
        else:
            self.model = load_model(args)
            self.lens, self.metadata = load_lens(args, self.model)
        # Keep fitted matrices on their layer devices for repeated requests.
        # Offloaded meta parameters cannot be made resident in advance.
        for layer, matrix in self.lens.jacobians.items():
            parameter = next(self.model.layers[layer].parameters(), None)
            device = parameter.device if parameter is not None else self.model.input_device
            if device.type != "meta":
                self.lens.jacobians[layer] = matrix.to(device)
        identity = model_identity(args, self.model)
        self.manifest = {
            **identity,
            "backend": args.backend,
            "lens_sha256": file_digest(args.resolved_lens_path),
            "lens_info": lens_info(self.lens),
            "lens_calibration_identity": (self.metadata or {}).get("artifact_identity"),
            "upstream": __import__("jlens_cli").UPSTREAM_COMMIT,
        }
        self.manifest_id = digest(self.manifest)
        self.load_seconds = time.perf_counter() - started
        self.forward_calls = 0
        self.mask = None

    def prepared(self, sample):
        return prepare(self.model, sample, max_tokens=self.args.max_seq_len)

    def concepts(self, specs):
        if not isinstance(specs, dict):
            raise CLIError("concepts must map family names to alias/token-ID lists.")
        if (
            any(not isinstance(v, list) for v in specs.values())
            or sum(len(v) for v in specs.values()) > self.args.max_tracked_tokens * 8
        ):
            raise CLIError("Concept aliases exceed the session budget.", code="budget_exceeded")
        result = {}
        size = self.model._hf_model.get_output_embeddings().weight.shape[0]
        for name, aliases in specs.items():
            if not isinstance(name, str) or not isinstance(aliases, list) or not aliases:
                raise CLIError("Each concept family needs a nonempty alias list.")
            resolved, unsupported = [], []
            for alias in aliases:
                if type(alias) is int:
                    ids = [integer(alias, "concept token ID")]
                elif isinstance(alias, str):
                    ids = self.model.tokenizer.encode(alias, add_special_tokens=False)
                else:
                    raise CLIError("Concept aliases must be strings or token IDs.")
                if len(ids) == 1 and ids[0] < size:
                    resolved.append(ids[0])
                else:
                    unsupported.append(
                        {"alias": alias, "token_ids": ids, "reason": "requires_single_token"}
                    )
            result[name] = {
                "token_ids": sorted(set(resolved)),
                "unsupported": unsupported,
                "coverage": "partial"
                if unsupported and resolved
                else ("supported" if resolved else "unsupported"),
            }
        return result

    def readout(self, params):
        from .operations import rows

        samples = params.get("samples")
        for key in ("baseline", "readout_filter"):
            if key in params and type(params[key]) is not bool:
                raise CLIError(f"{key} must be a boolean.")
        if not isinstance(samples, list) or not samples or len(samples) > self.args.max_samples:
            raise CLIError(
                "samples must be nonempty and within max_samples.", code="budget_exceeded"
            )
        top_k = integer(params.get("top_k", 10), "top_k", 1)
        if top_k > 1000:
            raise CLIError("top_k exceeds the result budget.", code="budget_exceeded")
        kind = params.get("kind", "transported")
        if kind not in ("transported", "logit"):
            raise CLIError("kind must be transported or logit.")
        spec = params.get("layers")
        if isinstance(spec, list):
            spec = ",".join(str(integer(v, "layer")) for v in spec)
        layers = indices(spec, self.model.n_layers, default=self.lens.source_layers)
        if kind == "transported" and set(layers) - set(self.lens.source_layers):
            raise CLIError("Requested layers have no fitted Jacobian.", code="incompatible_lens")
        concepts = self.concepts(params.get("concepts", {}))
        tracked = params.get("track_token_ids", [])
        if not isinstance(tracked, list):
            raise CLIError("track_token_ids must be a list.")
        tracked = sorted(
            {integer(v, "tracked token ID") for v in tracked}
            | {v for family in concepts.values() for v in family["token_ids"]}
        )
        if len(tracked) > self.args.max_tracked_tokens:
            raise CLIError("Tracked concepts exceed max_tracked_tokens.", code="budget_exceeded")
        if not all(isinstance(s, dict) for s in samples):
            raise CLIError("Each sample must be an input object.")
        prepared = [self.prepared({"id": i, **sample}) for i, sample in enumerate(samples)]
        ids = [p["id"] for p in prepared]
        if any(not isinstance(v, (str, int)) or isinstance(v, bool) for v in ids) or len(
            set(ids)
        ) != len(ids):
            raise CLIError("Sample IDs must be unique strings/integers within a request.")
        include_baseline = bool(params.get("baseline", False)) and kind != "logit"
        cells = sum(
            len(p["positions"]) * (len(layers) * (2 if include_baseline else 1) + 1)
            for p in prepared
        )
        if cells > self.args.max_readout_cells:
            raise CLIError("Request exceeds max_readout_cells.", code="budget_exceeded")
        deadline = params.get("deadline_seconds")
        if deadline is not None and (
            type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0
        ):
            raise CLIError("deadline_seconds must be positive.")
        started, calls, results = time.perf_counter(), self.forward_calls, []
        torch, _ = dependencies()
        filtered = (
            self.args.backend == "jpp" and kind != "logit" and params.get("readout_filter", True)
        )
        if filtered and self.mask is None:
            from .jpp import readout_mask

            self.mask = readout_mask(self.model.tokenizer, self.model._lm_head.weight.shape[0])
        for sample in prepared:
            if deadline is not None and time.perf_counter() - started > deadline:
                raise CLIError(
                    "Deadline reached between samples; no partial result published.",
                    code="budget_exceeded",
                )
            with torch.no_grad():
                residuals = capture(self.model, sample, layers)
                self.forward_calls += 1
                readouts = []
                for readout_kind in (
                    ["logit"]
                    if kind == "logit"
                    else [self.args.backend] + (["logit"] if include_baseline else [])
                ):
                    for layer in layers:
                        value = residuals[layer]
                        if readout_kind != "logit":
                            value = self.lens.transport(value, layer)
                        logits = self.model.unembed(value).float().cpu()
                        readouts.append(
                            {
                                "kind": readout_kind,
                                "layer": layer,
                                "positions": rows(
                                    logits,
                                    sample["positions"],
                                    sample["input_ids"],
                                    self.model.tokenizer,
                                    top_k,
                                    tracked,
                                    self.mask if filtered and readout_kind != "logit" else None,
                                ),
                            }
                        )
                final = self.model.unembed(residuals[self.model.n_layers - 1]).float().cpu()
                readouts.append(
                    {
                        "kind": "model",
                        "layer": self.model.n_layers - 1,
                        "positions": rows(
                            final,
                            sample["positions"],
                            sample["input_ids"],
                            self.model.tokenizer,
                            top_k,
                            tracked,
                        ),
                    }
                )
            results.append({**sample, "readouts": readouts})
        result = {
            "manifest_id": self.manifest_id,
            "manifest": self.manifest,
            "concepts": concepts,
            "analysis_spec": {
                "layers": layers,
                "kind": kind,
                "baseline": include_baseline,
                "top_k": top_k,
                "track_token_ids": tracked,
                "readout_filter": bool(filtered),
            },
            "probability_basis": "full-vocabulary",
            "results": results,
            "metrics": {
                "elapsed_seconds": time.perf_counter() - started,
                "forward_calls": self.forward_calls - calls,
                "session_load_seconds": self.load_seconds,
                "tokens": sum(p["num_tokens"] for p in prepared),
            },
        }
        result["artifact_id"] = digest(
            {key: value for key, value in result.items() if key != "metrics"}
        )
        return result
