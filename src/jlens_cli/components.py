"""Contrastive residual-write projections; these scores are not causal attribution."""

from __future__ import annotations

import math
import statistics
import time

from .io import CLIError
from .runtime import dependencies
from .session import capture


def catalog(session):
    model = session.model
    root = model.layout.path + "." + model.layout.layers + "."
    result = []
    for name, module in model._hf_model.named_modules():
        if not name.startswith(root) or name.rsplit(".", 1)[-1] not in (
            "o_proj",
            "down_proj",
            "c_proj",
        ):
            continue
        layer = int(name[len(root) :].split(".", 1)[0])
        weight = getattr(module, "weight", None)
        if layer not in session.lens.source_layers or weight is None or weight.ndim != 2:
            continue
        if hasattr(module, "in_features"):
            width_in, width_out = module.in_features, module.out_features
        elif type(module).__name__ == "Conv1D":
            width_in, width_out = weight.shape
        else:
            continue
        if width_out != model.d_model:
            continue
        result.append(
            {
                "name": name,
                "layer": layer,
                "in_features": int(width_in),
                "out_features": int(width_out),
                "lora_params_per_rank": int(width_in + width_out),
                "kind": "mlp" if ".mlp." in name else "attention_output",
            }
        )
    return {
        "manifest_id": session.manifest_id,
        "modules": result,
        "scope": "whole residual-output projections; no Q/K/V, head or element attribution",
        "evidence_type": "associational_linear_projection",
    }


def score(session, params):
    known = {item["name"]: item for item in catalog(session)["modules"]}
    names = params.get("modules")
    if (
        not isinstance(names, list)
        or not names
        or len(set(names)) != len(names)
        or any(name not in known for name in names)
    ):
        raise CLIError("modules must be unique names from components.list.")
    groups = []
    for key in ("failure_samples", "reference_samples"):
        samples = params.get(key)
        if (
            not isinstance(samples, list)
            or not samples
            or not all(isinstance(s, dict) for s in samples)
        ):
            raise CLIError(f"{key} must contain input samples.")
        groups.append([session.prepared({"id": i, **sample}) for i, sample in enumerate(samples)])
    if (
        sum(map(len, groups)) > session.args.max_samples
        or sum(len(s["positions"]) for group in groups for s in group) * len(names)
        > session.args.max_readout_cells
    ):
        raise CLIError("Component request exceeds the session budget.", code="budget_exceeded")
    resolved = session.concepts(params.get("concepts", {}))
    need = resolved.get("need", {}).get("token_ids", [])
    wrong = resolved.get("wrong", {}).get("token_ids", [])
    if (
        not need
        or resolved.get("need", {}).get("unsupported")
        or resolved.get("wrong", {}).get("unsupported")
    ):
        raise CLIError("Scoring requires fully supported single-token need/wrong concepts.")
    beta = params.get("wrong_weight", 1.0)
    background_weight = params.get("background_weight", 1.0)
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0
        for v in (beta, background_weight)
    ):
        raise CLIError("Projection weights must be finite and nonnegative.")
    torch, _ = dependencies()
    model = session.model
    head = model._lm_head.weight
    with torch.no_grad():
        direction = head[need].float().mean(0)
        if wrong:
            direction = direction - beta * head[wrong].float().mean(0)
        directions = {}
        for name in names:
            layer = known[name]["layer"]
            matrix = session.lens.jacobians[layer]
            vector = direction.to(matrix.device) @ matrix
            norm = vector.norm()
            if norm.item() == 0:
                raise CLIError("Concept direction has zero norm; no projection can be scored.")
            directions[name] = vector / norm
    modules = dict(model._hf_model.named_modules())
    measurements = [{name: [] for name in names} for _ in groups]
    started, initial_calls = time.perf_counter(), session.forward_calls
    for group_index, samples in enumerate(groups):
        for sample in samples:
            values, handles = {}, []

            def hook(name, sample=sample, values=values):
                def record(_module, _inputs, output):
                    output = output if torch.is_tensor(output) else output[0]
                    if output.shape[-1] != model.d_model:
                        raise CLIError("Component output is not in the residual basis.")
                    value = output[0, sample["positions"]].detach().float()
                    values[name] = (value @ directions[name].to(value.device)).mean().item()

                return record

            try:
                for name in names:
                    handles.append(modules[name].register_forward_hook(hook(name)))
                capture(model, sample, [])
                session.forward_calls += 1
            finally:
                for handle in handles:
                    handle.remove()
            if set(values) != set(names) or any(not math.isfinite(v) for v in values.values()):
                raise CLIError("Selected components were not called or produced invalid values.")
            for name in names:
                measurements[group_index][name].append(values[name])
    ranked = []
    for name in names:
        failure, reference = [group[name] for group in measurements]
        uncertainty = (
            math.sqrt(
                statistics.variance(failure) / len(failure)
                + background_weight**2 * statistics.variance(reference) / len(reference)
            )
            if len(failure) > 1 and len(reference) > 1
            else None
        )
        ranked.append(
            {
                **known[name],
                "score": statistics.mean(failure) - background_weight * statistics.mean(reference),
                "failure_mean": statistics.mean(failure),
                "reference_mean": statistics.mean(reference),
                "standard_error": uncertainty,
                "failure_samples": len(failure),
                "reference_samples": len(reference),
            }
        )
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return {
        "manifest_id": session.manifest_id,
        "modules": ranked,
        "concepts": resolved,
        "evidence_type": "associational_linear_projection",
        "formula": "failure_mean_projection - background_weight * reference_mean_projection",
        "wrong_weight": beta,
        "background_weight": background_weight,
        "limitations": [
            "No causal or optimal-edit-location claim",
            "Final norm, residual scaling and downstream compensation are not modeled",
            "Held-out behavioral trials must validate any candidate",
        ],
        "metrics": {
            "forward_calls": session.forward_calls - initial_calls,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
