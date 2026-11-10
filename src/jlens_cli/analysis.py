"""CPU-only analysis of saved observations and bounded trial proposals."""

from __future__ import annotations

import math
import random

from .inputs import integer
from .io import CLIError, digest


def artifact(value):
    if not isinstance(value, dict):
        raise CLIError("An observation artifact must be an object.")
    if value.get("status") == "ok" and "data" in value:
        value = value["data"]
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("results"), list)
        or not value.get("manifest")
    ):
        raise CLIError("Expected a resident-tool readout artifact with manifest and results.")
    if value.get("manifest_id") != digest(value["manifest"]):
        raise CLIError(
            "Observation manifest content does not match its ID.", code="incompatible_artifacts"
        )
    if value.get("artifact_id") != digest(
        {k: v for k, v in value.items() if k not in ("metrics", "artifact_id")}
    ):
        raise CLIError(
            "Observation content does not match its artifact ID.", code="incompatible_artifacts"
        )
    return value


def cells(data):
    result = {}
    for sample in data["results"]:
        for row in sample["readouts"]:
            for cell in row["positions"]:
                key = (sample["id"], row["kind"], row["layer"], cell["position"])
                if key in result:
                    raise CLIError("Artifact has duplicate sample/layer/position coordinates.")
                result[key] = cell
    return result


def trajectory(params):
    data = artifact(params.get("observation"))
    threshold = integer(params.get("rank_threshold", 25), "rank_threshold", 1)
    families = data.get("concepts", {})
    result = []
    for sample in data["results"]:
        for position in sample["positions"]:
            for name, family in families.items():
                tokens = set(family["token_ids"])
                curve = []
                for row in sample["readouts"]:
                    if row["kind"] in ("logit", "model"):
                        continue
                    cell = next(c for c in row["positions"] if c["position"] == position)
                    ranks = {v["token_id"]: v["rank"] for v in cell["tracked_tokens"]}
                    rank = (
                        min((ranks[t] for t in tokens), default=None)
                        if tokens <= ranks.keys()
                        else None
                    )
                    curve.append(
                        {
                            "layer": row["layer"],
                            "rank": rank,
                            "observed": rank <= threshold if rank is not None else None,
                        }
                    )
                curve.sort(key=lambda point: point["layer"])
                active = [i for i, point in enumerate(curve) if point["observed"]]
                complete = (
                    bool(curve)
                    and bool(tokens)
                    and all(p["rank"] is not None for p in curve)
                    and not family.get("unsupported")
                )
                if not complete:
                    pattern = "unknown"
                elif not active:
                    pattern = "not_observed_at_sampled_layers"
                elif any(not point["observed"] for point in curve[active[0] + 1 :]):
                    pattern = "appeared_then_dropped_at_sampled_layers"
                else:
                    pattern = "present_through_last_sampled_layer"
                result.append(
                    {
                        "sample_id": sample["id"],
                        "position": position,
                        "family": name,
                        "curve": curve,
                        "pattern": pattern,
                        "first_observed_layer": curve[active[0]]["layer"] if active else None,
                        "coverage": "complete" if complete else "partial_or_unsupported",
                    }
                )
    return {
        "source_artifact_id": data.get("artifact_id"),
        "rank_threshold": threshold,
        "trajectories": result,
        "interpretation": "descriptive patterns; no proof of use, suppression or causality",
    }


def compatible(before, after, *, allow_lens_change=False):
    a, b = before["manifest"], after["manifest"]
    if a["artifact_identity"]["tokenizer_sha256"] != b["artifact_identity"]["tokenizer_sha256"]:
        raise CLIError(
            "Artifacts use different tokenizers; token ranks are not comparable.",
            code="incompatible_artifacts",
        )
    if a["backend"] != b["backend"] or before["analysis_spec"] != after["analysis_spec"]:
        raise CLIError(
            "Artifacts use different backends or analysis specifications.",
            code="incompatible_artifacts",
        )
    if a["lens_sha256"] != b["lens_sha256"] and not allow_lens_change:
        raise CLIError(
            "Lens coordinates changed; explicitly allow explanatory comparison.",
            code="incompatible_artifacts",
        )


def compare(params):
    if "allow_lens_change" in params and type(params["allow_lens_change"]) is not bool:
        raise CLIError("allow_lens_change must be a boolean.")
    before, after = artifact(params.get("before")), artifact(params.get("after"))
    compatible(before, after, allow_lens_change=params.get("allow_lens_change", False))
    index_a, index_b = cells(before), cells(after)
    hashes_a = {
        sample["id"]: (sample["input_hash"], sample["original_positions"])
        for sample in before["results"]
    }
    hashes_b = {
        sample["id"]: (sample["input_hash"], sample["original_positions"])
        for sample in after["results"]
    }
    if hashes_a != hashes_b or index_a.keys() != index_b.keys():
        raise CLIError(
            "Fixed replay requires identical sample IDs, tokens, masks and source positions.",
            code="incompatible_artifacts",
        )
    changes, agreement, transported = [], [], []
    for key, a in index_a.items():
        b = index_b[key]
        left = {v["token_id"] for v in a["top_tokens"]}
        right = {v["token_id"] for v in b["top_tokens"]}
        agreement.append(len(left & right) / max(1, len(left)))
        if key[1] not in ("logit", "model"):
            transported.append(agreement[-1])
        tracked_a = {v["token_id"]: v for v in a["tracked_tokens"]}
        tracked_b = {v["token_id"]: v for v in b["tracked_tokens"]}
        if tracked_a.keys() != tracked_b.keys():
            raise CLIError("Tracked token sets differ.", code="incompatible_artifacts")
        for token, value in tracked_a.items():
            changes.append(
                {
                    "sample_id": key[0],
                    "kind": key[1],
                    "layer": key[2],
                    "position": key[3],
                    "token_id": token,
                    "rank_before": value["rank"],
                    "rank_after": tracked_b[token]["rank"],
                    "rank_delta": tracked_b[token]["rank"] - value["rank"],
                    "probability_delta": tracked_b[token]["probability"] - value["probability"],
                }
            )
    return {
        "mode": "fixed_replay",
        "before_artifact_id": before.get("artifact_id"),
        "after_artifact_id": after.get("artifact_id"),
        "lens_changed": before["manifest"]["lens_sha256"] != after["manifest"]["lens_sha256"],
        "mean_top_k_overlap": sum(agreement) / len(agreement) if agreement else None,
        "transported_top_k_overlap": sum(transported) / len(transported) if transported else None,
        "transported_cells": len(transported),
        "cells": len(agreement),
        "tracked_changes": changes,
        "interpretation": "paired observation differences, not a release decision",
    }


def audit(params):
    data = artifact(params.get("observation"))
    trajectories = trajectory(params)["trajectories"]
    signals = []
    for name in data.get("concepts", {}):
        sample_states = {}
        for row in trajectories:
            if row["family"] == name:
                sample_states.setdefault(row["sample_id"], []).append(row)
        supported = [
            rows
            for rows in sample_states.values()
            if all(row["coverage"] == "complete" for row in rows)
        ]
        flagged = [
            rows[0]["sample_id"]
            for rows in supported
            if any(point["observed"] for row in rows for point in row["curve"])
        ]
        signals.append(
            {
                "family": name,
                "supported_samples": len(supported),
                "total_samples": len(data["results"]),
                "occurrence_rate": len(flagged) / len(supported) if supported else None,
                "evidence_sample_ids": flagged,
            }
        )
    return {
        "source_artifact_id": data.get("artifact_id"),
        "signals": signals,
        "decision": "external_behavioral_gate_required",
        "purpose": "held-out observation, not a training reward",
    }


def health(params):
    if "observation" in params:
        if "reference" in params or "candidate" in params:
            raise CLIError("Choose either canaries or reference-lens comparison.")
        return canary_health(params)
    reference, candidate = artifact(params.get("reference")), artifact(params.get("candidate"))
    result = compare({"before": reference, "after": candidate, "allow_lens_change": True})
    a, b = reference["manifest"], candidate["manifest"]
    reasons = []
    if a["artifact_identity"] != b["artifact_identity"]:
        reasons.append(
            "reference must be fitted and evaluated on the same effective weights/tokenizer"
        )
    if (
        a["lens_sha256"] == b["lens_sha256"]
        or a.get("lens_calibration_identity") != a["artifact_identity"]
    ):
        reasons.append("an independently fitted, content-bound reference lens is required")
    threshold, minimum = params.get("min_overlap"), params.get("min_cells")
    if (
        type(threshold) not in (float, int)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
        or type(minimum) is not int
        or minimum < 1
    ):
        raise CLIError(
            "Set explicit min_overlap in [0,1] and min_cells >= 1; calibrate these on your probes."
        )
    if result["transported_cells"] < minimum:
        reasons.append("insufficient probe coverage")
    status = (
        "unknown"
        if reasons
        else ("consistent" if result["transported_top_k_overlap"] >= threshold else "degraded")
    )
    return {
        "status": status,
        "reasons": reasons,
        "mean_top_k_overlap": result["transported_top_k_overlap"],
        "cells": result["transported_cells"],
        "recommendation": "evaluate_or_refit"
        if status != "consistent"
        else "continue_independent_behavioral_audit",
        "scope": "readout agreement only; no safety or deployment certification",
    }


def canary_health(params):
    data = artifact(params["observation"])
    probes = params.get("canaries")
    threshold = integer(params.get("rank_threshold", 25), "rank_threshold", 1)
    minimum = integer(params.get("min_samples"), "min_samples", 1)
    target = params.get("min_pass_rate")
    if type(target) not in (int, float) or not math.isfinite(target) or not 0 <= target <= 1:
        raise CLIError("Set an explicit min_pass_rate in [0,1].")
    if not isinstance(probes, list) or not probes:
        raise CLIError("canaries must contain labeled sample_id/expected_token_ids probes.")
    samples = {s["id"]: s for s in data["results"]}
    results, seen = [], set()
    for probe in probes:
        if (
            not isinstance(probe, dict)
            or not isinstance(probe.get("sample_id"), (str, int))
            or isinstance(probe.get("sample_id"), bool)
        ):
            raise CLIError("Each canary needs a sample_id.")
        sample_id = probe["sample_id"]
        if sample_id in seen or sample_id not in samples:
            raise CLIError("Canary sample IDs must be unique and present in the artifact.")
        seen.add(sample_id)
        tokens = probe.get("expected_token_ids")
        if not isinstance(tokens, list) or not tokens:
            raise CLIError("expected_token_ids must be single-token alternatives.")
        tokens = {integer(t, "canary token ID") for t in tokens}
        ranks, complete = [], True
        for row in samples[sample_id]["readouts"]:
            if row["kind"] in ("model", "logit"):
                continue
            for cell in row["positions"]:
                tracked = {t["token_id"]: t["rank"] for t in cell["tracked_tokens"]}
                complete = complete and tokens <= tracked.keys()
                ranks.extend(tracked[t] for t in tokens if t in tracked)
        best = min(ranks) if ranks else None
        results.append(
            {
                "sample_id": sample_id,
                "best_rank": best,
                "supported": complete and bool(ranks),
                "passed": best <= threshold if best is not None and complete else None,
            }
        )
    supported = [p for p in results if p["supported"]]
    rate = sum(p["passed"] for p in supported) / len(supported) if supported else None
    reasons = []
    if len(supported) != len(results):
        reasons.append("some expected tokens or transported readouts were not measured")
    if len(supported) < minimum:
        reasons.append("insufficient labeled probe coverage")
    status = "unknown" if reasons else ("consistent" if rate >= target else "degraded")
    return {
        "mode": "labeled_canary",
        "status": status,
        "reasons": reasons,
        "pass_rate": rate,
        "supported_samples": len(supported),
        "canaries": results,
        "scope": "caller-supplied held-out probes only; not a behavioral or release verdict",
        "recommendation": "evaluate_or_refit"
        if status != "consistent"
        else "continue_independent_behavioral_audit",
    }


def propose(params):
    report = params.get("component_report")
    if (
        not isinstance(report, dict)
        or not report.get("manifest_id")
        or not isinstance(report.get("modules"), list)
    ):
        raise CLIError("A component report with manifest_id and modules is required.")
    allowed = params.get("allowed_modules")
    if not isinstance(allowed, list) or not allowed or not all(isinstance(n, str) for n in allowed):
        raise CLIError("Supply an explicit allowed_modules whitelist.")
    rank = integer(params.get("rank", 8), "rank", 1)
    budget = integer(params.get("max_trainable_parameters"), "max_trainable_parameters", 1)
    algorithm = params.get("algorithm")
    if (
        algorithm not in ("sft", "grpo", "crt")
        or not isinstance(params.get("training_data_id"), str)
        or not params["training_data_id"]
    ):
        raise CLIError("Specify algorithm (sft/grpo/crt) and an immutable training_data_id.")
    scope = params.get("loss_scope", "reflection" if algorithm == "crt" else "response")
    positions = params.get("loss_positions", {})
    if scope not in ("response", "reflection", "selected_positions") or not isinstance(
        positions, dict
    ):
        raise CLIError("Invalid loss scope/positions.")
    if scope == "selected_positions":
        if not positions:
            raise CLIError("selected_positions requires loss_positions per sample ID.")
        for sample, selected in positions.items():
            if not isinstance(sample, str) or not isinstance(selected, list) or not selected:
                raise CLIError("Map training sample IDs to nonempty loss token-index lists.")
            for p in selected:
                integer(p, "loss token position")
    elif positions:
        raise CLIError("Explicit loss_positions require selected_positions scope.")
    modules = [row for row in report["modules"] if row["name"] in allowed]
    if not modules or len({row["name"] for row in modules}) != len(modules):
        raise CLIError("No unique scored modules match the whitelist.")
    for row in modules:
        integer(row["lora_params_per_rank"], "module parameter count", 1)
        if not isinstance(row.get("score"), (int, float)) or not math.isfinite(row["score"]):
            raise CLIError("Module scores must be finite.")
    ranked = sorted(modules, key=lambda row: row["score"], reverse=True)
    shuffled = list(modules)
    random.Random(integer(params.get("seed", 0), "seed")).shuffle(shuffled)
    late = sorted(
        (row for row in modules if row["kind"] == "mlp"), key=lambda row: row["layer"], reverse=True
    )
    candidates, selections = [], set()
    for method, ordered in (
        ("contrastive_projection", ranked),
        ("random_budgeted", shuffled),
        ("late_mlp", late),
    ):
        selected, total = [], 0
        for row in ordered:
            size = rank * row["lora_params_per_rank"]
            if total + size <= budget:
                total += size
                selected.append(row["name"])
        key = tuple(sorted(selected))
        if not selected or key in selections:
            continue
        selections.add(key)
        candidates.append(
            {
                "selector": method,
                "manifest_id": report["manifest_id"],
                "training_data_id": params["training_data_id"],
                "algorithm": algorithm,
                "supervision": {
                    "scope": scope,
                    "loss_positions": positions,
                    "coordinates": "original training tokens; trainer validates masks and bounds",
                },
                "estimated_trainable_parameters": total,
                "peft_config": {
                    "r": rank,
                    "lora_alpha": 2 * rank,
                    "bias": "none",
                    "target_modules": selected,
                },
                "requires": [
                    "pilot_training",
                    "heldout_behavior",
                    "retention_evaluation",
                    "lens_audit",
                ],
                "state": "unvalidated_trial_candidate",
            }
        )
    return {
        "proposal_id": digest(candidates),
        "candidates": candidates,
        "budget": budget,
        "limitations": [
            "Budgeted arms are not guaranteed to have identical parameter counts",
            "Gradient-probe and all-linear baselines must come from the external trainer",
            "No candidate is selected, trained, promoted or merged by this tool",
        ],
    }
