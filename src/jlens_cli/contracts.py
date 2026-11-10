"""Lightweight JSON Schemas for tool adapters; no model-runtime imports."""

SAMPLE = {
    "type": "object",
    "properties": {
        "id": {"type": ["string", "integer"]},
        "text": {"type": "string", "minLength": 1},
        "messages": {"type": "array", "minItems": 1, "items": {"type": "object"}},
        "input_ids": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 0}},
        "attention_mask": {"type": "array", "items": {"enum": [0, 1]}},
        "position_ids": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "positions": {
            "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "integer"}}]
        },
        "max_seq_len": {"type": "integer", "minimum": 1},
        "truncation": {"enum": ["error", "left", "right"], "default": "error"},
        "add_generation_prompt": {"type": "boolean", "default": True},
    },
    "oneOf": [{"required": [key]} for key in ("text", "messages", "input_ids")],
    "additionalProperties": False,
}
CONCEPTS = {
    "type": "object",
    "additionalProperties": {
        "type": "array",
        "minItems": 1,
        "items": {"type": ["string", "integer"]},
    },
}
SAMPLES = {"type": "array", "minItems": 1, "items": SAMPLE}
POSITIVE = {"type": "integer", "minimum": 1}
REQUIRED = {
    "input.prepare": ["sample"],
    "concepts.resolve": ["concepts"],
    "readout": ["samples"],
    "concepts.trajectory": ["observation"],
    "audit.scan": ["observation"],
    "components.score": ["modules", "failure_samples", "reference_samples", "concepts"],
    "checkpoint.compare": ["before", "after"],
    "trials.propose": [
        "component_report",
        "allowed_modules",
        "algorithm",
        "training_data_id",
        "max_trainable_parameters",
    ],
}
FIELDS = {
    "canaries": {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "required": ["sample_id", "expected_token_ids"],
            "properties": {
                "sample_id": {"type": ["string", "integer"]},
                "expected_token_ids": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "integer", "minimum": 0},
                },
            },
        },
    },
    "min_pass_rate": {"type": "number", "minimum": 0, "maximum": 1},
    "min_samples": POSITIVE,
    "sample": SAMPLE,
    "samples": SAMPLES,
    "failure_samples": SAMPLES,
    "reference_samples": SAMPLES,
    "concepts": CONCEPTS,
    "kind": {"enum": ["transported", "logit"]},
    "layers": {
        "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "integer", "minimum": 0}}]
    },
    "top_k": {**POSITIVE, "maximum": 1000},
    "rank_threshold": POSITIVE,
    "baseline": {"type": "boolean"},
    "readout_filter": {"type": "boolean"},
    "track_token_ids": {"type": "array", "items": {"type": "integer", "minimum": 0}},
    "deadline_seconds": {"type": "number", "exclusiveMinimum": 0},
    "wrong_weight": {"type": "number", "minimum": 0},
    "background_weight": {"type": "number", "minimum": 0},
    "modules": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string"}},
    "allowed_modules": {"type": "array", "minItems": 1, "items": {"type": "string"}},
    "algorithm": {"enum": ["sft", "grpo", "crt"]},
    "training_data_id": {"type": "string", "minLength": 1},
    "rank": POSITIVE,
    "max_trainable_parameters": POSITIVE,
    "seed": {"type": "integer", "minimum": 0},
    "min_overlap": {"type": "number", "minimum": 0, "maximum": 1},
    "min_cells": POSITIVE,
    "allow_lens_change": {"type": "boolean"},
    "loss_scope": {"enum": ["response", "reflection", "selected_positions"]},
    "loss_positions": {
        "type": "object",
        "additionalProperties": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "integer", "minimum": 0},
        },
    },
}


def input_schema(method, params):
    result = {
        "type": "object",
        "properties": {name: FIELDS.get(name, {"type": "object"}) for name in params},
        "required": REQUIRED.get(method, []),
        "additionalProperties": False,
    }
    if method == "lens.health":
        result["oneOf"] = [
            {"required": ["reference", "candidate", "min_overlap", "min_cells"]},
            {"required": ["observation", "canaries", "min_pass_rate", "min_samples"]},
        ]
    return result
