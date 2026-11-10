"""Exact rollout tokens, explicit templates and observable truncation."""

from __future__ import annotations

from .io import CLIError, digest, indices


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise CLIError(f"{name} must be an integer >= {minimum}.")
    return value


def prepare(model, request, *, max_tokens=512):
    if not isinstance(request, dict):
        raise CLIError("Input must be an object.")
    allowed = {
        "id",
        "input_ids",
        "text",
        "messages",
        "add_generation_prompt",
        "max_seq_len",
        "truncation",
        "attention_mask",
        "position_ids",
        "positions",
    }
    if set(request) - allowed:
        raise CLIError("Unknown input sample fields.")
    if not isinstance(request.get("id", 0), (str, int)) or isinstance(request.get("id", 0), bool):
        raise CLIError("Sample id must be a string or integer.")
    if "add_generation_prompt" in request and type(request["add_generation_prompt"]) is not bool:
        raise CLIError("add_generation_prompt must be a boolean.")
    kinds = [key for key in ("input_ids", "text", "messages") if key in request]
    if len(kinds) != 1:
        raise CLIError("Provide exactly one of input_ids, text or messages.")
    kind = kinds[0]
    if kind == "input_ids":
        ids = request[kind]
        if not isinstance(ids, list):
            raise CLIError("input_ids must be a list of integers.")
        ids = [integer(v, "token ID") for v in ids]
    elif kind == "text":
        if not isinstance(request[kind], str) or not request[kind].strip():
            raise CLIError("text must be a nonempty string.")
        ids = model.tokenizer(request[kind], truncation=False)["input_ids"]
    else:
        if not isinstance(request[kind], list) or not request[kind]:
            raise CLIError("messages must be a nonempty list.")
        ids = model.tokenizer.apply_chat_template(
            request[kind],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=request.get("add_generation_prompt", True),
        )
    if not ids or any(v >= model._hf_model.get_input_embeddings().num_embeddings for v in ids):
        raise CLIError("Input is empty or contains IDs outside the model vocabulary.")
    limit = integer(request.get("max_seq_len", max_tokens), "max_seq_len", 1)
    if limit > max_tokens:
        raise CLIError("max_seq_len exceeds the session token budget.", code="budget_exceeded")
    original = len(ids)
    mode = request.get("truncation", "error")
    if mode not in ("error", "left", "right"):
        raise CLIError("truncation must be error, left or right.")
    if original > limit and mode == "error":
        raise CLIError(
            f"Input has {original} tokens, budget {limit}; choose explicit truncation.",
            code="input_truncated",
        )
    offset = max(0, original - limit) if mode == "left" else 0
    ids = ids[offset : offset + limit]
    auxiliary = {}
    for key in ("attention_mask", "position_ids"):
        if key in request:
            values = request[key]
            if not isinstance(values, list) or len(values) != original:
                raise CLIError(f"{key} must match the original input length.")
            values = [integer(v, key) for v in values]
            if key == "attention_mask" and (
                any(v not in (0, 1) for v in values) or not any(values[offset : offset + limit])
            ):
                raise CLIError("attention_mask must contain binary values and an unmasked token.")
            auxiliary[key] = values[offset : offset + limit]
    spec = request.get("positions", "-1")
    if isinstance(spec, list):
        spec = ",".join(str(integer(v, "position", -len(ids))) for v in spec)
    if not isinstance(spec, str):
        raise CLIError("positions must be a slice string or integer list.")
    selected = indices(spec, len(ids))
    if "attention_mask" in auxiliary and any(not auxiliary["attention_mask"][p] for p in selected):
        raise CLIError("Selected positions must not be padding.")
    payload = {"input_ids": ids, **auxiliary}
    return {
        **payload,
        "id": request.get("id", 0),
        "input_hash": digest(payload),
        "input_kind": kind,
        "num_tokens": len(ids),
        "original_num_tokens": original,
        "truncated": original > limit,
        "truncation": mode,
        "position_offset": offset,
        "positions": selected,
        "original_positions": [p + offset for p in selected],
        "predicts_original_positions": [p + offset + 1 for p in selected],
        "tokens": [
            {"position": p, "token_id": token, "token": model.tokenizer.decode([token])}
            for p, token in enumerate(ids)
        ],
    }
