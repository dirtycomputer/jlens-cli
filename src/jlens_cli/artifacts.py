"""Content identities and merge lineage, independent of the numerical runtime."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .io import CLIError, digest


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def model_files(name):
    path = Path(name)
    if not path.is_dir():
        return []
    patterns = (
        "*.jinja",
        "*.safetensors",
        "pytorch_model*.bin",
        "*.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer*.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "*.model",
        "vocab.txt",
        "merges.txt",
        "*.tiktoken",
    )
    return sorted({p for pattern in patterns for p in path.glob(pattern) if p.is_file()})


def weights_identity(args, model):
    files = model_files(args.model)
    commit = getattr(model._hf_model.config, "_commit_hash", None)
    if files:
        payload = {p.name: file_digest(p) for p in files}
        identity = {"kind": "local-content", "sha256": digest(payload)}
    elif commit:
        identity = {"kind": "hub-commit", "model": args.model, "commit": commit}
    else:
        identity = {"kind": "unverified", "model": args.model, "revision": args.revision}
    tokenizer = model.tokenizer
    backend = getattr(tokenizer, "backend_tokenizer", None)
    token_state = backend.to_str() if backend is not None else tokenizer.get_vocab()
    return {
        "weights": identity,
        "tokenizer_sha256": digest(
            {
                "state": token_state,
                "special_tokens": tokenizer.special_tokens_map,
                "chat_template": getattr(tokenizer, "chat_template", None),
                "force_bos": not args.no_force_bos,
            }
        ),
    }


def lineage(metadata):
    if not metadata:
        return []
    if metadata.get("fit"):
        return [metadata["fit"]]
    result = []
    for child in metadata.get("input_metadata", []):
        result.extend(lineage(child))
    return result


def check_lineage(metadata):
    leaves = [leaf for item in metadata for leaf in lineage(item)]
    configurations = [
        {k: v for k, v in leaf.items() if k not in ("prompts_sha256", "shard_index", "num_shards")}
        for leaf in leaves
    ]
    if configurations and any(item != configurations[0] for item in configurations[1:]):
        raise CLIError(
            "Cannot merge lenses fitted with different estimator settings or weights.",
            code="incompatible_lens",
        )
    hashes = [leaf["prompts_sha256"] for leaf in leaves]
    if len(hashes) != len(set(hashes)):
        raise CLIError("Input lineage identifies repeated prompt subsets; use disjoint shards.")
    return leaves


def complete_lineage(metadata):
    if not isinstance(metadata, dict):
        return False
    if metadata.get("fit"):
        return {"model", "layers", "prompts_sha256"} <= metadata["fit"].keys()
    children = metadata.get("input_metadata", [])
    return bool(children) and all(complete_lineage(child) for child in children)


def protect_paths(args, extra=()):
    """Resolve directory aliases and protect source artifacts even with --force."""
    inputs = [getattr(args, k, None) for k in ("prompts", "input", "prompt_file", "lens")]
    inputs.extend(getattr(args, "lenses", []))
    lens = getattr(args, "lens", None)
    if lens and Path(lens).is_dir():
        inputs.append(
            Path(lens)
            / (
                getattr(args, "lens_filename", None)
                or getattr(args, "lens_file", None)
                or "lens.pt"
            )
        )
    inputs.extend(model_files(getattr(args, "model", "") or "__no_model__"))
    inputs.extend(extra)
    protected = {Path(p).resolve() for p in inputs if p and str(p) != "-"}
    protected |= {Path(str(p) + ".meta.json") for p in protected}
    outputs = [getattr(args, k, None) for k in ("output", "checkpoint")]
    targets = {Path(p).resolve() for p in outputs if p and str(p) != "-"}
    if getattr(args, "command", None) in ("fit", "merge"):
        targets |= {Path(str(p) + ".meta.json") for p in targets}
    if targets & protected:
        raise CLIError("Output/checkpoint paths must not overwrite input files or their metadata.")
