"""Validated text input, index selection and atomic result files."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


class CLIError(Exception):
    def __init__(self, message: str, *, code: str = "invalid_input", exit_code: int = 2):
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


def read_text(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")


def read_records(path: str, *, text_field: str = "text", input_format: str = "auto") -> list[dict]:
    content = read_text(path)
    if input_format == "auto":
        suffix = Path(path).suffix.lower()
        input_format = {".json": "json", ".jsonl": "jsonl"}.get(suffix, "text")
    try:
        if input_format == "json":
            raw = json.loads(content)
            if not isinstance(raw, list):
                raise CLIError("JSON input must be an array of strings or objects.")
        elif input_format == "jsonl":
            raw = []
            for number, line in enumerate(content.splitlines(), 1):
                if line.strip():
                    try:
                        raw.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise CLIError(f"Invalid JSONL at line {number}: {exc.msg}") from exc
        else:
            raw = [line for line in content.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise CLIError(f"Invalid JSON: {exc.msg}") from exc
    records = []
    for index, item in enumerate(raw):
        value = item.get(text_field) if isinstance(item, dict) else item
        if not isinstance(value, str) or not value.strip():
            raise CLIError(f"Record {index + 1} must contain non-empty text ({text_field!r}).")
        record_id = item.get("id", index) if isinstance(item, dict) else index
        if not isinstance(record_id, (str, int)) or isinstance(record_id, bool):
            raise CLIError(f"Record {index + 1} id must be a string or integer.")
        records.append({"id": record_id, "text": value})
    if not records:
        raise CLIError("Input contains no prompts.")
    return records


def indices(spec: str | None, size: int, *, default: list[int] | None = None) -> list[int]:
    """Resolve integers and Python slices, preserving order; reject invalid integers."""
    if spec is None:
        return list(range(size)) if default is None else list(default)
    if spec == "all":
        return list(range(size))
    resolved = []
    try:
        for part in spec.split(","):
            part = part.strip()
            if ":" in part:
                fields = part.split(":")
                if len(fields) not in (2, 3):
                    raise ValueError("slices require start:stop[:step]")
                values = [int(value) if value else None for value in fields]
                selection = list(range(size))[slice(*values)]
            else:
                value = int(part)
                value = value + size if value < 0 else value
                if not 0 <= value < size:
                    raise ValueError(f"index {part} is out of range for size {size}")
                selection = [value]
            resolved.extend(selection)
    except (ValueError, TypeError) as exc:
        raise CLIError(f"Invalid index selection {spec!r}: {exc}") from exc
    if not resolved:
        raise CLIError(f"Index selection {spec!r} is empty.")
    if len(set(resolved)) != len(resolved):
        raise CLIError(f"Index selection {spec!r} contains duplicates.")
    return resolved


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def ensure_output(path: str | None, *, force: bool = False) -> None:
    if path and path != "-":
        target = Path(path)
        if target.exists() and not force:
            raise CLIError(
                f"Output exists: {path}. Use --force to replace it.", code="output_exists"
            )
        if target.is_dir():
            raise CLIError(f"Output is a directory: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)


def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def atomic_lens(lens, path: str, *, dtype) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path_obj.parent, prefix=f".{path_obj.name}.")
    os.close(fd)
    try:
        lens.save(temp, dtype=dtype)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def write_json(path: str | Path, data: object) -> None:
    atomic_text(path, json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def read_metadata(lens_path: str) -> dict | None:
    path = Path(f"{lens_path}.meta.json")
    if Path(lens_path).is_file() and path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise CLIError(f"Invalid metadata in {path}")
        return data
    return None
