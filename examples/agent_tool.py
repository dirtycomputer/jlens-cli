"""Minimal local tool adapter. The caller supplies model/lens paths; no server required."""

from __future__ import annotations

import json
import subprocess


def read_representations(model: str, lens: str, prompt: str, *, timeout: int = 600) -> dict:
    # Argument arrays and stdin avoid shell interpolation and quoting problems.
    completed = subprocess.run(
        [
            "j-lens",
            "apply",
            "--model",
            model,
            "--lens",
            lens,
            "--prompt-file",
            "-",
            "--positions=-1",
            "--top-k",
            "10",
        ],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        lines = completed.stderr.strip().splitlines()
        try:
            error = json.loads(lines[-1])["error"]
        except (IndexError, KeyError, ValueError):
            error = {"code": "process_error", "message": completed.stderr}
        raise RuntimeError(f"{error['code']}: {error['message']}")
    return json.loads(completed.stdout)["data"]
