"""Minimal synchronous process adapter; caller owns scheduling and timeouts."""

from __future__ import annotations

import json
import subprocess


class ResidentTool:
    def __init__(self, config_path, executable="jlens"):
        self.process = subprocess.Popen(
            [executable, "serve", "--config", str(config_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.counter = 0

    def call(self, method, params=None, manifest_id=None):
        self.counter += 1
        request = {"id": self.counter, "method": method, "params": params or {}}
        if manifest_id is not None:
            request["manifest_id"] = manifest_id
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Lens worker exited: {self.process.poll()}")
        reply = json.loads(line)
        if reply["id"] != self.counter:
            raise RuntimeError("Unexpected tool response ID")
        if reply["status"] != "ok":
            raise RuntimeError(reply["error"])
        return reply["data"]

    def close(self):
        try:
            if self.process.poll() is None:
                self.call("close")
        finally:
            self.process.wait(timeout=30)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
