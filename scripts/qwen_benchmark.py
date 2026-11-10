#!/usr/bin/env python3
"""Benchmark real, revision-pinned Qwen checkpoints with per-device telemetry."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

PROMPT = (
    "The research team compares language models using the same input and numerical "
    "precision. A careful experiment records elapsed time, memory consumption, "
    "and hardware utilization before drawing conclusions. "
) * 40


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def child(args):
    import psutil
    import pynvml as nv
    import torch
    import transformers
    from huggingface_hub import snapshot_download

    import jlens_cli.legacy as cli

    spec = json.loads(args.manifest.read_text())[args.model_index]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    info = {
        "model_id": spec["id"],
        "model_revision": spec["sha"],
        "nominal_size_b": spec["nominal_size_b"],
        "repository_parameters": spec["safetensors"]["total"],
        "gpu_family": args.family,
        "dtype": "bfloat16",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cpu_threads": 4,
        "sampling_interval_s": 0.1,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU fallback is prohibited")
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        nv.nvmlInit()
        visible_uuid = os.environ["CUDA_VISIBLE_DEVICES"]
        handle = nv.nvmlDeviceGetHandleByUUID(visible_uuid)
        name = nv.nvmlDeviceGetName(handle)
        name = name.decode() if isinstance(name, bytes) else name
        if args.family not in name:
            raise RuntimeError(f"Expected {args.family}, received {name}")
        memory = nv.nvmlDeviceGetMemoryInfo(handle)
        info.update(
            gpu_name=name,
            gpu_uuid=visible_uuid,
            gpu_total_bytes=memory.total,
            driver_version=nv.nvmlSystemGetDriverVersion(),
            device_capability=list(torch.cuda.get_device_capability(0)),
        )
        started = time.perf_counter()
        model_dir = snapshot_download(
            spec["id"],
            revision=spec["sha"],
            cache_dir=args.cache,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
            max_workers=4,
        )
        info["download_seconds_excluded"] = time.perf_counter() - started
        print("MODEL_DOWNLOADED", spec["id"], flush=True)
        proc = psutil.Process()
        telemetry = []

        def measure(stage, iteration, function):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            samples = []
            stop = threading.Event()
            origin = time.perf_counter()
            proc.cpu_percent()

            def sample():
                while not stop.is_set():
                    try:
                        util = nv.nvmlDeviceGetUtilizationRates(handle)
                        mem = nv.nvmlDeviceGetMemoryInfo(handle)
                        samples.append(
                            {
                                "elapsed_s": time.perf_counter() - origin,
                                "gpu_util_pct": util.gpu,
                                "memory_bandwidth_util_pct": util.memory,
                                "gpu_memory_used_bytes": mem.used,
                                "host_rss_bytes": proc.memory_info().rss,
                                "cpu_pct_one_core": proc.cpu_percent(),
                            }
                        )
                    except Exception as error:
                        samples.append({"sampling_error": str(error)})
                    stop.wait(0.1)

            thread = threading.Thread(target=sample, daemon=True)
            thread.start()
            status = "passed"
            error_text = ""
            value = None
            try:
                value = function()
                torch.cuda.synchronize()
            except Exception as error:
                status = "failed"
                error_text = f"{type(error).__name__}: {error}"
                raise
            finally:
                elapsed = time.perf_counter() - origin
                stop.set()
                thread.join()
                valid = [s for s in samples if "sampling_error" not in s]
                row = dict(info, stage=stage, iteration=iteration, status=status)
                row.update(
                    elapsed_s=elapsed,
                    error=error_text,
                    sample_count=len(valid),
                    sampling_errors=len(samples) - len(valid),
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
                if valid:
                    weights = [
                        max(
                            0,
                            (valid[i + 1]["elapsed_s"] if i + 1 < len(valid) else elapsed)
                            - sample["elapsed_s"],
                        )
                        for i, sample in enumerate(valid)
                    ]
                    total_weight = sum(weights)
                    for field in ("gpu_util_pct", "memory_bandwidth_util_pct"):
                        row[field + "_mean"] = (
                            sum(s[field] * w for s, w in zip(valid, weights, strict=True))
                            / total_weight
                        )
                        row[field + "_max"] = max(s[field] for s in valid)
                    row["nvml_peak_used_bytes"] = max(s["gpu_memory_used_bytes"] for s in valid)
                    row["vram_occupancy_peak_pct"] = (
                        100 * row["nvml_peak_used_bytes"] / memory.total
                    )
                    row["host_rss_peak_bytes"] = max(s["host_rss_bytes"] for s in valid)
                    row["cpu_cores_mean"] = (
                        sum(s["cpu_pct_one_core"] * w for s, w in zip(valid, weights, strict=True))
                        / total_weight
                        / 100
                    )
                rows.append(row)
                telemetry.extend(dict(s, stage=stage, iteration=iteration) for s in samples)
                with gzip.open(out / "telemetry.csv.gz", "wt") as stream:
                    fields = sorted({key for sample in telemetry for key in sample})
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(telemetry)
                save_json(out / "metrics.json", rows)
                print("QWEN_METRIC " + json.dumps(row), flush=True)
            return value

        corpus = out / "corpus.txt"
        corpus.write_text(PROMPT + "\n", encoding="utf-8")
        config = spec["config"].get("text_config", spec["config"])
        layers = [0, config["num_hidden_layers"] // 2, config["num_hidden_layers"] - 2]
        info.update(
            selected_layers=layers,
            total_layers=config["num_hidden_layers"],
            hidden_size=config["hidden_size"],
            fit_seq_len=32,
            apply_seq_len=128,
            dim_batch=8,
            corpus_prompts=1,
            batch_size=1,
        )
        common = [
            "--model",
            model_dir,
            "--device",
            "cuda:0",
            "--dtype",
            "bfloat16",
            "--local-files-only",
        ]
        lens_path = out / "lens.pt"
        fit_command = [
            "fit",
            *common,
            "--prompts",
            str(corpus),
            "--layers",
            ",".join(map(str, layers)),
            "--skip-first",
            "0",
            "--max-seq-len",
            "32",
            "--dim-batch",
            "8",
            "-o",
            str(lens_path),
            "--force",
        ]
        parsed = cli.parser().parse_args(fit_command)
        model = measure("model_load", 0, lambda: cli.load_model(parsed))
        info["loaded_parameters"] = sum(p.numel() for p in model._hf_model.parameters())
        info["actual_fit_tokens"] = model.encode(PROMPT, max_length=32).shape[1]
        info["actual_apply_tokens"] = model.encode(PROMPT, max_length=128).shape[1]
        original_load = cli.load_model
        cli.load_model = lambda unused: model

        def run(command):
            code = cli.main(command)
            if code:
                raise RuntimeError(f"CLI {command[0]} exited with code {code}")

        # Warm the GPU using a forward pass; this is not included in timed stages.
        with torch.no_grad():
            model.forward(model.encode(PROMPT, max_length=32))
        torch.cuda.synchronize()
        for iteration in range(2):
            gc.collect()
            torch.cuda.empty_cache()
            measure("resident_cli_fit", iteration, lambda: run(fit_command))
        apply_command = [
            "apply",
            *common,
            "--lens",
            str(lens_path),
            "--prompt",
            PROMPT,
            "--max-seq-len",
            "128",
            "--positions=-1",
            "--top-k",
            "5",
            "-o",
            str(out / "apply.json"),
            "--force",
        ]
        run(apply_command)

        def repeat_apply():
            start = time.perf_counter()
            repeats = 0
            while repeats < 5 or time.perf_counter() - start < 15:
                run(apply_command)
                repeats += 1
            return repeats

        for iteration in range(2):
            repeats = measure("resident_cli_apply", iteration, repeat_apply)
            rows[-1]["repeats"] = repeats
            rows[-1]["seconds_per_call"] = rows[-1]["elapsed_s"] / repeats
            rows[-1]["input_tokens_per_s"] = (
                repeats * info["actual_apply_tokens"] / rows[-1]["elapsed_s"]
            )
        cli.load_model = original_load
        for row in rows:
            row.update(info)
        with gzip.open(out / "telemetry.csv.gz", "wt") as stream:
            fields = sorted({key for sample in telemetry for key in sample})
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(telemetry)
        save_json(out / "model-info.json", info)
        save_json(out / "metrics.json", rows)
        save_json(out / "status.json", {"status": "passed", "model_id": spec["id"]})
        return 0
    except Exception as error:
        failure = dict(
            info,
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        save_json(out / "status.json", failure)
        save_json(out / "metrics.json", rows)
        print("QWEN_FAILURE " + json.dumps(failure), flush=True)
        return 1


def orchestrate(args):
    specs = json.loads(args.manifest.read_text())
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).split()
    if len(uuids) != 2:
        raise RuntimeError(f"Expected exactly two assigned GPUs; got {len(uuids)}")
    args.output.mkdir(parents=True, exist_ok=True)
    assignments = [[], []]
    totals = [0, 0]
    for index in sorted(
        range(len(specs)), key=lambda i: specs[i]["safetensors"]["total"], reverse=True
    ):
        worker = min(range(2), key=lambda i: totals[i])
        assignments[worker].append(index)
        totals[worker] += specs[index]["safetensors"]["total"]
    failures = []
    print_lock = threading.Lock()

    def work(worker):
        for index in reversed(assignments[worker]):
            spec = specs[index]
            directory = args.output / spec["id"].split("/")[-1]
            directory.mkdir(exist_ok=True)
            command = [
                sys.executable,
                __file__,
                "--child",
                "--manifest",
                str(args.manifest),
                "--model-index",
                str(index),
                "--family",
                args.family,
                "--output",
                str(directory),
                "--cache",
                str(args.cache),
            ]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuids[worker])
            with print_lock:
                print("QWEN_START", args.family, spec["id"], "GPU", worker, flush=True)
            try:
                with (directory / "worker.log").open("w") as stream:
                    process = subprocess.Popen(
                        command, env=environment, stdout=stream, stderr=stream
                    )
                    started = time.monotonic()
                    while True:
                        try:
                            status = process.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic() - started >= 2400:
                                process.kill()
                                process.wait()
                                raise
                            with print_lock:
                                tail = (directory / "worker.log").read_text()[-500:]
                                print("QWEN_PROGRESS", spec["id"], tail, flush=True)
            except subprocess.TimeoutExpired:
                status = 124
                save_json(
                    directory / "status.json",
                    {"model_id": spec["id"], "status": "timeout", "timeout_s": 2400},
                )
            with print_lock:
                print("QWEN_DONE", args.family, spec["id"], "exit", status, flush=True)
                for filename in ("metrics.json", "status.json"):
                    path = directory / filename
                    if path.exists():
                        print(
                            "QWEN_RESULT "
                            + json.dumps(
                                {
                                    "family": args.family,
                                    "model_id": spec["id"],
                                    "file": filename,
                                    "data": json.loads(path.read_text()),
                                }
                            ),
                            flush=True,
                        )
                if status:
                    failures.append(spec["id"])
                    print((directory / "worker.log").read_text()[-7000:], flush=True)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary = {"family": args.family, "models": len(specs), "failed": failures}
    save_json(args.output / "summary.json", summary)
    print("QWEN_SUMMARY", json.dumps(summary), flush=True)
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--model-index", type=int)
    args = parser.parse_args()
    return child(args) if args.child else orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
