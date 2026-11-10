#!/usr/bin/env python3
"""J++ inference benchmark: two models, one process per assigned GPU, offline."""

from __future__ import annotations

import argparse
import csv
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


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def child(args):
    import psutil
    import pynvml as nv
    import torch
    import transformers

    from jlens_cli import jpp
    from jlens_cli import legacy as cli

    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "CPU fallback prohibited"
    torch.cuda.set_device(0)
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByUUID(os.environ["CUDA_VISIBLE_DEVICES"])
    gpu_name = nv.nvmlDeviceGetName(handle)
    assert args.family in gpu_name
    total = nv.nvmlDeviceGetMemoryInfo(handle).total
    proc = psutil.Process()
    preset = jpp.PRESETS[args.preset]
    info = dict(
        preset=args.preset,
        gpu_family=args.family,
        gpu_name=gpu_name,
        gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],
        gpu_total_bytes=total,
        model_id=preset["model"],
        model_revision=preset["model_revision"],
        lens_revision=preset["lens_revision"],
        model_dtype="bfloat16",
        lens_dtype="float32",
        batch_size=1,
        sequence_length=128,
        positions="last",
        cpu_threads=4,
        attention="eager",
        kv_cache=False,
        sample_interval_s=0.1,
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        cuda=torch.version.cuda,
    )
    rows, telemetry = [], []
    args.output.mkdir(parents=True, exist_ok=True)

    def measure(stage, iteration, function):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples, stop = [], threading.Event()
        origin = time.perf_counter()
        proc.cpu_percent()

        def sample():
            while not stop.is_set():
                util = nv.nvmlDeviceGetUtilizationRates(handle)
                samples.append(
                    dict(
                        elapsed_s=time.perf_counter() - origin,
                        gpu_util_pct=util.gpu,
                        memory_bandwidth_util_pct=util.memory,
                        gpu_memory_used_bytes=nv.nvmlDeviceGetMemoryInfo(handle).used,
                        host_rss_bytes=proc.memory_info().rss,
                        cpu_pct=proc.cpu_percent(),
                    )
                )
                stop.wait(0.1)

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        try:
            value = function()
            torch.cuda.synchronize()
        finally:
            elapsed = time.perf_counter() - origin
            stop.set()
            thread.join()
        if not samples:
            raise RuntimeError("missing NVML samples")
        weights = [
            max(0, min(samples[i + 1]["elapsed_s"], elapsed) - s["elapsed_s"])
            if i + 1 < len(samples)
            else max(0, elapsed - s["elapsed_s"])
            for i, s in enumerate(samples)
        ]
        row = dict(
            info,
            stage=stage,
            iteration=iteration,
            elapsed_s=elapsed,
            sample_count=len(samples),
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            nvml_peak_used_bytes=max(s["gpu_memory_used_bytes"] for s in samples),
            host_rss_peak_bytes=max(s["host_rss_bytes"] for s in samples),
        )
        for key in ("gpu_util_pct", "memory_bandwidth_util_pct", "cpu_pct"):
            row[key + "_mean"] = sum(
                s[key] * w for s, w in zip(samples, weights, strict=True)
            ) / sum(weights)
        row["vram_peak_pct"] = row["nvml_peak_used_bytes"] / total * 100
        if isinstance(value, dict) and "requests" in value:
            row.update(value)
            row["latency_ms"] = elapsed / value["requests"] * 1000
        rows.append(row)
        telemetry.extend(dict(s, stage=stage, iteration=iteration) for s in samples)
        save(args.output / "metrics.json", rows)
        with gzip.open(args.output / "telemetry.csv.gz", "wt") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(telemetry[0]))
            writer.writeheader()
            writer.writerows(telemetry)
        print("JPP_METRIC", json.dumps(row), flush=True)
        return value

    try:
        model_path = args.cache / ("models--" + preset["model"].replace("/", "--"))
        model_path = model_path / "snapshots" / preset["model_revision"]
        lens_path = args.lens_cache / "models--koayon--jpp-lenses" / "snapshots"
        lens_path = lens_path / preset["lens_revision"] / preset["lens_file"]
        command = [
            "apply",
            "--preset",
            args.preset,
            "--model",
            str(model_path),
            "--model-identity",
            preset["model"],
            "--lens",
            str(lens_path),
            "--device",
            "cuda:0",
            "--dtype",
            "bfloat16",
            "--local-files-only",
            "--max-seq-len",
            "128",
            "--prompt",
            PROMPT,
            "--top-k",
            "10",
        ]
        options = jpp.parser().parse_args(command)
        jpp.resolve(options)
        model = measure("model_load", 0, lambda: cli.load_model(options))
        info["model_parameters_loaded"] = sum(p.numel() for p in model._hf_model.parameters())
        lens, config = measure("lens_load_validate", 0, lambda: jpp.load_checkpoint(lens_path))
        jpp.validate_model(lens, config, model, preset["model"])
        info["lens_file_bytes"] = lens_path.stat().st_size
        mask = measure(
            "readout_filter_setup",
            0,
            lambda: jpp.readout_mask(model.tokenizer, model._lm_head.out_features),
        )
        for mode, layers in [
            ("seven_layers", preset["paper_layers"]),
            ("all_layers", lens.source_layers),
        ]:
            options.layers = layers
            info["readout_layers"] = len(layers)
            measure(
                mode + "_lens_to_gpu",
                0,
                lambda layers=layers: [
                    lens.jacobians.__setitem__(layer, lens.jacobians[layer].to("cuda:0"))
                    for layer in layers
                ],
            )
            for _ in range(2):
                example = jpp.infer(options, model, lens, config, PROMPT, mask)
            assert len(example["input_token_ids"]) == 128
            assert len(example["layers"]) == len(layers)
            assert all(len(v[0]) == 10 for v in example["layers"].values())
            save(args.output / (mode + "_output.json"), example)

            def window(example=example):
                start, count = time.perf_counter(), 0
                while time.perf_counter() - start < args.window_seconds:
                    result = jpp.infer(options, model, lens, config, PROMPT, mask)
                    assert result["layers"] == example["layers"], "nondeterministic readout"
                    count += 1
                return {"requests": count}

            for iteration in range(2):
                measure(mode, iteration, window)
        # Exercise the actual parser / command / JSON file output with the published lens.
        output = args.output / "cli_output.json"
        started = time.perf_counter()
        lens = model = mask = None
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        assert jpp.main(["--legacy", *command, "--paper-layers", "-o", str(output)]) == 0
        actual = json.loads(output.read_text())
        expected = json.loads((args.output / "seven_layers_output.json").read_text())
        assert actual["layers"] == expected["layers"]
        save(
            args.output / "status.json",
            dict(
                status="passed",
                cli_output_verified=True,
                cli_cold_seconds=time.perf_counter() - started,
                **info,
            ),
        )
        return 0
    except Exception as error:
        traceback.print_exc()
        save(args.output / "status.json", dict(status="failed", error=str(error), **info))
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--lens-cache", type=Path, required=True)
    parser.add_argument("--preset")
    parser.add_argument("--window-seconds", type=float, default=15)
    args = parser.parse_args()
    if args.preset:
        return child(args)
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).split()
    assert len(uuids) == 2
    args.output.mkdir(parents=True, exist_ok=True)
    processes = []
    for uuid, preset in zip(uuids, ["qwen3.5-9b", "qwen3.6-27b"], strict=True):
        out = args.output / preset
        out.mkdir(exist_ok=True)
        with (out / "worker.log").open("w") as stream:
            process = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--family",
                    args.family,
                    "--output",
                    str(out),
                    "--cache",
                    str(args.cache),
                    "--lens-cache",
                    str(args.lens_cache),
                    "--preset",
                    preset,
                    "--window-seconds",
                    str(args.window_seconds),
                ],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid),
                stdout=stream,
                stderr=stream,
            )
        processes.append((preset, process))
    deadline = time.monotonic() + 2400
    while any(p.poll() is None for _, p in processes):
        if time.monotonic() > deadline:
            for _, p in processes:
                if p.poll() is None:
                    p.kill()
            break
        for preset, p in processes:
            print(
                "JPP_PROGRESS",
                preset,
                p.poll(),
                (args.output / preset / "worker.log").read_text()[-800:],
                flush=True,
            )
        time.sleep(20)
    codes = {preset: p.wait() for preset, p in processes}
    save(args.output / "summary.json", codes)
    return int(any(codes.values()))


if __name__ == "__main__":
    sys.exit(main())
