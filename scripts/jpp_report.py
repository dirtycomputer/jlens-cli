#!/usr/bin/env python3
"""Export completed J++ measurements and workload-specific resource guidance."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

GIB = 1024**3


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    measurements, summaries = [], []
    for family in ("H100", "B200"):
        for preset in ("qwen3.5-9b", "qwen3.6-27b"):
            directory = args.input / family / preset
            status = json.loads((directory / "status.json").read_text())
            if status["status"] != "passed" or not status["cli_output_verified"]:
                raise ValueError(f"incomplete benchmark: {family}/{preset}")
            rows = json.loads((directory / "metrics.json").read_text())
            with gzip.open(directory / "telemetry.csv.gz", "rt") as stream:
                samples = list(csv.DictReader(stream))
            # An in-flight sample can finish just after the workload. Clip its
            # interval to the measured window; never assign negative weight.
            for row in rows:
                matched = [
                    s
                    for s in samples
                    if s["stage"] == row["stage"] and int(s["iteration"]) == row["iteration"]
                ]
                times = [float(s["elapsed_s"]) for s in matched]
                weights = [
                    max(0, min(times[i + 1], row["elapsed_s"]) - t)
                    if i + 1 < len(times)
                    else max(0, row["elapsed_s"] - t)
                    for i, t in enumerate(times)
                ]
                row["sampling_tail_s"] = max(0, max(times) - row["elapsed_s"])
                for key in ("gpu_util_pct", "memory_bandwidth_util_pct", "cpu_pct"):
                    row[key + "_mean"] = sum(
                        float(s[key]) * w for s, w in zip(matched, weights, strict=True)
                    ) / sum(weights)
            measurements.extend(rows)
            for stage in ("seven_layers", "all_layers"):
                windows = [r for r in rows if r["stage"] == stage]
                if len(windows) != 2 or any(r["elapsed_s"] < 15 for r in windows):
                    raise ValueError("expected two measured windows of at least 15 seconds")
                duration = sum(r["elapsed_s"] for r in windows)
                requests = sum(r["requests"] for r in windows)
                relevant = [
                    r
                    for r in rows
                    if r["stage"]
                    in (
                        "model_load",
                        "lens_load_validate",
                        "readout_filter_setup",
                        stage + "_lens_to_gpu",
                        stage,
                    )
                ]
                vram = max(r["nvml_peak_used_bytes"] for r in relevant) / GIB
                rss = max(r["host_rss_peak_bytes"] for r in relevant) / GIB
                row = dict(
                    model=preset,
                    gpu_family=family,
                    mode=stage,
                    readout_layers=windows[0]["readout_layers"],
                    batch_size=1,
                    input_tokens=128,
                    positions="last",
                    requests=requests,
                    measured_seconds=duration,
                    latency_ms=1000 * duration / requests,
                    requests_per_second=requests / duration,
                    gpu_util_pct=sum(r["gpu_util_pct_mean"] * r["elapsed_s"] for r in windows)
                    / duration,
                    memory_bandwidth_util_pct=sum(
                        r["memory_bandwidth_util_pct_mean"] * r["elapsed_s"] for r in windows
                    )
                    / duration,
                    resident_nvml_peak_gib=max(r["nvml_peak_used_bytes"] for r in windows) / GIB,
                    torch_peak_allocated_gib=max(r["torch_peak_allocated_bytes"] for r in windows)
                    / GIB,
                    gpu_capacity_gib=windows[0]["gpu_total_bytes"] / GIB,
                    workload_vram_peak_gib=vram,
                    vram_occupancy_pct=100 * vram / (windows[0]["gpu_total_bytes"] / GIB),
                    host_rss_peak_gib=rss,
                    cpu_cores_mean=sum(r["cpu_pct_mean"] * r["elapsed_s"] for r in windows)
                    / duration
                    / 100,
                    configured_cpu_threads=4,
                    suggested_single_gpu_vram_gib=math.ceil(vram * 1.2),
                    suggested_host_ram_gib=math.ceil(rss * 1.25 / 8) * 8,
                    suggested_cpu_cores=4,
                    resource_rule="VRAM peak*1.20; RSS peak*1.25 rounded to 8 GiB; 4 CPU threads",
                    validation="published checkpoint + actual CLI passed; no quality evaluation",
                )
                summaries.append(row)
    write_csv(args.output / "measurements.csv", measurements)
    write_csv(args.output / "resource_summary.csv", summaries)
    inventory = args.input / "storage_inventory.json"
    if inventory.exists():
        storage = json.loads(inventory.read_text())
        for row in storage:
            row["logical_gib"] = row["logical_bytes"] / GIB
            row["allocated_gib"] = row["allocated_bytes"] / GIB
        write_csv(args.output / "storage_inventory.csv", storage)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.4))
    fig.patch.set_facecolor("#f6f8fc")
    order = sorted(
        summaries,
        key=lambda r: (
            r["model"],
            r["mode"] == "all_layers",
            r["gpu_family"] == "B200",
        ),
    )
    labels = [
        r["model"].replace("qwen", "Qwen")
        + "  /  "
        + ("7 layers" if r["mode"] == "seven_layers" else f"{r['readout_layers']} layers")
        + "  /  "
        + r["gpu_family"]
        for r in order
    ]
    colors = ["#2165c6" if r["gpu_family"] == "H100" else "#089b88" for r in order]
    for ax, field, title, unit in zip(
        axes,
        ["latency_ms", "resident_nvml_peak_gib", "gpu_util_pct"],
        ["Readout latency", "Peak resident VRAM", "Mean GPU utilization"],
        ["ms / request", "GiB", "%"],
        strict=True,
    ):
        ax.set_facecolor("#f6f8fc")
        bars = ax.barh(range(len(order)), [r[field] for r in order], color=colors, height=0.66)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", weight="bold", pad=16)
        ax.set_xlabel(unit)
        ax.set_yticks(range(len(order)), labels if ax == axes[0] else [""] * len(order))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(r[field] for r in order) * 1.24)
        ax.bar_label(bars, fmt="%.1f", padding=5, fontsize=10)
    fig.suptitle(
        "J++ checkpoint inference · H100 vs B200",
        x=0.03,
        y=0.97,
        ha="left",
        fontsize=23,
        weight="bold",
        color="#17243d",
    )
    fig.text(
        0.03,
        0.9,
        "Real Qwen weights · BF16 model + FP32 lens · batch 1 · "
        "128 input tokens · last-token top-10",
        color="#4b5b75",
        fontsize=12,
    )
    fig.legend(
        handles=[Patch(color="#2165c6", label="H100"), Patch(color="#089b88", label="B200")],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.03,
        0.04,
        "2 warm-ups + 2 × ≥15 s per mode; 100 ms NVML sampling.\n"
        "Loading excluded from latency; not a training or Recall@10 benchmark.",
        color="#4b5b75",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.01, 0.12, 0.99, 0.86), w_pad=2)
    fig.savefig(args.output / "gpu_comparison.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(args.output / "gpu_comparison.svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
