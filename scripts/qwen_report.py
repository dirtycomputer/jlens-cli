#!/usr/bin/env python3
"""Aggregate qwen_benchmark outputs into auditable CSVs and a comparison plot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

GIB = 1024**3
STAGES = ("model_load", "resident_cli_fit", "resident_cli_apply")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            )


def aggregate(root, manifest):
    raw, summary, recommendations, statuses = [], [], [], []
    for spec in manifest:
        for family in ("H100", "B200"):
            directory = root / family / spec["id"].split("/")[-1]
            status_path = directory / "status.json"
            status = (
                json.loads(status_path.read_text())
                if status_path.exists()
                else {"status": "missing"}
            )
            statuses.append(
                {
                    "model_id": spec["id"],
                    "gpu_family": family,
                    "status": status["status"],
                    "error": status.get("error", ""),
                }
            )
            path = directory / "metrics.json"
            rows = json.loads(path.read_text()) if path.exists() else []
            raw.extend(rows)
            passed = [row for row in rows if row["status"] == "passed"]
            for stage in STAGES:
                measured = [row for row in passed if row["stage"] == stage]
                if not measured:
                    continue
                elapsed = sum(r["elapsed_s"] for r in measured)
                item = {
                    "model_id": spec["id"],
                    "model_revision": spec["sha"],
                    "nominal_size_b": spec["nominal_size_b"],
                    "loaded_parameters": measured[0].get("loaded_parameters", ""),
                    "dtype": measured[0]["dtype"],
                    "gpu_family": family,
                    "gpu_name": measured[0]["gpu_name"],
                    "gpu_total_gib": measured[0]["gpu_total_bytes"] / GIB,
                    "stage": stage,
                    "measurements": len(measured),
                    "elapsed_median_s": statistics.median(r["elapsed_s"] for r in measured),
                    "elapsed_min_s": min(r["elapsed_s"] for r in measured),
                    "elapsed_max_s": max(r["elapsed_s"] for r in measured),
                }
                for metric in (
                    "gpu_util_pct_mean",
                    "memory_bandwidth_util_pct_mean",
                    "cpu_cores_mean",
                ):
                    item[metric] = sum(r[metric] * r["elapsed_s"] for r in measured) / elapsed
                for source, target in (
                    ("nvml_peak_used_bytes", "vram_peak_gib"),
                    ("torch_peak_allocated_bytes", "torch_peak_allocated_gib"),
                    ("torch_peak_reserved_bytes", "torch_peak_reserved_gib"),
                    ("host_rss_peak_bytes", "host_rss_peak_gib"),
                ):
                    item[target] = max(r[source] for r in measured) / GIB
                item["vram_occupancy_peak_pct"] = max(
                    r["vram_occupancy_peak_pct"] for r in measured
                )
                item["sample_count"] = sum(r["sample_count"] for r in measured)
                item["sampling_errors"] = sum(r["sampling_errors"] for r in measured)
                if stage == "resident_cli_apply":
                    repeats = sum(r.get("repeats", 0) for r in measured)
                    item["apply_calls"] = repeats
                    if repeats:
                        item["seconds_per_call"] = elapsed / repeats
                        item["input_tokens_per_s"] = (
                            sum(r["repeats"] * r["actual_apply_tokens"] for r in measured) / elapsed
                        )
                summary.append(item)
            rec = {
                "model_id": spec["id"],
                "model_revision": spec["sha"],
                "nominal_size_b": spec["nominal_size_b"],
                "gpu_family": family,
                "status": status["status"],
                "error": status.get("error", ""),
                "dtype": "bfloat16",
                "batch_size": 1,
                "fit_seq_len": 32,
                "apply_seq_len": 128,
                "fit_layers": 3,
                "dim_batch": 8,
                "fit_internal_replicated_batch": 8,
            }
            if passed:
                peak = (
                    max(
                        max(r["nvml_peak_used_bytes"], r["torch_peak_reserved_bytes"])
                        for r in passed
                    )
                    / GIB
                )
                rss = max(r["host_rss_peak_bytes"] for r in passed) / GIB
                rec.update(
                    observed_vram_peak_gib=peak,
                    vram_capacity_with_20pct_free_gib=peak / 0.8,
                    observed_process_rss_peak_gib=rss,
                    process_ram_capacity_with_20pct_free_gib=math.ceil(rss / 0.8),
                    suggested_host_ram_gib=4 * math.ceil((rss / 0.8 + 4) / 4),
                    host_ram_extra_allowance_gib=4,
                    benchmark_cpu_threads=4,
                    suggested_cpu_cores_start=4,
                    gpu_count_tested_per_model=1,
                    gpu_count_recommended=1 if status["status"] == "passed" else "",
                    assigned_job_gpus=2,
                    assigned_job_cpu_cores=47 if family == "H100" else 45,
                    assigned_job_memory_mb=409600 if family == "H100" else 460800,
                    model_weight_storage_gib=spec["safetensors"]["total"] * 2 / GIB,
                    scope="Only measured workload; RAM excludes OS/cache/other processes; "
                    "suggested host RAM adds a heuristic 4 GiB allowance to process RAM; "
                    "CPU is a starting configuration, not an optimized minimum; "
                    "storage is a BF16 estimate; no automatic multi-GPU model sharding",
                )
            recommendations.append(rec)
    write_csv(root / "measurements.csv", raw)
    write_csv(root / "summary.csv", summary)
    write_csv(root / "resource_recommendations.csv", recommendations)
    write_csv(root / "status.csv", statuses)
    comparison = []
    lookup = {(r["model_id"], r["gpu_family"], r["stage"]): r for r in summary}
    for spec in manifest:
        row = {"model_id": spec["id"], "nominal_size_b": spec["nominal_size_b"]}
        for stage, metric, name in (
            ("resident_cli_fit", "elapsed_median_s", "fit"),
            ("resident_cli_apply", "seconds_per_call", "apply"),
        ):
            h = lookup.get((spec["id"], "H100", stage), {}).get(metric)
            b = lookup.get((spec["id"], "B200", stage), {}).get(metric)
            row[f"H100_{name}_s"] = h
            row[f"B200_{name}_s"] = b
            row[f"B200_{name}_speedup_vs_H100"] = h / b if h and b else None
        comparison.append(row)
    write_csv(root / "H100_vs_B200.csv", comparison)
    return summary, statuses


def plot(root, manifest, summary, statuses):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    manifest = sorted(manifest, key=lambda s: (s["id"].rsplit("-", 1)[0], s["nominal_size_b"]))
    names = [s["id"].split("/")[-1] for s in manifest]
    lookup = {(r["model_id"], r["gpu_family"], r["stage"]): r for r in summary}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.subplots_adjust(left=0.105, right=0.96, top=0.89, bottom=0.09, wspace=0.30, hspace=0.32)
    fig.set_facecolor("#f5f7fb")
    panels = [
        ("resident_cli_fit", "elapsed_median_s", "J-lens fit time", "Seconds · median of 2", True),
        (None, "vram_peak_gib", "Peak GPU memory", "GiB · max across measured stages", False),
        (
            "resident_cli_fit",
            "gpu_util_pct_mean",
            "GPU utilization during fitting",
            "% · elapsed-time weighted mean",
            False,
        ),
        (
            "resident_cli_apply",
            "seconds_per_call",
            "J-lens apply latency",
            "Seconds per call · repeated 128-token analysis",
            True,
        ),
    ]
    colors = {"H100": "#2463b5", "B200": "#e17c24"}
    y = np.arange(len(names))
    for ax, (stage, metric, title, xlabel, log) in zip(axes.flat, panels, strict=True):
        ax.set_facecolor("white")
        for offset, family in ((-0.19, "H100"), (0.19, "B200")):
            values = []
            for spec in manifest:
                if stage:
                    value = lookup.get((spec["id"], family, stage), {}).get(metric, math.nan)
                else:
                    candidates = [
                        r[metric]
                        for r in summary
                        if r["model_id"] == spec["id"] and r["gpu_family"] == family
                    ]
                    value = max(candidates, default=math.nan)
                values.append(value)
            ax.barh(y + offset, values, height=0.34, color=colors[family], label=family)
            for yy, value in zip(y + offset, values, strict=True):
                if math.isfinite(value):
                    ax.annotate(
                        f"{value:.2f}" if value < 10 else f"{value:.1f}",
                        (value, yy),
                        xytext=(4, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=8,
                    )
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", weight="bold", pad=12)
        ax.set_xlabel(xlabel + (" · log scale" if log else ""))
        if log:
            ax.set_xscale("log")
            ax.margins(x=0.22)
        elif metric == "gpu_util_pct_mean":
            ax.set_xlim(0, 115)
        else:
            ax.margins(x=0.22)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0, 0].legend(loc="lower right", frameon=False)
    count = sum(s["status"] == "passed" for s in statuses)
    fig.suptitle(
        "Qwen on H100 vs B200 | measured J-lens workload",
        fontsize=19,
        weight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.937,
        f"BF16 · 1 GPU/model · batch 1 · fit: 32 tokens, 3 layers, dim-batch 8"
        f" · {count}/20 complete",
        fontsize=12,
        ha="center",
        color="#4c566a",
    )
    fig.text(
        0.01,
        0.012,
        "Downloads excluded; model loading measured separately. "
        "Apply is activation analysis, not text generation. "
        "Missing bars mean unavailable results. Hardware runs are unpaired.",
        fontsize=10,
        color="#4c566a",
    )
    fig.savefig(root / "qwen_gpu_comparison.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(root / "qwen_gpu_comparison.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    summary, statuses = aggregate(args.root, manifest)
    plot(args.root, manifest, summary, statuses)
    print(
        json.dumps(
            {
                "stages": len(summary),
                "models": len(statuses),
                "passed": sum(s["status"] == "passed" for s in statuses),
            }
        )
    )


if __name__ == "__main__":
    main()
