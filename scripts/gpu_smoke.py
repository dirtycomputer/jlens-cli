#!/usr/bin/env python3
"""Run real CUDA CLI checks on each requested GPU, using a tiny local model."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1", help="visible CUDA device indices")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()

    import torch
    import transformers
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from jlens_cli.legacy import main as cli

    devices = [int(value) for value in args.devices.split(",")]
    if not devices or len(set(devices)) != len(devices):
        parser.error("--devices must contain distinct CUDA indices")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; this test does not fall back to CPU")
    if any(index < 0 or index >= torch.cuda.device_count() for index in devices):
        parser.error("a requested device is outside the visible CUDA device range")
    if args.output.exists():
        parser.error("--output must be a new directory")
    if args.dtype == "bfloat16":
        for index in devices:
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported():
                    parser.error(f"cuda:{index} does not support bfloat16")

    args.output.mkdir(parents=True)
    torch.manual_seed(7)
    model_dir = args.output / "tiny-model"
    vocab = {
        token: i
        for i, token in enumerate(
            ["[UNK]", "[BOS]", "[EOS]", "the", "cat", "sat", "on", "a", "mat", "."]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(model_dir)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    model.save_pretrained(model_dir)
    del model
    report = {
        "status": "running",
        "python": platform.python_version(),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "dtype": args.dtype,
        "devices": [],
        "checks": [],
        "scope": "Tiny random Llama; functional CUDA checks, not a pretrained-model benchmark.",
    }

    def run(command):
        started = time.perf_counter()
        code = cli(command)
        report["checks"].append(
            {"command": command, "exit_code": code, "seconds": time.perf_counter() - started}
        )
        if code:
            raise RuntimeError(f"CLI command failed: {command[0]} (exit {code})")

    lenses = []
    try:
        for shard_index, index in enumerate(devices):
            torch.cuda.set_device(index)
            torch.cuda.reset_peak_memory_stats(index)
            directory = args.output / f"cuda-{index}"
            directory.mkdir()
            corpus = directory / "corpus.txt"
            # Disjoint shards exercise weighted merging after fitting on each GPU.
            corpus.write_text(
                "the cat sat on a mat " + ". " * (shard_index + 1) + "\n",
                encoding="utf-8",
            )
            lens = directory / "lens.pt"
            checkpoint = directory / "checkpoint.pt"
            model_args = [
                "--model",
                str(model_dir),
                "--device",
                f"cuda:{index}",
                "--dtype",
                args.dtype,
                "--local-files-only",
            ]
            fit = [
                "fit",
                *model_args,
                "--prompts",
                str(corpus),
                "--skip-first",
                "0",
                "--max-seq-len",
                "16",
                "--dim-batch",
                "8",
                "--checkpoint",
                str(checkpoint),
                "-o",
                str(lens),
            ]
            run(fit)
            run([*fit, "--resume", "--force"])
            prompt_args = ["--lens", str(lens), "--prompt", "the cat sat on a mat ."]
            result_file = directory / "apply.json"
            run(
                [
                    "apply",
                    *model_args,
                    *prompt_args,
                    "--positions=-2,-1",
                    "--top-k",
                    "3",
                    "-o",
                    str(result_file),
                ]
            )
            result = json.loads(result_file.read_text())
            if len(result["positions"]) != 2 or len(result["layers"]) != 2:
                raise RuntimeError("Unexpected apply output shape")
            run(
                [
                    "render",
                    *model_args,
                    *prompt_args,
                    "--top-k",
                    "3",
                    "-o",
                    str(directory / "slice.html"),
                ]
            )
            torch.cuda.synchronize(index)
            peak = torch.cuda.max_memory_allocated(index)
            if peak <= 0:
                raise RuntimeError(f"No CUDA allocation was recorded on device {index}")
            properties = torch.cuda.get_device_properties(index)
            report["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "peak_allocated_bytes": peak,
                }
            )
            lenses.append(lens)
        merged = args.output / "merged.pt"
        run(["merge", *(str(lens) for lens in lenses), "-o", str(merged)])
        metadata = args.output / "merged.json"
        run(["inspect", "--lens", str(merged), "-o", str(metadata)])
        if json.loads(metadata.read_text())["n_prompts"] != len(devices):
            raise RuntimeError("Merged prompt count does not match the number of shards")
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"CUDA checks passed; report: {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
