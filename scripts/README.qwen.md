# Reproducible Qwen GPU benchmark

This measures this CLI's Jacobian fitting and activation analysis with real,
revision-pinned Qwen weights. It is not a training or text-generation benchmark.
It does not measure lens quality.

Install the CLI with its engine dependencies, plus `nvidia-ml-py` and `psutil`.
Use CUDA-enabled PyTorch appropriate for the GPU. The initial environment uses
PyTorch 2.12.0 / CUDA 12.9 and Transformers 5.22.1.

Use the bundled `scripts/qwen-manifest.json`, or prepare a JSON manifest:
an array of objects containing `id` (Hugging Face
repository), `sha` (exact revision), `nominal_size_b`, `safetensors.total` (repository
parameter count), and `config` (the repository's config JSON). Preserve the manifest
alongside results. A multimodal repository's parameter count can differ from the
loaded text model count; both are recorded.

```bash
python scripts/qwen_benchmark.py \
  --manifest scripts/qwen-manifest.json --family H100 \
  --output results/H100 --cache /path/to/hf-cache
```

The orchestrator requires exactly two assigned GPUs and runs one independent
process per GPU. Repeat with `--family B200` on two B200 GPUs. Each model fits on
one GPU; this does not introduce multi-GPU sharding. Each model has a 40-minute
timeout; the caller should also enforce a whole-job timeout.

For every model, the benchmark downloads pinned weights before measurement,
loads them once in BF16, performs a forward warmup, fits two lenses, warms up
analysis, and performs two analysis windows (at least 15 seconds and five calls
per window). Only the benchmark's loader is replaced with a cached model to
separate resident computation from model loading. The CLI's computations are
unchanged. The first fit can include initialization costs.

The fitting corpus contains one synthetic prompt truncated to 32 tokens, with
source layers `0`, `n_layers // 2`, and `n_layers - 2`, `skip_first=0`, and
`dim_batch=8`. Upstream replicates that one input eight times for Jacobian
backpropagation. Analysis uses 128 input tokens, the final position, and top-5
results. It includes lens loading, tokenization, JSON creation, and output I/O.
Fitting includes lens output I/O. This small workload calibrates resource use,
not a full fitting corpus or long-context capacity.

Telemetry samples every 100 ms:

- NVML GPU utilization: time with GPU work active, not SM or Tensor Core occupancy.
- NVML memory utilization: memory access activity, not allocated VRAM percentage.
- Device VRAM use and process RSS; CUDA allocator peaks provide an additional check.
- Process CPU use expressed as core equivalents; Torch/BLAS threads are limited to four.

Device-wide NVML memory can include driver/context overhead. Short peaks can fall
between samples. GPU utilization measurements on short stages are approximate.
Loading includes filesystem/cache effects; downloads are recorded separately and
excluded. Results on different machines include differences in CPU, driver, and
storage, not just GPU silicon.

Generate CSVs and a PNG/SVG comparison:

```bash
pip install matplotlib numpy
python scripts/qwen_report.py --root results --manifest scripts/qwen-manifest.json
```

`measurements.csv` preserves individual measurements; `summary.csv` aggregates
successful stages; `status.csv` preserves failures; `resource_recommendations.csv`
uses the maximum observed memory divided by 0.8 to retain 20% free capacity.
This is a workload-specific starting point, not a proven minimum. Process RSS
excludes OS memory and shared filesystem cache. Suggested host RAM adds a
heuristic 4 GiB allowance beyond the process estimate and rounds up to 4 GiB.
CPU recommendations retain the
tested thread setting; no CPU scaling sweep is performed. Queue CPU/RAM packages
are allocations for two concurrent GPU workers and are not per-model requirements.
The script does not extrapolate multi-GPU count for models that fail to fit.

# 中文说明

该测试使用固定版本的真实 Qwen 权重，测量 J-lens 拟合和内部激活分析，
不代表大模型训练、文本生成或镜头解释质量。H100、B200 各使用两张卡，
每张卡独立运行一个模型，不进行模型多卡切分。

模型下载不计入性能耗时；模型加载单独计时。拟合使用一条合成文本、
32 token、三个源层、`dim_batch=8`，上游会把输入复制成内部 batch 8。
分析使用 128 token、最后一个位置、top-5；拟合重复两次，分析重复两个
至少 15 秒且至少五次调用的窗口。分析时间包含镜头加载和 JSON 文件写入。

每 100ms 采集 GPU 利用率、显存使用量、显存访问活动、进程 RSS 和 CPU。
显存访问活动百分比与显存占用率是不同指标；GPU 利用率也不是 SM／Tensor
Core 占用率。显存建议为实测峰值除以 0.8，保留 20% 空闲空间。主机内存
建议只覆盖进程 RSS，另需操作系统、文件缓存及其他进程空间。CPU 采用实测
的四线程起始设置，未测试最低 CPU 核数。更长上下文、更多拟合层或更大
批量需要重新测量，不能直接使用本测试的资源建议。
