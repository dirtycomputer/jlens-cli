# J++ checkpoint CLI and GPU benchmark

[English](#english) | [简体中文](#简体中文)

## English

Install the project with `pip install -e '.[engine]'` after installing a suitable
PyTorch build. Entry points: `jpplens`, `j-lens++`, `python -m jlens_cli.jpp`.

```bash
jpplens presets
jpplens download --preset qwen3.5-9b --include-model
jpplens inspect --preset qwen3.5-9b
jpplens apply --preset qwen3.5-9b --paper-layers \
  --device cuda:0 --dtype bfloat16 \
  --prompt 'Fact: In humans, the organ that pumps blood through the body has this many chambers: ' \
  --top-k 10 --format text

j-lens++ apply --preset qwen3.6-27b --paper-layers \
  --device cuda:0 --dtype bfloat16 --local-files-only \
  --prompt-file prompt.txt -o out/readout.json
```

Presets pin both model and lens commits. The lens revision is
`237dab6ea0e611d964bf407f13ba1655f1a6ae43`. Models:

| Preset | Base-model revision | Source layers | Paper layers | Lens bytes |
|---|---|---|---|---:|
| qwen3.5-9b | c202236235762e1c871ad0ccb60c8ee5ba337b9a | 0–30 | 4,8,12,16,20,24,28 | 2,080,385,107 |
| qwen3.6-27b | 8f33b98937c6031bb49e713dee3e8e185aafd26f | 0–62 | 8,16,24,32,40,48,56 | 6,606,048,507 |

All source layers are read by default. Use `--paper-layers` for the seven
published depths, or `--layers 4,8,12`. Positions default to the last token;
`--positions=-2,-1` and `--all-positions` are available. Inputs are raw text, with
no chat template. Layer numbers are zero-based block outputs.

The compatible reader loads `parameters.jacobians` with
`torch.load(weights_only=True, mmap=True)`, validates model identity, width,
source layers, target offset and finite FP32 matrices, and computes
`final_norm + LM_head(J_layer @ block_output)`. Selected matrices stay on the
model's device in FP32. Default model precision on supported CUDA devices is BF16.
The backend uses eager attention, no KV cache, and one device per model.

By default, tokens with no Unicode letter/digit are removed from lens rankings.
`--no-readout-filter` disables this. Probabilities always use full-vocabulary
softmax, without renormalization after filtering. Final model predictions are
reported separately without filtering. This exposes intermediate readouts; it
does not generate a completion.

For a local model copy, use `--model /path/to/model
--model-identity Qwen/Qwen3.5-9B`. Identity is a user-supplied assertion, not a
weight hash. For cache relocation, set `HF_HUB_CACHE`; `--cache-dir` controls lens
lookup and both downloads in the `download` command. `--local-files-only`
prohibits model/lens network downloads. JSON uses schema version 1; errors go to
stderr. Existing output files require `--force`.

This is an independent checkpoint-compatible reader, **not the official
`workspace_lens` package**. The code repository linked in the model card
(`github.com/koayon/jpp_lens`) returned HTTP 404 when checked on 2026-11-07.
The published checkpoints contain one merged matrix per layer; no inference
router is required. The CLI does **not** provide J++ fitting, clustering, LRP
backward rules or expert-weight optimization. The existing `jlens fit` is the
original Jacobian-lens algorithm and is not J++ fitting. No Recall@10 reproduction
or agreement with the unavailable official code is claimed.

### Reproduce resource measurements

Run separately on a host with two assigned H100 GPUs and on one with two B200
GPUs. Pre-download both presets into the cache roots first. The benchmark is
offline and places one model on each GPU.

```bash
python scripts/jpp_benchmark.py --family H100 \
  --cache /path/to/base-model-cache --lens-cache /path/to/lens-cache \
  --output out/jpp/H100
python scripts/jpp_benchmark.py --family B200 \
  --cache /path/to/base-model-cache --lens-cache /path/to/lens-cache \
  --output out/jpp/B200
python scripts/jpp_report.py --input out/jpp --output out/jpp-report
```

Batch 1, 128 input tokens, last-position top-10 readout; seven/all source layers;
2 warm-ups and 2 windows of at least 15 seconds per mode. Measurement includes
tokenization, forward pass, transport, CPU logit copies, filtering and top-k.
Model/lens loading, validation and transfers are separate. NVML is sampled every
100 ms; GPU utilization is time-weighted, memory includes the CUDA context, and
PyTorch allocator peaks are recorded independently. Filter setup is measured
once, then reused. The full parser/CLI output is additionally checked against the
resident result. CSV recommendations describe this workload with explicit
headroom, not long-context, batching, training, or saturated serving performance.

Validated on 2026-11-07 with PyTorch 2.12.0 / CUDA 12.9 and Transformers 5.22.1.
All four model/GPU combinations passed, including actual CLI output checks.
The table below is the **all-source-layer** result:

| Model | GPU | Layers | Latency (ms) | Peak resident VRAM (GiB) |
|---|---|---:|---:|---:|
| Qwen3.5-9B | H100 | 31 | 150.4 | 19.93 |
| Qwen3.5-9B | B200 | 31 | 129.2 | 20.20 |
| Qwen3.6-27B | H100 | 63 | 320.7 | 57.65 |
| Qwen3.6-27B | B200 | 63 | 286.5 | 57.92 |

Seven-layer latency was 85.0/78.1 ms for 9B and 153.8/132.3 ms for 27B
(H100/B200). One H100 80GB accommodated each model separately. These results do
not require multiple GPUs for a single model. There were 36 phase measurements
and 10,260 telemetry samples; both two-GPU jobs exited successfully.

## 简体中文

安装项目及 `engine` 依赖后，可使用 `jpplens`、`j-lens++` 或
`python -m jlens_cli.jpp`。上述示例分别展示下载、查看元数据和读出。
9B、27B 预设均固定基础模型和镜头版本；默认分析全部源层，
`--paper-layers` 使用论文给出的七个深度。

默认读取最后一个词元；支持 `--layers`、`--positions=-2,-1`、
`--all-positions`、文本/JSON 输出、离线加载和已有文件保护。
输入不添加聊天模板；层号从 0 开始，读取 block 输出。
本地模型副本需传 `--model-identity` 声明原始 HF 模型 ID。
该声明用于检查元数据匹配，不校验权重内容哈希。

推理按公开格式执行 FP32 Jacobian 变换，再使用模型最终归一化与 LM head。
默认从镜头排名中去掉没有 Unicode 字母或数字的词元；
`--no-readout-filter` 可关闭。概率始终来自完整词表 softmax，
过滤后不重新归一化，最终模型预测单独列出且不过滤。
所选矩阵常驻 GPU；模型默认使用支持设备上的 BF16，单卡运行。

这是公开 checkpoint 的独立兼容实现。模型卡链接的官方源码仓库在
2026-11-07 检查时返回 404，因此没有封装或声称复现完整 J++ 训练流程：
聚类、LRP、专家权重优化尚不提供。公开权重已是每层一张合并矩阵，
推理不需要聚类路由。原来的 `jlens fit` 仍是 J-lens，不是 J++。
本测试不验证论文 Recall@10，也不声称与不可访问的官方代码逐项一致。

GPU 测试覆盖两模型 × H100/B200 四种组合。每种组合测试七层与全层读出，
batch=1、128 tokens、最后位置 top-10，预热两次、正式测两段至少 15 秒。
100 ms 采样 GPU 利用率、显存、主机 RSS 和 CPU；另外记录 PyTorch 显存峰值。
加载和传输独立计时，推理包含 tokenizer、前向、矩阵读出、CPU 拷贝和 top-k。
资源建议只适用于该短上下文推理负载，不能用于估算完整 J++ 训练。

2026-11-07 四组已全部通过（PyTorch 2.12.0 / CUDA 12.9、
Transformers 5.22.1）。上表为全层结果：9B 约 20 GiB、27B 约 58 GiB
显存；按 20% 左右余量，可分别按 32 GiB、80 GiB 单卡容量规划。
正式记录共 36 条阶段测量、10,260 条采样，两个作业均成功结束。

来源：[J++ 模型卡](https://huggingface.co/koayon/jpp-lenses)；
[论文](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace)。
