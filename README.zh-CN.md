# jlens-cli

将 [Anthropic Jacobian Lens](https://github.com/anthropics/jacobian-lens) 封装为可供人和 AI Agent 调用的命令行工具。提供 lens 拟合、内部表征读取、logit lens 对照、分片合并、JSON/JSONL 输出与交互式 HTML 报告。

这是独立封装，非 Anthropic 官方产品。数值计算直接使用官方 `jlens`，上游固定至 `581d398613e5602a5af361e1c34d3a92ea82ba8e`，不重新实现 Jacobian 算法。

0.2 版将 `j-lens-cli` 的能力统一迁入本仓库，并保留两个仓库的 Git 历史。
`jpplens` / `j-lens++` 保留 J++ checkpoint 读取，同时使用相同的 JSON/JSONL 协议，支持批量输入、指定 token 排名跟踪与离线 HTML。
详见[迁移兼容说明](docs/migration-0.2.md)、[Agent 接入](docs/agent-integration.md)和 [J++ GPU 测试](scripts/README.jpp.md)。

```bash
jpplens apply --preset qwen3.5-9b --input requests.jsonl \
  --output-format jsonl --track-token-id 1234 --paper-layers
```

默认 JSON 已改为带 `schema_version/status/command/data` 的 envelope。原 0.1 脚本可使用 `jlens --legacy ...` 或 `jpplens --legacy ...`；`jlens render` 仍提供上游可视化。

## 安装

需要 Python 3.10+ 和 Git。计算功能通过 `engine` extra 安装；基础安装可使用帮助、版本、schema 和 presets。建议先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装适合本机的 CPU/CUDA 版本。

```bash
git clone https://github.com/dirtycomputer/jlens-cli.git
cd jlens-cli
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[engine]'
j-lens doctor
```

`j-lens`、`jlens`、`python -m jlens_cli` 等价。本版本通过 GitHub 安装，尚未发布至 PyPI。

## 最小使用流程

```bash
# 1. 为指定模型拟合 lens
j-lens fit \
  --model Qwen/Qwen3-0.6B \
  --prompts examples/corpus.jsonl \
  --layers 0,4,8,12 --device cuda --dtype float32 \
  --checkpoint out/fit.ckpt.pt --output out/lens.pt

# 2. 查看指定 token 位置的逐层表征，并与 logit lens 对照
j-lens apply \
  --model Qwen/Qwen3-0.6B --lens out/lens.pt \
  --prompt 'The currency used in the country shaped like a boot is' \
  --positions=-2,-1 --top-k 10 --baseline \
  --output out/readout.json

# 3. 导出层 × token 的可交互报告
j-lens visualize \
  --model Qwen/Qwen3-0.6B --lens out/lens.pt \
  --prompt 'The cat sat on the mat.' \
  --output out/report.html
```

HTML 可直接在浏览器打开，点击单元格查看 top-k；无需服务器或 CDN。示例语料仅用于检查流程，不能替代有代表性的预训练风格校准语料。已有与模型权重匹配的预拟合 lens 时，可以直接执行 `apply`。

| 命令 | 功能 |
| --- | --- |
| `fit` | 拟合、断点续算、输入分片 |
| `apply` | 单条或批量读取，输出 JSON/JSONL |
| `visualize` | 导出自包含 HTML |
| `inspect` | 查看 lens 的层、维度、prompt 数及元数据 |
| `merge` | 按有效 prompt 数加权合并互不重叠的分片 |
| `doctor` | 检查依赖和 CUDA/MPS；检查结果中的 `runtime_ready` |
| `schema` | 输出 Agent 可读取的调用契约 |

## 接入 Agent / RSI

```bash
j-lens schema
j-lens apply --help

j-lens apply \
  --model /models/my-model --lens out/lens.pt \
  --input prompts.jsonl --output-format jsonl \
  --positions=-1 --top-k 10
```

输入 JSONL 每行形如 `{"id":"case-1","text":"你的 prompt"}`。模型在一次批量调用中仅加载一次，逐条处理。成功结果在 stdout，日志及错误在 stderr；失败时 stderr 最后一行是结构化 JSON。返回码 `0/1/2/130` 分别表示成功、运行时错误、输入错误和中断。

完整的 Python 工具调用样例见 [examples/agent_tool.py](examples/agent_tool.py)，接口约定见 [docs/agent-integration.md](docs/agent-integration.md)。

## 关键约定

- 层索引从 0 开始，表示 Transformer block 的输出残差。拟合默认选择最终层之前的所有层，目标固定为最终层。
- token 位置以实际编码结果为准，包含 BOS 处理和截断；`-1` 为输入序列最后一个 token，`all` 为所有位置。支持 `0:16:2` 等 Python 切片。
- 输入是原始文本，不自动套用 chat template；不执行自回归生成。
- 概率来自全词表 softmax，top-k 概率之和通常小于 1。`--track-token-id ID` 可重复使用，以跟踪指定词表 token 的完整排名。
- 默认拟合排除前 16 个位置和最后一个位置。过短样本会被跳过；全部过短则失败。
- `.pt` 是最终 lens；`.ckpt.pt` 是拟合中间状态，二者不能混用。保留 `.meta.json`，以便检查模型身份及恢复配置。
- 重复相同拟合命令可续算。已有最终输出时增加 `--force`；主动丢弃旧 checkpoint 并重算则使用 `--no-resume --force`。
- lens 必须与模型权重相匹配。相同隐藏维度并不能证明兼容；模型微调后也需要重新验证或重新拟合。
- J-Lens 输出是表征的词表读出，不是权重梯度或因果贡献分数。接入 RSI 后仍需设计干预实验和任务评测。

## 验证范围

测试使用本地随机初始化的小型 GPT-2，实际运行官方 Jacobian 拟合与反向传播，验证读取数值、分片合并、断点恢复、文件保护、批量输出和 HTML 导出。大型 Qwen/Llama、CUDA、MPS 及多卡性能不属于本次本地验证范围。详见 [验证记录](docs/validation.md)。

完整参数、预拟合 lens 示例及分片方法见 [English README](README.md)。代码使用 Apache-2.0 许可证，保留上游归属；不附带模型权重。

## 保留的 GPU 测试工具

这些脚本使用 legacy API，保留原有测量流程。以下为 0.1 历史 GPU 结果，不代表对 0.2 新接口做过 GPU 验证。

### GPU 冒烟测试

在已安装 CUDA 版 PyTorch 且有两张可见 GPU 的机器上运行：

```bash
python scripts/gpu_smoke.py --devices 0,1 --dtype bfloat16 --output out/gpu-smoke
```

脚本在每张 GPU 上拟合独立语料分片，验证断点续算、镜头分析和 HTML 渲染，
然后合并镜头。`report.json` 记录设备名称、CUDA 分配内存峰值、命令耗时和结果。
CUDA 不可用时会直接报错，不会退回 CPU；输出目录必须是新目录。
两张 GPU 依次处理独立分片，不涉及模型分片或分布式训练。

2026-11-07 已在两张 NVIDIA H100 80GB HBM3 上完成 BF16 实测：
Python 3.11.2、PyTorch 2.12.0（CUDA 12.9）、Transformers 5.22.1，
共 10 次 CLI 调用全部通过，两张卡均记录到实际 CUDA 显存分配。
此测试使用随机权重的微型模型验证功能，不是预训练模型性能或可解释性效果评估。

### 真实 Qwen 模型资源测试

[`scripts/qwen_benchmark.py`](scripts/qwen_benchmark.py) 使用固定版本的 Qwen
权重，在两张独立分配的 H100 或 B200 上测试模型加载、常驻模型拟合和分析，
记录 GPU 利用率、显存、进程内存及 CPU。
[`scripts/qwen_report.py`](scripts/qwen_report.py) 导出测量明细、汇总、
资源建议 CSV 和 PNG／SVG 对比图。配置及测量范围见
[中英文测试说明](scripts/README.qwen.md)。
