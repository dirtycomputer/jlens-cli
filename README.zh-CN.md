# J-Lens CLI

将 [Anthropic Jacobian Lens](https://github.com/anthropics/jacobian-lens) 封装为可供人和 AI Agent 调用的命令行工具。提供 lens 拟合、内部表征读取、logit lens 对照、分片合并、JSON/JSONL 输出与交互式 HTML 报告。

这是独立封装，非 Anthropic 官方产品。数值计算直接使用官方 `jlens`，上游固定至 `581d398613e5602a5af361e1c34d3a92ea82ba8e`，不重新实现 Jacobian 算法。

## 安装

需要 Python 3.10+ 和 Git。建议先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装适合本机的 CPU/CUDA 版本。

```bash
git clone https://github.com/dirtycomputer/j-lens-cli.git
cd j-lens-cli
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
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
