> Historical jlens-cli 0.1 documentation. Use `--legacy` for these schemas and fitting semantics. See [0.2 migration](migration-0.2.md).

# jlens-cli

[English](legacy-cli.md) | **简体中文**

将 [Anthropic Jacobian lens（J-lens）](https://github.com/anthropics/jacobian-lens)
封装成命令行：读取模型内部激活对应的词元、拟合镜头、合并结果、导出交互式 HTML。

**这是非官方封装，与 Anthropic 无隶属关系。** 数值计算直接使用上游实现，固定在
`581d398613e5602a5af361e1c34d3a92ea82ba8e`，没有重新实现研究算法。
此工具针对本地/Hugging Face 开放权重模型，不读取 Claude API 的内部激活。

包内同时提供 `jpplens` / `j-lens++`，用于兼容读取公开的
[J++ lens checkpoint](https://huggingface.co/koayon/jpp-lenses) 并进行推理。
详见 [J++ 使用方法和 GPU 测试说明](../scripts/README.jpp.md)。

## 安装

建议 Python 3.12；最低 Python 3.10。需要 Git 以安装固定版本的上游依赖。
请在仓库目录中执行以下命令：

```bash
python -m venv .venv
source .venv/bin/activate
# CPU 环境先安装 CPU 版 PyTorch；GPU 环境按 pytorch.org 选择对应版本。
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[engine]'
jlens --help
```

`jlens`、`j-lens` 和 `python -m jlens_cli` 等价。仅安装 `pip install -e .`
即可使用帮助、版本与 `presets`；计算命令需要 `engine` 依赖。

## 快速开始

以下预设来自上游 notebook；首次运行会下载模型与镜头，需要足够内存/显存。
4B/27B 模型不是小型示例，CPU 可以运行但会很慢。预设路径不代表已在当前环境
下载或验证这些大型模型。

```bash
jlens presets

jlens apply --preset qwen3.5-4b \
  --prompt 'Fact: The currency used in the country shaped like a boot is' \
  --positions=-2 --top-k 5 --format text

jlens render --preset qwen3.5-4b \
  --prompt 'Fact: The currency used in the country shaped like a boot is' \
  --layer-stride 2 --mask-display -o out/slice.html
```

`render` 直接使用上游可视化模板，生成时从 jsDelivr 下载并校验 D3，
输出包含 D3 与数据的单文件 HTML，可离线在浏览器中打开。
`--local-files-only` 不影响这次 D3 下载。默认最多追踪 100 个高排名词元；长文本可使用
`--last-n-tokens 64 --max-tracked 50` 控制输出体积。

## 自定义模型与镜头

```bash
jlens apply \
  --model /path/to/hf-model \
  --lens /path/to/lens.pt \
  --prompt-file prompt.txt \
  --layers 0,4,8 --positions=-3,-1 \
  --device cuda --dtype bfloat16 -o out/result.json

cat prompt.txt | jlens apply \
  --model /path/to/hf-model --lens /path/to/lens.pt \
  --prompt-file - --all-positions
```

镜头也可以从 Hub 下载：`--lens org/repo --lens-file folder/lens.pt
--lens-revision COMMIT`。模型可通过 `--model-revision COMMIT` 固定版本。
`--local-files-only` 让 apply/render/fit 仅从本地文件或缓存加载模型、tokenizer 与镜头。

输入按原始文本编码，不自动添加聊天模板。位置指的是截断后的词元序列，
包括 tokenizer 添加的 BOS 等特殊词元；
默认只分析最后一个词元，`--positions=-2,-1` 的等号可避免参数歧义。
层索引为零起点。`apply --layers` 使用非负索引，`fit --layers` 也允许负索引。
默认使用镜头中全部拟合层；模型最终输出单独列出。

JSON 输出到 stdout，日志输出到 stderr；`--format text` 可读性更强。
`--logit-lens` 切换为未经过 Jacobian 变换的基线，仍需提供镜头以确定默认层。
返回的概率是在完整词表上计算的 softmax；展示的 top-k 概率之和不必为 1。

输出格式 `schema_version: 1`：

```text
model, lens, method
input_token_ids                    截断后的完整输入
positions, input_tokens            所选位置与对应文本
layers["层索引"][位置偏移][排名]     token_id, token, probability
model_output[位置偏移][排名]         模型最终层的预测
```

嵌套数组中的“位置偏移”指所选位置列表中的偏移，不是绝对词元索引。

## 拟合、续算与合并

语料可以是 UTF-8 `.txt`（每个非空行一个 prompt），也可以是 `.jsonl`
（每行 `{"text": "..."}`）。上游默认跳过前 16 个词元与最后一个位置，
因此每条 prompt 至少需要 18 个词元。过短的 prompt 会被跳过；全部过短会报错。

```bash
jlens fit --model /path/to/hf-model \
  --prompts corpus.jsonl --limit 100 \
  --dim-batch 8 --max-seq-len 128 \
  --checkpoint out/fit.ckpt.pt -o out/lens.pt

# 使用同一条命令并添加 --resume；输出若已存在，另加 --force。
jlens fit --model /path/to/hf-model \
  --prompts corpus.jsonl --limit 100 \
  --dim-batch 8 --max-seq-len 128 \
  --checkpoint out/fit.ckpt.pt --resume -o out/lens.pt

jlens inspect --lens out/lens.pt
jlens merge out/shard-a.pt out/shard-b.pt -o out/merged.pt
```

Checkpoint 与最终镜头格式不同，不能混用。CLI 额外写入 `.cli.json` 元数据，
续算时验证语料哈希、模型标识和关键参数；请保留此文件。要复现实验，
使用固定的模型 revision，不要在续算期间修改本地模型目录。
不同分片必须使用同一模型、层配置与互不重叠的语料；合并按成功 prompt 数加权。
镜头文件本身不记录模型身份，维度检查无法识别同尺寸但不同模型的误配。
CLI 最终镜头以 float32 保存，避免合并时额外降低精度。

默认不会覆盖已有输出，需显式使用 `--force`。新拟合也不会覆盖已有 checkpoint；
请选择新路径或使用 `--resume`。`--force` 仅控制最终输出。

## 运行边界

- `--device auto` 选择 CUDA、MPS 或 CPU；不进行多 GPU 分片。大型模型需能装入单个设备。
- `--dtype auto` 在支持 BF16 的 CUDA 上使用 BF16，其余情况使用 float32。
- 使用 eager attention，以兼容 Jacobian 拟合所需的反向传播。
- Hugging Face 自定义代码默认关闭，需显式传 `--trust-remote-code`。
- 模型架构兼容性遵循上游 `jlens.from_hf`；并非所有 HF 模型都支持。
- 退出码：成功 0，运行错误 1，参数错误 2，中断 130；`jlens --debug ...` 显示堆栈。
- 上游标注为参考实现、不维护。PyTorch/Transformers 由上游声明约束，
  未来版本变更可能需要重新验证。

## 开发与验证

```bash
pip install -e '.[engine,dev]'
ruff check .
ruff format --check .
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q
python -m build
```

测试在本地生成一个随机权重的微型 Llama 与 tokenizer，实际执行拟合、
checkpoint 续算、top-k 分析、按权重合并和 HTML 渲染，不下载模型；
渲染测试需要联网取得 D3。
此测试验证 CLI 与上游接口的连通性，不验证大型预训练模型的可解释性效果。
GitHub Actions 在 Python 3.10 与 3.12 上运行相同流程。

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

[`scripts/qwen_benchmark.py`](../scripts/qwen_benchmark.py) 使用固定版本的 Qwen
权重，在两张独立分配的 H100 或 B200 上测试模型加载、常驻模型拟合和分析，
记录 GPU 利用率、显存、进程内存及 CPU。
[`scripts/qwen_report.py`](../scripts/qwen_report.py) 导出测量明细、汇总、
资源建议 CSV 和 PNG／SVG 对比图。配置及测量范围见
[中英文测试说明](../scripts/README.qwen.md)。

## 许可证与来源

Apache-2.0，见 [LICENSE](../LICENSE) 与 [NOTICE](../NOTICE)。
研究论文：[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)。
仓库不附带模型权重、镜头文件或训练语料；运行时下载资源受其各自许可证约束。
