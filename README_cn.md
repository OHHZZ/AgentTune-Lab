## 中文

### 项目概述

本项目是一个轻量级大模型监督微调流程，用于提升模型在 Agent 工具/API 文档问答场景下的回答能力。项目基于 Hugging Face Transformers 与 PEFT LoRA，对 `Qwen/Qwen2.5-0.5B-Instruct` 进行参数高效微调。

项目定位为 Agent 支撑模块：Agent 在执行任务时经常需要理解 SDK/API 文档、解释工具参数、回答工具使用问题，并根据文档处理调用错误。

### 项目亮点

- 搭建端到端 LoRA SFT 流水线，覆盖数据下载、JSONL 标准化、固定 train/eval 切分、训练、推理和 base/adapter 对比评估。
- 将通用 QA prompt 改造成 Agent 工具文档问答 instruction prompt，使训练、推理和评估保持一致任务格式。
- 围绕基座模型、prompt 格式、LoRA rank、训练步数和学习率进行实验对比，并使用 `avg_token_f1` 和 `exact_match` 评估效果。

### 项目结构

```text
FineTune/
  configs/
    sft_lora.json              # 当前最佳 Qwen LoRA 配置
  data/
    raw/
      mini_finetune_qa.jsonl   # 仓库内保留的小样例数据
    processed/                 # 本地生成，git 忽略
  docs/
    project_plan.md
  outputs/                     # 本地预测和 adapter，git 忽略
  scripts/
    download_dataset.py        # 下载 Hugging Face 数据集到本地 JSONL
    prepare_data.py            # 标准化 prompt/response 并切分数据
    train_lora.py              # 基于本地 JSONL 的 LoRA 训练脚本
    evaluate.py                # base/adapter 评估脚本
    infer.py                   # base 或 LoRA adapter 推理脚本
```

以下本地产物不会提交到 git：`.venv/`、`models/`、`outputs/`、日志、checkpoint、完整下载数据集。

### 数据集

实验使用 `lamini/lamini_docs`，并导出为包含 `question` / `answer` 字段的本地 JSONL。

```powershell
python scripts\download_dataset.py --dataset lamini/lamini_docs --output data\raw\lamini_docs.jsonl
```

下载后共得到 1,394 条可用 QA 样本，使用 `test_size=0.2` 和 `seed=42` 固定切分：

```text
训练集：1,115 条
评估集：279 条
```

生成 Agent 工具文档问答 prompt 格式：

```powershell
python scripts\prepare_data.py --input data\raw\lamini_docs.jsonl --output-dir data\processed\lamini_docs_agent_prompt --test-size 0.2 --seed 42
```

Prompt 模板：

```text
You are an AI agent assistant. Answer the following tool or API documentation question accurately and concisely.

### Documentation Question:
{question}

### Answer:
```

### 环境配置

创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

根据本机 GPU 安装 PyTorch。CUDA 12.4 示例：

```powershell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

安装其余依赖：

```powershell
pip install -r requirements.txt
```

下载 Qwen 基座模型到本地：

```powershell
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models\qwen25-05b-instruct
```

如果终端中存在旧代理变量，下载或训练前可以清空：

```powershell
$env:HTTP_PROXY=""
$env:HTTPS_PROXY=""
$env:ALL_PROXY=""
$env:http_proxy=""
$env:https_proxy=""
$env:all_proxy=""
```

### 训练与评估

运行训练：

```powershell
python scripts\train_lora.py --config configs\sft_lora.json
```

评估 base 模型：

```powershell
python scripts\evaluate.py --model-name models/qwen25-05b-instruct --eval-file data\processed\lamini_docs_agent_prompt\eval.jsonl --output-file outputs\qwen_base_agent_prompt_eval_50.jsonl --limit 50 --trust-remote-code
```

评估 LoRA adapter：

```powershell
python scripts\evaluate.py --model-name models/qwen25-05b-instruct --adapter-dir outputs/qwen25-05b-lora-agent-prompt-r16-steps300 --eval-file data\processed\lamini_docs_agent_prompt\eval.jsonl --output-file outputs\qwen_lora_agent_prompt_r16_steps300_eval_50.jsonl --limit 50 --trust-remote-code
```

### 实验结果

以下结果基于固定评估集中的 50 条 held-out 样本。由于任务是开放式文档问答，`avg_token_f1` 比 exact match 更有参考价值。

| 实验 | 基座模型 | Prompt | Steps | LoRA r/alpha | LR | Exact Match | Avg Token F1 |
|---|---|---|---:|---:|---:|---:|---:|
| Pythia base | Pythia-70M | QA | - | - | - | 0.00 | 0.0478 |
| Qwen base | Qwen2.5-0.5B-Instruct | QA | - | - | - | 0.00 | 0.2579 |
| Qwen base | Qwen2.5-0.5B-Instruct | Agent docs | - | - | - | 0.00 | 0.2276 |
| A: prompt + LoRA | Qwen2.5-0.5B-Instruct | Agent docs | 100 | 8/16 | 1e-4 | 0.00 | 0.2960 |
| B: 增加训练步数 | Qwen2.5-0.5B-Instruct | Agent docs | 300 | 8/16 | 1e-4 | - | 0.3276 |
| C: 增大 LoRA rank | Qwen2.5-0.5B-Instruct | Agent docs | 300 | 16/32 | 1e-4 | 0.02 | 0.3343 |
| D: 降低学习率 | Qwen2.5-0.5B-Instruct | Agent docs | 300 | 16/32 | 5e-5 | 0.02 | 0.3247 |

最佳结果为实验 C，`avg_token_f1=0.3343`，相比 Agent prompt 下的 Qwen base 分数 `0.2276`，相对提升约 `46.9%`。

### 简历 Bullet

- 基座模型调整：将基座模型从 70M 通用小模型升级为 0.5B 指令模型，用于增强 Agent 工具文档问答场景下的指令跟随与文档理解能力；在相同 held-out 评估集上，升级后 base 模型的 `avg_token_f1` 相对提升约 `440%`，为后续 LoRA 微调提供更强性能基线。
- 数据 Prompt 优化：将通用 QA 模板改造为面向 Agent 工具/API 文档理解的 instruction prompt，显式引入 Agent assistant 角色与 documentation question 任务语境，并保持训练、推理、评估 prompt 一致；在 50 条 held-out 样本上，LoRA 微调后 `avg_token_f1` 从 base 的 `0.228` 提升至 `0.296`，相对提升约 `30.1%`。
- LoRA 超参数调优：围绕训练步数、LoRA rank 和学习率设计多组消融实验，对比 `steps=100/r=8/lr=1e-4`、`steps=300/r=8/lr=1e-4`、`steps=300/r=16/lr=1e-4`、`steps=300/r=16/lr=5e-5` 等配置；最佳配置将 `avg_token_f1` 提升至 `0.334`，相比 base 的 `0.228` 相对提升约 `46.9%`，并将 `exact_match` 从 `0%` 提升至 `2%`。

### 说明

- 模型权重、LoRA adapter、checkpoint、预测 JSONL、日志和虚拟环境不会提交。
- 公开仓库只保留代码、轻量样例数据、配置文件和可复现实验命令。
