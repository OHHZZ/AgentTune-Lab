# AgentTune-Lab

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#环境准备)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-6f42c1)](https://huggingface.co/docs/peft)
[![Base Model](https://img.shields.io/badge/Base-Qwen2.5--0.5B--Instruct-00a67e)](#实验结果)

一个面向 Agent 工具/API 文档问答的轻量级 LoRA 微调项目。项目使用 Hugging Face Transformers 与 PEFT，对 `Qwen/Qwen2.5-0.5B-Instruct` 进行监督微调，让模型更适合回答 SDK、工具参数和 API 文档相关问题。

[English README](README_en.md)

## 项目特点

- 端到端流程：数据下载、JSONL 清洗、固定 train/eval 切分、LoRA 训练、推理与评估。
- Agent 文档问答 Prompt：训练、推理和评估保持统一任务格式。
- 可复现实验：对比基座模型、Prompt、训练步数、LoRA rank 与学习率。
- 轻量仓库：只提交代码、配置和样例数据；模型权重与训练产物本地保存。

## 项目结构

```text
FineTune/
  configs/
    sft_lora.json              # 当前最佳 LoRA 配置
  data/
    raw/
      mini_finetune_qa.jsonl   # 轻量样例数据
    processed/                 # 本地生成，git 忽略
  docs/
    project_plan.md
  scripts/
    download_dataset.py        # 下载 Hugging Face 数据集
    prepare_data.py            # 构造 prompt/response 并切分数据
    train_lora.py              # LoRA SFT 训练
    evaluate.py                # base/adapter 评估
    infer.py                   # 推理脚本
  outputs/                     # adapter、预测文件和日志，git 忽略
```

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# CUDA 12.4 示例；也可以按本机环境安装其他 PyTorch 版本
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models\qwen25-05b-instruct
```

## 数据准备

实验数据来自 `lamini/lamini_docs`，下载后得到 `1,394` 条可用 QA 样本，并按 `test_size=0.2`、`seed=42` 固定切分。

```powershell
python scripts\download_dataset.py --dataset lamini/lamini_docs --output data\raw\lamini_docs.jsonl

python scripts\prepare_data.py `
  --input data\raw\lamini_docs.jsonl `
  --output-dir data\processed\lamini_docs_agent_prompt `
  --test-size 0.2 `
  --seed 42
```

Prompt 模板：

```text
You are an AI agent assistant. Answer the following tool or API documentation question accurately and concisely.

### Documentation Question:
{question}

### Answer:
```

## 训练与评估

训练当前最佳配置：

```powershell
python scripts\train_lora.py --config configs\sft_lora.json
```

评估基座模型：

```powershell
python scripts\evaluate.py `
  --model-name models/qwen25-05b-instruct `
  --eval-file data\processed\lamini_docs_agent_prompt\eval.jsonl `
  --output-file outputs\qwen_base_agent_prompt_eval_50.jsonl `
  --limit 50 `
  --trust-remote-code
```

评估 LoRA adapter：

```powershell
python scripts\evaluate.py `
  --model-name models/qwen25-05b-instruct `
  --adapter-dir outputs/qwen25-05b-lora-agent-prompt-r16-steps300 `
  --eval-file data\processed\lamini_docs_agent_prompt\eval.jsonl `
  --output-file outputs\qwen_lora_agent_prompt_r16_steps300_eval_50.jsonl `
  --limit 50 `
  --trust-remote-code
```

## 实验结果

以下结果基于固定评估集中的 50 条 held-out 样本。文档问答是开放式生成任务，因此 `Avg Token F1` 比 `Exact Match` 更有参考价值。

| 组别 | 对比项 | 基座模型 | Prompt | Steps | LoRA r/alpha | LR | Exact Match | Avg Token F1 |
|:--|:--|:--|:--|--:|:--:|:--:|--:|--:|
| Baseline | Pythia base | Pythia-70M | QA | - | - | - | 0.00 | 0.0478 |
| Baseline | Qwen base | Qwen2.5-0.5B-Instruct | QA | - | - | - | 0.00 | 0.2579 |
| Baseline | Qwen base | Qwen2.5-0.5B-Instruct | Agent docs | - | - | - | 0.00 | 0.2276 |
| A | Prompt + LoRA | Qwen2.5-0.5B-Instruct | Agent docs | 100 | 8/16 | 1e-4 | 0.00 | 0.2960 |
| B | 增加训练步数 | Qwen2.5-0.5B-Instruct | Agent docs | 300 | 8/16 | 1e-4 | - | 0.3276 |
| C | 增大 LoRA rank | Qwen2.5-0.5B-Instruct | Agent docs | 300 | 16/32 | 1e-4 | 0.02 | **0.3343** |
| D | 降低学习率 | Qwen2.5-0.5B-Instruct | Agent docs | 300 | 16/32 | 5e-5 | 0.02 | 0.3247 |

最佳结果为实验 C：`Avg Token F1 = 0.3343`，相比 Agent Prompt 下的 Qwen base `0.2276`，相对提升约 `46.9%`。

## 仓库说明

- `.venv/`、`models/`、`outputs/`、完整下载数据集、日志和 checkpoint 均不会提交。
- 公开仓库只保留可复现代码、轻量样例数据、配置文件和项目文档。
- 当前推荐配置见 [configs/sft_lora.json](configs/sft_lora.json)。
