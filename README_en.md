# Agent Documentation QA LoRA Fine-tuning

## English

### Overview

This project is a lightweight supervised fine-tuning pipeline for improving an LLM on Agent-oriented tool/API documentation question answering. It uses Hugging Face Transformers and PEFT LoRA to fine-tune `Qwen/Qwen2.5-0.5B-Instruct` on Lamini documentation QA data.

The project is framed as an Agent support component: an Agent often needs to understand SDK/API documentation, explain tool parameters, answer usage questions, and recover from tool-call errors based on documentation.

### Highlights

- Built an end-to-end LoRA SFT workflow covering dataset download, JSONL normalization, fixed train/eval split, training, inference, and base-vs-adapter evaluation.
- Reformatted the prompt into an Agent documentation QA instruction template so training, inference, and evaluation share the same task framing.
- Compared base model choice, prompt format, LoRA rank, training steps, and learning rate using `avg_token_f1` and `exact_match`.

### Project Structure

```text
FineTune/
  configs/
    sft_lora.json              # Current best Qwen LoRA config
  data/
    raw/
      mini_finetune_qa.jsonl   # Small sample data committed to the repo
    processed/                 # Generated locally, ignored by git
  docs/
    project_plan.md
  outputs/                     # Local predictions/adapters, ignored by git
  scripts/
    download_dataset.py        # Download HF dataset to local JSONL
    prepare_data.py            # Normalize prompt/response JSONL and split data
    train_lora.py              # Local JSONL LoRA training
    evaluate.py                # Base/adapter evaluation
    infer.py                   # Inference with base model or LoRA adapter
```

Large local artifacts are intentionally excluded from git: `.venv/`, `models/`, `outputs/`, logs, checkpoints, and downloaded full datasets.

### Dataset

The experiments use `lamini/lamini_docs`, exported to local JSONL with `question` and `answer` fields.

```powershell
python scripts\download_dataset.py --dataset lamini/lamini_docs --output data\raw\lamini_docs.jsonl
```

The downloaded dataset contained 1,394 usable QA records. It was split with `test_size=0.2` and `seed=42`:

```text
train: 1,115 examples
eval: 279 examples
```

Prepare the Agent documentation QA prompt format:

```powershell
python scripts\prepare_data.py --input data\raw\lamini_docs.jsonl --output-dir data\processed\lamini_docs_agent_prompt --test-size 0.2 --seed 42
```

Prompt template:

```text
You are an AI agent assistant. Answer the following tool or API documentation question accurately and concisely.

### Documentation Question:
{question}

### Answer:
```

### Environment

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install CUDA PyTorch according to your GPU. Example for CUDA 12.4:

```powershell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

Download the Qwen base model locally:

```powershell
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models\qwen25-05b-instruct
```

If your shell has stale proxy variables, clear them before downloading/training:

```powershell
$env:HTTP_PROXY=""
$env:HTTPS_PROXY=""
$env:ALL_PROXY=""
$env:http_proxy=""
$env:https_proxy=""
$env:all_proxy=""
```

### Training And Evaluation

Run training:

```powershell
python scripts\train_lora.py --config configs\sft_lora.json
```

Run base evaluation:

```powershell
python scripts\evaluate.py --model-name models/qwen25-05b-instruct --eval-file data\processed\lamini_docs_agent_prompt\eval.jsonl --output-file outputs\qwen_base_agent_prompt_eval_50.jsonl --limit 50 --trust-remote-code
```

Run adapter evaluation:

```powershell
python scripts\evaluate.py --model-name models/qwen25-05b-instruct --adapter-dir outputs/qwen25-05b-lora-agent-prompt-r16-steps300 --eval-file data\processed\lamini_docs_agent_prompt\eval.jsonl --output-file outputs\qwen_lora_agent_prompt_r16_steps300_eval_50.jsonl --limit 50 --trust-remote-code
```

### Experiment Results

Evaluation uses 50 held-out examples from the fixed eval split. The task is open-ended documentation QA, so `avg_token_f1` is more informative than exact match.

| Experiment | Base Model | Prompt | Steps | LoRA r/alpha | LR | Exact Match | Avg Token F1 |
|---|---|---|---:|---:|---:|---:|---:|
| Pythia base | Pythia-70M | QA | - | - | - | 0.00 | 0.0478 |
| Qwen base | Qwen2.5-0.5B-Instruct | QA | - | - | - | 0.00 | 0.2579 |
| Qwen base | Qwen2.5-0.5B-Instruct | Agent docs | - | - | - | 0.00 | 0.2276 |
| A: prompt + LoRA | Qwen2.5-0.5B-Instruct | Agent docs | 100 | 8/16 | 1e-4 | 0.00 | 0.2960 |
| B: more steps | Qwen2.5-0.5B-Instruct | Agent docs | 300 | 8/16 | 1e-4 | - | 0.3276 |
| C: larger LoRA | Qwen2.5-0.5B-Instruct | Agent docs | 300 | 16/32 | 1e-4 | 0.02 | 0.3343 |
| D: lower LR | Qwen2.5-0.5B-Instruct | Agent docs | 300 | 16/32 | 5e-5 | 0.02 | 0.3247 |

Best result: Experiment C reached `avg_token_f1=0.3343`, a relative improvement of about `46.9%` over the Agent-prompt Qwen base score `0.2276`.

### Resume Bullets

- Base model upgrade: Replaced a 70M general-purpose small model with a 0.5B instruction-tuned model to improve instruction following and documentation understanding for Agent tool/API QA; on the same held-out evaluation split, the base `avg_token_f1` improved by about `440%`.
- Prompt optimization: Reframed the generic QA template as an Agent tool/API documentation instruction prompt and kept the prompt consistent across training, inference, and evaluation; after LoRA fine-tuning, `avg_token_f1` improved from `0.228` to `0.296` on 50 held-out samples, a relative gain of about `30.1%`.
- LoRA hyperparameter tuning: Ran ablations over training steps, LoRA rank, and learning rate, comparing `steps=100/r=8/lr=1e-4`, `steps=300/r=8/lr=1e-4`, `steps=300/r=16/lr=1e-4`, and `steps=300/r=16/lr=5e-5`; the best setting reached `avg_token_f1=0.334`, a `46.9%` relative improvement over the base score, and improved `exact_match` from `0%` to `2%`.

### Notes

- Model weights, LoRA adapters, checkpoints, prediction JSONL files, logs, and virtual environments are not committed.
- The public repository contains only code, lightweight sample data, configuration, and reproducible commands.
