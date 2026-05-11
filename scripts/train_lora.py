import argparse
import inspect
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
    return records


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Optional JSON config file.")
    known, _ = pre_parser.parse_known_args()

    defaults = {}
    if known.config:
        with Path(known.config).open("r", encoding="utf-8") as f:
            defaults = json.load(f)

    parser = argparse.ArgumentParser(description="Run a small LoRA SFT job.")
    parser.add_argument("--config", default=known.config, help="Optional JSON config file.")
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--train-file", default="data/processed/train.jsonl")
    parser.add_argument("--eval-file", default="data/processed/eval.jsonl")
    parser.add_argument("--output-dir", default="outputs/pythia70m-lora")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="all-linear")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.set_defaults(**defaults)
    return parser.parse_args()


def get_lora_targets(value: str):
    if value == "all-linear":
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def load_tokenizer(model_name: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_batch(batch, tokenizer, max_length: int):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for prompt, response in zip(batch["prompt"], batch["response"]):
        response = response + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = tokenizer(
            prompt + response,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        labels = list(full["input_ids"])
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length

        input_ids_list.append(full["input_ids"])
        attention_mask_list.append(full["attention_mask"])
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


class JsonlSFTDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        records = read_jsonl(path)
        if not records:
            raise ValueError(f"No records found in {path}")
        tokenized = tokenize_batch(
            {
                "prompt": [item["prompt"] for item in records],
                "response": [item["response"] for item in records],
            },
            tokenizer,
            max_length,
        )
        self.features = [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            for input_ids, attention_mask, labels in zip(
                tokenized["input_ids"],
                tokenized["attention_mask"],
                tokenized["labels"],
            )
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return self.features[index]


class DataCollatorForCausalSFT:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_length = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []
        for item in features:
            pad_length = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_length)
            attention_mask.append(item["attention_mask"] + [0] * pad_length)
            labels.append(item["labels"] + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_training_args(args: argparse.Namespace, has_eval: bool) -> TrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "report_to": [],
        "remove_unused_columns": False,
        "fp16": torch.cuda.is_available(),
    }

    signature = inspect.signature(TrainingArguments.__init__)
    strategy_name = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[strategy_name] = "steps" if has_eval else "no"
    if has_eval:
        kwargs["eval_steps"] = args.eval_steps

    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    train_file = Path(args.train_file)
    eval_file = Path(args.eval_file) if args.eval_file else None
    has_eval = bool(eval_file and eval_file.exists())

    tokenizer = load_tokenizer(args.model_name, args.trust_remote_code)
    train_dataset = JsonlSFTDataset(train_file, tokenizer, args.max_length)
    eval_dataset = JsonlSFTDataset(eval_file, tokenizer, args.max_length) if has_eval else None

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=get_lora_targets(args.lora_target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=build_training_args(args, has_eval),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCausalSFT(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
