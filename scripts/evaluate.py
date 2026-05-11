import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = prediction.lower().split()
    target_tokens = target.lower().split()
    if not pred_tokens or not target_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(target_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def load_model(model_name: str, adapter_dir: str | None, trust_remote_code: bool):
    tokenizer_path = adapter_dir if adapter_dir and Path(adapter_dir).exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer


def generate(prompt: str, model, tokenizer, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a base model or LoRA adapter on JSONL prompts.")
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--eval-file", default="data/processed/eval.jsonl")
    parser.add_argument("--output-file", default="outputs/eval_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(Path(args.eval_file))
    if args.limit > 0:
        records = records[: args.limit]

    model, tokenizer = load_model(args.model_name, args.adapter_dir, args.trust_remote_code)

    predictions = []
    f1_scores = []
    exact_matches = []

    for item in tqdm(records, desc="evaluating"):
        prediction = generate(item["prompt"], model, tokenizer, args.max_new_tokens)
        target = item["response"].strip()
        exact = prediction.strip() == target
        score = token_f1(prediction, target)
        exact_matches.append(exact)
        f1_scores.append(score)
        predictions.append(
            {
                "prompt": item["prompt"],
                "target": target,
                "prediction": prediction,
                "exact_match": exact,
                "token_f1": score,
            }
        )

    write_jsonl(Path(args.output_file), predictions)
    metrics = {
        "count": len(predictions),
        "exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
        "avg_token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }
    print(json.dumps(metrics, indent=2))
    print(f"Wrote predictions to {args.output_file}")


if __name__ == "__main__":
    main()
