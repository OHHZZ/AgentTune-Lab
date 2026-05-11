import argparse
import json
import random
from pathlib import Path


PROMPT_TEMPLATE = """You are an AI agent assistant. Answer the following tool or API documentation question accurately and concisely.

### Documentation Question:
{question}

### Answer:
"""



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


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_example(item: dict) -> dict:
    if "question" in item and "answer" in item:
        question = str(item["question"]).strip()
        response = str(item["answer"]).strip()
    elif "instruction" in item and "output" in item:
        instruction = str(item["instruction"]).strip()
        input_text = str(item.get("input", "")).strip()
        question = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
        response = str(item["output"]).strip()
    elif "input" in item and "output" in item:
        question = str(item["input"]).strip()
        response = str(item["output"]).strip()
    elif "prompt" in item and "response" in item:
        prompt = str(item["prompt"]).strip()
        response = str(item["response"]).strip()
        text = prompt + response
        return {"prompt": prompt, "response": response, "text": text}
    else:
        keys = ", ".join(sorted(item.keys()))
        raise ValueError(
            "Each row must contain question/answer, instruction/output, "
            f"input/output, or prompt/response fields. Got: {keys}"
        )

    prompt = PROMPT_TEMPLATE.format(question=question)
    return {"prompt": prompt, "response": response, "text": prompt + response}


def train_eval_split(records: list[dict], test_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 <= test_size < 1:
        raise ValueError("--test-size must be in [0, 1).")
    records = list(records)
    random.Random(seed).shuffle(records)
    if len(records) <= 1 or test_size == 0:
        return records, []

    eval_count = max(1, round(len(records) * test_size))
    eval_count = min(eval_count, len(records) - 1)
    return records[eval_count:], records[:eval_count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare JSONL instruction data for SFT.")
    parser.add_argument("--input", default="data/raw/mini_finetune_qa.jsonl", help="Raw JSONL file.")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for train/eval JSONL files.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction used for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    raw_records = read_jsonl(input_path)
    if not raw_records:
        raise ValueError(f"No records found in {input_path}")

    records = [normalize_example(item) for item in raw_records]
    train_records, eval_records = train_eval_split(records, args.test_size, args.seed)

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "eval.jsonl", eval_records)

    print(f"Loaded {len(raw_records)} examples from {input_path}")
    print(f"Wrote {len(train_records)} train examples to {output_dir / 'train.jsonl'}")
    print(f"Wrote {len(eval_records)} eval examples to {output_dir / 'eval.jsonl'}")
    print("Example prompt:")
    print(train_records[0]["prompt"])


if __name__ == "__main__":
    main()
