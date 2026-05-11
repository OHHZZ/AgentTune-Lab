import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = """You are an AI agent assistant. Answer the following tool or API documentation question accurately and concisely.

### Documentation Question:
{question}

### Answer:
"""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the base model or a LoRA adapter.")
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--adapter-dir", default=None, help="Path to a LoRA adapter directory.")
    parser.add_argument("--question", default=None, help="Question to ask. If omitted, starts interactive mode.")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def build_prompt(question: str) -> str:
    if "### Question:" in question and "### Answer:" in question:
        return question
    return PROMPT_TEMPLATE.format(question=question.strip())


def load_model(args: argparse.Namespace):
    tokenizer_path = args.adapter_dir if args.adapter_dir and Path(args.adapter_dir).exists() else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer


def generate_answer(question: str, model, tokenizer, max_new_tokens: int, temperature: float) -> str:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args)

    if args.question:
        print(generate_answer(args.question, model, tokenizer, args.max_new_tokens, args.temperature))
        return

    while True:
        question = input("\nQuestion> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print(generate_answer(question, model, tokenizer, args.max_new_tokens, args.temperature))


if __name__ == "__main__":
    main()
