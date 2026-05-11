import argparse
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path


API_BASE = "https://datasets-server.huggingface.co"
SSL_CONTEXT = None


def fetch_json(endpoint: str, params: dict[str, str | int]) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{API_BASE}/{endpoint}?{query}"
    with urllib.request.urlopen(url, timeout=60, context=SSL_CONTEXT) as response:
        return json.loads(response.read().decode("utf-8"))


def get_default_config(dataset: str) -> str:
    payload = fetch_json("splits", {"dataset": dataset})
    splits = payload.get("splits", [])
    if not splits:
        raise ValueError(f"No splits found for dataset {dataset}")
    return splits[0].get("config") or "default"


def get_splits(dataset: str, config: str, requested: list[str] | None) -> list[str]:
    payload = fetch_json("splits", {"dataset": dataset})
    splits = [
        item["split"]
        for item in payload.get("splits", [])
        if item.get("config") == config
    ]
    if not splits:
        raise ValueError(f"No splits found for dataset {dataset} config {config}")
    if requested:
        missing = sorted(set(requested) - set(splits))
        if missing:
            raise ValueError(f"Requested split(s) not found: {', '.join(missing)}")
        return requested
    return splits


def extract_row(row: dict) -> dict:
    if "row" in row and isinstance(row["row"], dict):
        return row["row"]
    return row


def normalize_record(item: dict, question_key: str, answer_key: str) -> dict | None:
    question = item.get(question_key)
    answer = item.get(answer_key)
    if question is None or answer is None:
        return None
    question = str(question).strip()
    answer = str(answer).strip()
    if not question or not answer:
        return None
    return {"question": question, "answer": answer}


def iter_rows(
    dataset: str,
    config: str,
    split: str,
    batch_size: int,
    sleep_seconds: float,
):
    offset = 0
    while True:
        payload = fetch_json(
            "rows",
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": batch_size,
            },
        )
        rows = payload.get("rows", [])
        if not rows:
            break
        for row in rows:
            yield extract_row(row)
        offset += len(rows)
        if len(rows) < batch_size:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset to local question/answer JSONL."
    )
    parser.add_argument("--dataset", default="lamini/lamini_docs")
    parser.add_argument("--config", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--output", default="data/raw/lamini_docs.jsonl")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable TLS certificate verification for local Python installs without CA certs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global SSL_CONTEXT
    if args.no_verify_ssl:
        SSL_CONTEXT = ssl._create_unverified_context()

    config = args.config or get_default_config(args.dataset)
    splits = get_splits(args.dataset, config, args.splits)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    with output.open("w", encoding="utf-8") as f:
        for split in splits:
            for item in iter_rows(
                args.dataset,
                config,
                split,
                args.batch_size,
                args.sleep_seconds,
            ):
                record = normalize_record(item, args.question_key, args.answer_key)
                if record is None:
                    skipped += 1
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
                if args.limit > 0 and total >= args.limit:
                    break
            if args.limit > 0 and total >= args.limit:
                break

    print(f"Dataset: {args.dataset}")
    print(f"Config: {config}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Wrote {total} records to {output}")
    if skipped:
        print(f"Skipped {skipped} rows without {args.question_key}/{args.answer_key}")


if __name__ == "__main__":
    main()
