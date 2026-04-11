import json
import argparse
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="例如: data/clean_json/4_content_ingestion.json")
    parser.add_argument("--tokenizer_name", default="BAAI/bge-m3")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=True
    )

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_tokens = 0

    for item in data:
        page_idx = item.get("page_idx")
        text = item.get("text", "")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_size = len(token_ids)
        total_tokens += token_size
        print(f"page_id: {page_idx}, token_size: {token_size}")

    print(f"\nTotal tokens: {total_tokens}")

if __name__ == "__main__":
    main()