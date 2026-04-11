import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="例如: data/clean_json/4_content_ingestion.json")
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_chars = 0

    for item in data:
        page_idx = item.get("page_idx")
        text = item.get("text", "")
        char_size = len(text)
        total_chars += char_size
        print(f"page_id: {page_idx}, char_size: {char_size}")

    print(f"\nTotal chars: {total_chars}")

if __name__ == "__main__":
    main()