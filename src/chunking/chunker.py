import os
import json
import argparse


CHUNK_SIZE = 400
OVERLAP = 80


def split_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    text = text.strip()

    if not text:
        return chunks

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_json(input_path, output_dir="data/chunked_json"):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_data = []

    for item in data:
        page_idx = item.get("page_idx")
        text = item.get("text", "")

        chunks = split_text(text)

        for i, chunk_text in enumerate(chunks):
            output_data.append({
                "page_idx": page_idx,
                "chunk_idx": i,
                "text": chunk_text
            })

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, os.path.basename(input_path))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {output_path}")
    print(f"Total chunks: {len(output_data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="例如: data/clean_json/1_content_ingestion.json")
    parser.add_argument("--output_dir", default="data/chunked_json", help="输出目录，默认 data/chunked_json")
    args = parser.parse_args()

    chunk_json(args.input_path, args.output_dir)


if __name__ == "__main__":
    main()