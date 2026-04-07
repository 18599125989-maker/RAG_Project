import os
import json
import argparse
import requests

API_URL = "https://api.siliconflow.cn/v1/embeddings"


def embed_texts(texts, model, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float"
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()

    embeddings = []
    for item in result["data"]:
        embeddings.append(item["embedding"])

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="例如: data/chunked_json/1_content_ingestion.json")
    parser.add_argument("--output_dir", default="data/embedding_json")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("请先执行: export SILICONFLOW_API_KEY='你的key'")

    with open(args.input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    output_data = []
    batch_items = []
    batch_texts = []

    for item in chunks:
        text = item.get("text", "").strip()
        if not text:
            continue

        batch_items.append(item)
        batch_texts.append(text)

        if len(batch_texts) == args.batch_size:
            embeddings = embed_texts(batch_texts, args.model, api_key)

            for old_item, emb in zip(batch_items, embeddings):
                new_item = {
                    "page_idx": old_item.get("page_idx"),
                    "chunk_idx": old_item.get("chunk_idx"),
                    "text": old_item.get("text"),
                    "embedding": emb
                }
                output_data.append(new_item)

            batch_items = []
            batch_texts = []

    # 处理最后一批
    if batch_texts:
        embeddings = embed_texts(batch_texts, args.model, api_key)

        for old_item, emb in zip(batch_items, embeddings):
            new_item = {
                "page_idx": old_item.get("page_idx"),
                "chunk_idx": old_item.get("chunk_idx"),
                "text": old_item.get("text"),
                "embedding": emb
            }
            output_data.append(new_item)

    output_path = os.path.join(args.output_dir, os.path.basename(args.input_path))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {output_path}")
    print(f"Total input chunks: {len(chunks)}")
    print(f"Total embedded chunks: {len(output_data)}")

    
    for item in output_data[:3]:
        print(
            f"page_idx={item['page_idx']}, "
            f"chunk_idx={item['chunk_idx']}, "
            f"embedding_dim={len(item['embedding'])}"
        )


if __name__ == "__main__":
    main()