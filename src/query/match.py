import os
import json
import argparse
import requests
import numpy as np
import faiss

API_URL = "https://api.siliconflow.cn/v1/embeddings"


def embed_query(query, model, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "input": query,
        "encoding_format": "float"
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()

    return result["data"][0]["embedding"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="要检索的问题")
    parser.add_argument("--index_path", default="data/faiss_store/index.faiss")
    parser.add_argument("--metadata_path", default="data/faiss_store/metadata.json")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("请先执行: export SILICONFLOW_API_KEY='你的key'")

    if not os.path.exists(args.index_path):
        raise FileNotFoundError(f"找不到索引文件: {args.index_path}")

    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"找不到元数据文件: {args.metadata_path}")

    # 1. 读取索引
    index = faiss.read_index(args.index_path)

    # 2. 读取 metadata
    with open(args.metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 3. query embedding
    query_embedding = embed_query(args.query, args.model, api_key)
    query_vector = np.array([query_embedding], dtype="float32")

    # 4. faiss 检索
    distances, indices = index.search(query_vector, args.top_k)

    # 5. 输出结果
    print(f"\nQuery: {args.query}")
    print(f"Top {args.top_k} results:\n")

    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < 0 or idx >= len(metadata):
            continue

        item = metadata[idx]

        print(f"Rank: {rank}")
        print(f"Score (L2 distance): {dist}")
        print(f"Source file: {item.get('source_file')}")
        print(f"Page idx: {item.get('page_idx')}")
        print(f"Chunk idx: {item.get('chunk_idx')}")
        print(f"Text: {item.get('text')}")
        print("-" * 80)


if __name__ == "__main__":
    main()