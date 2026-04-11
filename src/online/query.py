import argparse
from typing import List

import requests
from pymilvus import connections, Collection

from src.config import (
    EMBED_API_URL,
    EMBED_API_KEY,
    EMBED_MODEL,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION,
)


def get_query_embedding(query: str) -> List[float]:
    headers = {
        "Authorization": f"Bearer {EMBED_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": EMBED_MODEL,
        "input": query,
        "encoding_format": "float",
    }

    response = requests.post(
        EMBED_API_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    result = response.json()
    if "data" not in result or not result["data"]:
        raise ValueError(f"Invalid embedding response: {result}")

    embedding = result["data"][0]["embedding"]
    if not isinstance(embedding, list) or len(embedding) == 0:
        raise ValueError("Embedding is empty or invalid.")

    return embedding


def connect_milvus() -> Collection:
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )

    collection = Collection(MILVUS_COLLECTION)
    collection.load()
    return collection


def search_collection(
    collection: Collection,
    query_vector: List[float],
    top_k: int,
):
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_id", "source", "pdf_stem", "ingest_time", "text"],
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Query Milvus with a natural language question.")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k retrieval results")
    args = parser.parse_args()

    print(f"[1/4] 生成 query embedding")
    query_vector = get_query_embedding(args.query)
    print(f"query 向量维度: {len(query_vector)}")

    print(f"[2/4] 连接 Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    collection = connect_milvus()

    print(f"[3/4] 检索 collection: {MILVUS_COLLECTION}")
    results = search_collection(
        collection=collection,
        query_vector=query_vector,
        top_k=args.top_k,
    )

    print(f"[4/4] 输出 top-{args.top_k} 结果\n")
    if not results or not results[0]:
        print("没有检索到结果。")
        return

    for rank, hit in enumerate(results[0], start=1):
        entity = hit.entity
        chunk_id = entity.get("chunk_id")
        source = entity.get("source")
        pdf_stem = entity.get("pdf_stem")
        ingest_time = entity.get("ingest_time")
        text = entity.get("text", "")

        print("=" * 80)
        print(f"Rank        : {rank}")
        print(f"Score       : {hit.score}")
        print(f"Chunk ID    : {chunk_id}")
        print(f"Source      : {source}")
        print(f"PDF Stem    : {pdf_stem}")
        print(f"Ingest Time : {ingest_time}")
        print("Text:")
        print(text[:1000])
        print()

    print("=" * 80)
    print("检索完成")


if __name__ == "__main__":
    main()