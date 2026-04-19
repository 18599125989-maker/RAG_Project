import json
import argparse
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List

import requests
from minio import Minio

from src.config import (
    AK,
    SK,
    ENDPOINT,
    BUCKET,
    OUTPUT_PREFIX,
    EMBED_API_URL,
    EMBED_API_KEY,
    EMBED_MODEL,
)


def build_minio_client() -> Minio:
    secure = ENDPOINT.startswith("https://")
    endpoint_no_scheme = ENDPOINT.replace("http://", "").replace("https://", "")
    return Minio(
        endpoint_no_scheme,
        access_key=AK,
        secret_key=SK,
        secure=secure,
    )


def parse_pdf_stem(pdf_name: str) -> str:
    return Path(pdf_name).stem


def read_json_from_minio(client: Minio, object_name: str) -> Dict[str, Any]:
    response = client.get_object(BUCKET, object_name)
    try:
        data = response.read().decode("utf-8")
        return json.loads(data)
    finally:
        response.close()
        response.release_conn()


def upload_json_to_minio(client: Minio, object_name: str, data: Dict[str, Any]) -> None:
    content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    client.put_object(
        bucket_name=BUCKET,
        object_name=object_name,
        data=BytesIO(content),
        length=len(content),
        content_type="application/json",
    )


def get_embedding(text: str) -> List[float]:
    headers = {
        "Authorization": f"Bearer {EMBED_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": EMBED_MODEL,
        "input": text,
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

def run_embedding(pdf: str):
    pdf_stem = parse_pdf_stem(pdf)

    input_object_name = f"{OUTPUT_PREFIX}/{pdf_stem}/auto/{pdf_stem}_chunks.json"
    output_object_name = f"{OUTPUT_PREFIX}/{pdf_stem}/auto/{pdf_stem}_embeddings.json"

    client = build_minio_client()

    print(f"[1/4] 读取 chunk json: s3://{BUCKET}/{input_object_name}")
    chunk_data = read_json_from_minio(client, input_object_name)

    chunks = chunk_data.get("chunks", [])
    if not chunks:
        raise ValueError(f"No chunks found in s3://{BUCKET}/{input_object_name}")

    print(f"[2/4] 开始生成 embeddings，共 {len(chunks)} 个 chunks")

    embedded_chunks = []
    vector_dim = None

    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "").strip()
        if not text:
            print(f"  - 跳过空 chunk: chunk_id={chunk.get('chunk_id')}")
            continue

        print(f"  - [{i}/{len(chunks)}] embedding chunk_id={chunk.get('chunk_id')}")
        embedding = get_embedding(text)

        if vector_dim is None:
            vector_dim = len(embedding)
            print(f"    向量维度: {vector_dim}")

        embedded_chunk = {
            "chunk_id": chunk.get("chunk_id"),
            "text": text,
            "token_count": chunk.get("token_count"),
            "start_token": chunk.get("start_token"),
            "end_token": chunk.get("end_token"),
            "embedding": embedding,
        }
        embedded_chunks.append(embedded_chunk)

    result = {
        "source_chunks": f"s3://{BUCKET}/{input_object_name}",
        "pdf_name": pdf,
        "embed_model": EMBED_MODEL,
        "vector_dim": vector_dim,
        "total_chunks": len(embedded_chunks),
        "chunks": embedded_chunks,
    }

    print(f"[3/4] 上传 embedding json: s3://{BUCKET}/{output_object_name}")
    upload_json_to_minio(client, output_object_name, result)

    print("[4/4] 完成")
    print(f"共生成 {len(embedded_chunks)} 条 embedding")
    print(f"向量维度: {vector_dim}")
    print(f"已上传: s3://{BUCKET}/{output_object_name}")

    return {
        "pdf_stem": pdf_stem,
        "embedding_json_object_name": output_object_name,
        "total_chunks": len(embedded_chunks),
        "vector_dim": vector_dim,
    }


def main():
    parser = argparse.ArgumentParser(description="Read chunk json from MinIO, generate embeddings, and upload back to MinIO.")
    parser.add_argument("--pdf", required=True, help="PDF name, e.g. 11.pdf")
    args = parser.parse_args()

    run_embedding(args.pdf)


if __name__ == "__main__":
    main()