import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timezone

from minio import Minio
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from src.config import (
    AK,
    SK,
    ENDPOINT,
    BUCKET,
    OUTPUT_PREFIX,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION,
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


def connect_milvus() -> None:
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def recreate_collection_if_needed(collection_name: str, drop_old: bool) -> None:
    if utility.has_collection(collection_name):
        if drop_old:
            print(f"检测到旧 collection，删除后重建: {collection_name}")
            utility.drop_collection(collection_name)
        else:
            print(f"collection 已存在，将直接复用: {collection_name}")


def get_or_create_collection(collection_name: str, vector_dim: int) -> Collection:
    if utility.has_collection(collection_name):
        return Collection(collection_name)

    print(f"collection 不存在，开始创建: {collection_name}")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="pdf_stem", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="ingest_time", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Financial RAG collection with metadata",
        enable_dynamic_field=False,
    )

    collection = Collection(name=collection_name, schema=schema)
    print(f"collection 创建完成: {collection_name}")
    return collection


def build_index_if_needed(collection: Collection) -> None:
    if collection.indexes:
        print("已存在索引，跳过建索引")
        return

    print("开始创建索引 ...")
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("索引创建完成")


def prepare_insert_data(
    embeddings_json: Dict[str, Any],
    pdf_name: str,
) -> List[List[Any]]:
    chunks = embeddings_json.get("chunks", [])
    if not chunks:
        raise ValueError("No embedded chunks found.")

    chunk_ids = []
    sources = []
    pdf_stems = []
    ingest_times = []
    texts = []
    embeddings = []

    pdf_stem = Path(pdf_name).stem
    current_ingest_time = utc_now_str()

    for item in chunks:
        embedding = item.get("embedding")
        text = item.get("text", "")
        chunk_id = item.get("chunk_id")

        if embedding is None or not isinstance(embedding, list) or len(embedding) == 0:
            print(f"跳过无效 embedding: chunk_id={chunk_id}")
            continue

        if chunk_id is None:
            raise ValueError("chunk_id is missing in embedded chunk.")

        chunk_ids.append(int(chunk_id))
        sources.append(pdf_name)
        pdf_stems.append(pdf_stem)
        ingest_times.append(current_ingest_time)
        texts.append(text[:65535])
        embeddings.append(embedding)

    return [chunk_ids, sources, pdf_stems, ingest_times, texts, embeddings]

def run_indexing(pdf: str, drop_old: bool = False):
    pdf_stem = parse_pdf_stem(pdf)
    input_object_name = f"{OUTPUT_PREFIX}/{pdf_stem}/auto/{pdf_stem}_embeddings.json"

    print(f"[1/7] 读取 embedding json: s3://{BUCKET}/{input_object_name}")
    minio_client = build_minio_client()
    embeddings_json = read_json_from_minio(minio_client, input_object_name)

    vector_dim = embeddings_json.get("vector_dim")
    if not vector_dim:
        raise ValueError("vector_dim not found in embedding json.")

    print(f"[2/7] 连接 Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    connect_milvus()

    print(f"[3/7] 检查 collection: {MILVUS_COLLECTION}")
    recreate_collection_if_needed(MILVUS_COLLECTION, drop_old)

    print(f"[4/7] 获取或创建 collection: {MILVUS_COLLECTION}")
    collection = get_or_create_collection(MILVUS_COLLECTION, vector_dim)

    print("[5/7] 准备插入数据")
    data = prepare_insert_data(embeddings_json, pdf)
    row_count = len(data[0])
    if row_count == 0:
        raise ValueError("No valid rows to insert.")

    print(f"准备插入 {row_count} 条数据")

    print("[6/7] 插入数据到 Milvus")
    insert_result = collection.insert(data)
    print(f"插入完成，insert count: {insert_result.insert_count}")

    print("开始 flush ...")
    collection.flush()
    print("flush 完成")

    build_index_if_needed(collection)

    print("[7/7] load collection")
    collection.load()

    fresh_collection = Collection(MILVUS_COLLECTION)
    fresh_collection.load()

    print("完成")
    print(f"collection: {MILVUS_COLLECTION}")
    print(f"num_entities: {fresh_collection.num_entities}")
    print(f"indexes: {fresh_collection.indexes}")

    return {
        "pdf_stem": pdf_stem,
        "collection_name": MILVUS_COLLECTION,
        "insert_count": insert_result.insert_count,
        "num_entities": fresh_collection.num_entities,
    }


def main():
    parser = argparse.ArgumentParser(description="Read embeddings json from MinIO and insert into Milvus.")
    parser.add_argument("--pdf", required=True, help="PDF name, e.g. 11.pdf")
    parser.add_argument(
        "--drop_old",
        action="store_true",
        help="Drop existing collection before recreate. Use this once when schema changes.",
    )
    args = parser.parse_args()

    run_indexing(
        pdf=args.pdf,
        drop_old=args.drop_old,
    )


if __name__ == "__main__":
    main()