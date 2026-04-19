import argparse
from typing import List, Dict, Any

import requests
from openai import OpenAI
from pymilvus import connections, Collection

from src.config import (
    EMBED_API_URL,
    EMBED_API_KEY,
    EMBED_MODEL,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION,
    SILICONFLOW_API_KEY,
    LLM_MODEL,
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


def build_citation(source: str, chunk_id: int, rank: int) -> Dict[str, Any]:
    return {
        "citation_id": rank,
        "citation_mark": f"[{rank}]",
        "source": source,
        "chunk_id": chunk_id,
    }


def build_prompt(query: str, hits: List[Dict[str, Any]]) -> str:
    context_blocks = []

    for item in hits:
        block = (
            f"{item['citation']}\n"
            f"Source: {item['source']}\n"
            f"Chunk ID: {item['chunk_id']}\n"
            f"Text:\n{item['text']}\n"
        )
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    prompt = f"""你是一个金融文档问答助手，请严格根据给定资料回答问题。

要求：
1. 只能依据“资料”回答，不要编造。
2. 如果资料不足以回答，就明确说“根据当前检索到的资料，无法确定”。
3. 回答要尽量简洁、直接。
4. 回答中必须使用引用标记，如 [1]、[2]。
5. 一个结论如果来自多条资料，可以同时引用多个标记，如 [1][2]。

用户问题：
{query}

资料：
{context_text}

请给出最终回答：
"""
    return prompt


def generate_answer(prompt: str) -> str:
    client = OpenAI(
        api_key=SILICONFLOW_API_KEY,
        base_url="https://api.siliconflow.cn/v1",
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
        stream=False,
    )

    return resp.choices[0].message.content.strip()


def run_query(query: str, top_k: int = 5) -> Dict[str, Any]:
    print("[1/5] 生成 query embedding")
    query_vector = get_query_embedding(query)
    print(f"query 向量维度: {len(query_vector)}")

    print(f"[2/5] 连接 Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    collection = connect_milvus()

    print(f"[3/5] 检索 collection: {MILVUS_COLLECTION}")
    results = search_collection(
        collection=collection,
        query_vector=query_vector,
        top_k=top_k,
    )

    print(f"[4/5] 整理 top-{top_k} 结果")
    hits_data = []
    citations = []

    if results and results[0]:
        for rank, hit in enumerate(results[0], start=1):
            entity = hit.entity
            chunk_id = entity.get("chunk_id")
            source = entity.get("source")
            pdf_stem = entity.get("pdf_stem")
            ingest_time = entity.get("ingest_time")
            text = entity.get("text", "")

            citation = build_citation(
                source=source,
                chunk_id=chunk_id,
                rank=rank,
            )

            hit_data = {
                "rank": rank,
                "score": float(hit.score),
                "chunk_id": chunk_id,
                "source": source,
                "pdf_stem": pdf_stem,
                "ingest_time": ingest_time,
                "text": text,
                "citation": citation["citation_mark"],
            }
            hits_data.append(hit_data)

            citations.append({
                "citation_id": citation["citation_id"],
                "citation_mark": citation["citation_mark"],
                "source": citation["source"],
                "chunk_id": citation["chunk_id"],
                "text": text,
            })

    if not hits_data:
        return {
            "query": query,
            "top_k": top_k,
            "hits": [],
            "citations": [],
            "prompt": "",
            "answer": "没有检索到相关结果。",
        }

    prompt = build_prompt(query=query, hits=hits_data)

    print("[5/5] 调用 LLM 生成答案")
    answer = generate_answer(prompt)

    return {
        "query": query,
        "top_k": top_k,
        "hits": hits_data,
        "citations": citations,
        "prompt": prompt,
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser(description="Query Milvus and generate cited answer.")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k retrieval results")
    args = parser.parse_args()

    result = run_query(
        query=args.query,
        top_k=args.top_k,
    )

    print("=" * 80)
    print("Answer:")
    print(result["answer"])
    print()

    print("=" * 80)
    print("Citations:")
    for c in result["citations"]:
        print(f"{c['citation_mark']} {c['source']} / chunk_{c['chunk_id']}")

    print("=" * 80)
    print("完成")


if __name__ == "__main__":
    main()