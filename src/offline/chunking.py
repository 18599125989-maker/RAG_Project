import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any

from minio import Minio
from minio.error import S3Error

from src.config import AK, SK, ENDPOINT, BUCKET, OUTPUT_PREFIX


def build_minio_client() -> Minio:
    secure = ENDPOINT.startswith("https://")
    endpoint_no_scheme = ENDPOINT.replace("http://", "").replace("https://", "")
    return Minio(
        endpoint_no_scheme,
        access_key=AK,
        secret_key=SK,
        secure=secure,
    )


def simple_tokenize(text: str) -> List[str]:
    """
    轻量 tokenizer：
    - 中文按单字
    - 英文/数字按连续串
    - 标点单独保留
    """
    pattern = r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+|[^\w\s]"
    return re.findall(pattern, text)


def detokenize(tokens: List[str]) -> str:
    text = ""
    prev_is_english = False

    for token in tokens:
        is_chinese = bool(re.fullmatch(r"[\u4e00-\u9fff]", token))
        is_english = bool(re.fullmatch(r"[a-zA-Z0-9_]+", token))

        if not text:
            text += token
        else:
            if is_chinese:
                text += token
            elif is_english:
                text += (" " + token)
            else:
                text += token

        prev_is_english = is_english

    return text.strip()


def split_markdown_blocks(md_text: str) -> List[str]:
    """
    先按 markdown 结构粗分：
    - 标题单独成块
    - 空行分段
    - code block 尽量整体保留
    """
    lines = md_text.splitlines()
    blocks = []

    current_block = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            current_block.append(line)
            in_code_block = not in_code_block
            continue

        if in_code_block:
            current_block.append(line)
            continue

        if stripped.startswith("#"):
            if current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []
            blocks.append(line.strip())
            continue

        if stripped == "":
            if current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block).strip())

    return [b for b in blocks if b]


def chunk_tokens_with_overlap(tokens: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(tokens[start:end])

        if end == n:
            break

        start = end - overlap

    return chunks


def markdown_to_chunks(md_text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    blocks = split_markdown_blocks(md_text)

    block_token_lists = []
    for block in blocks:
        tokens = simple_tokenize(block)
        if tokens:
            block_token_lists.append(tokens)

    merged_chunks_tokens = []
    current_tokens = []

    for tokens in block_token_lists:
        if len(tokens) > chunk_size:
            if current_tokens:
                merged_chunks_tokens.extend(
                    chunk_tokens_with_overlap(current_tokens, chunk_size, overlap)
                )
                current_tokens = []

            merged_chunks_tokens.extend(
                chunk_tokens_with_overlap(tokens, chunk_size, overlap)
            )
            continue

        if len(current_tokens) + len(tokens) <= chunk_size:
            current_tokens.extend(tokens)
        else:
            merged_chunks_tokens.extend(
                chunk_tokens_with_overlap(current_tokens, chunk_size, overlap)
            )
            current_tokens = tokens[:]

    if current_tokens:
        merged_chunks_tokens.extend(
            chunk_tokens_with_overlap(current_tokens, chunk_size, overlap)
        )

    chunks = []
    global_start = 0

    for idx, token_list in enumerate(merged_chunks_tokens):
        token_count = len(token_list)
        chunks.append({
            "chunk_id": idx,
            "text": detokenize(token_list),
            "token_count": token_count,
            "start_token": global_start,
            "end_token": global_start + token_count
        })
        global_start += max(token_count - overlap, 0)

    return chunks


def parse_pdf_stem(pdf_name: str) -> str:
    """
    11.pdf -> 11
    """
    return Path(pdf_name).stem


def read_text_from_minio(client: Minio, object_name: str) -> str:
    response = client.get_object(BUCKET, object_name)
    try:
        return response.read().decode("utf-8")
    finally:
        response.close()
        response.release_conn()


def upload_json_to_minio(client: Minio, object_name: str, data: Dict[str, Any]) -> None:
    content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    from io import BytesIO
    client.put_object(
        bucket_name=BUCKET,
        object_name=object_name,
        data=BytesIO(content),
        length=len(content),
        content_type="application/json",
    )


def main():
    parser = argparse.ArgumentParser(description="Chunk markdown from MinIO and save chunk json back to MinIO.")
    parser.add_argument("--pdf", required=True, help="PDF name, e.g. 11.pdf")
    parser.add_argument("--chunk_size", type=int, required=True, help="Chunk size")
    parser.add_argument("--overlap", type=int, required=True, help="Chunk overlap")

    args = parser.parse_args()

    pdf_stem = parse_pdf_stem(args.pdf)

    md_object_name = f"{OUTPUT_PREFIX}/{pdf_stem}/auto/{pdf_stem}.md"
    output_json_object_name = f"{OUTPUT_PREFIX}/{pdf_stem}/auto/{pdf_stem}_chunks.json"

    client = build_minio_client()

    print(f"[1/4] 读取 MinIO 中的 md: s3://{BUCKET}/{md_object_name}")
    md_text = read_text_from_minio(client, md_object_name)

    print("[2/4] 开始 chunking ...")
    chunks = markdown_to_chunks(
        md_text=md_text,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    result = {
        "source_md": f"s3://{BUCKET}/{md_object_name}",
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }

    print(f"[3/4] 上传 chunk json 到 MinIO: s3://{BUCKET}/{output_json_object_name}")
    upload_json_to_minio(client, output_json_object_name, result)

    print("[4/4] 完成")
    print(f"共生成 {len(chunks)} 个 chunks")
    print(f"已上传: s3://{BUCKET}/{output_json_object_name}")


if __name__ == "__main__":
    main()