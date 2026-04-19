import argparse
from pathlib import Path
from src.offline.upload import run_upload
from src.offline.parsing_and_imagepath_renew import run_parsing_and_imagepath_renew
from src.offline.describe_image_byvlm import run_describe_image_byvlm
from src.offline.chunking import run_chunking
from src.offline.embedding import run_embedding
from src.offline.indexing import run_indexing

def run_offline_pipeline(
    pdf_path: str,
    chunk_size: int,
    overlap: int,
    drop_old: bool = False,
):
    pdf_name = Path(pdf_path).name   

    print("=== [1/6] upload ===")
    upload_result = run_upload(pdf_path)

    print("=== [2/6] parsing ===")
    parsing_result = run_parsing_and_imagepath_renew(pdf=pdf_name)

    print("=== [3/6] describe image ===")
    describe_result = run_describe_image_byvlm(pdf=pdf_name)

    print("=== [4/6] chunking ===")
    chunking_result = run_chunking(
        pdf=pdf_name,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    print("=== [5/6] embedding ===")
    embedding_result = run_embedding(pdf_name)

    print("=== [6/6] indexing ===")
    indexing_result = run_indexing(
        pdf=pdf_name,
        drop_old=drop_old,
    )

    return {
        "upload": upload_result,
        "parsing": parsing_result,
        "describe": describe_result,
        "chunking": chunking_result,
        "embedding": embedding_result,
        "indexing": indexing_result,
    }

def main():
    parser = argparse.ArgumentParser(description="Run full offline pipeline.")
    parser.add_argument("--pdf_path", required=True, help="Local PDF path, e.g. data/raw/12.pdf")
    parser.add_argument("--chunk_size", type=int, required=True, help="Chunk size")
    parser.add_argument("--overlap", type=int, required=True, help="Chunk overlap")
    parser.add_argument("--drop_old", action="store_true", help="Drop old Milvus collection if needed")

    args = parser.parse_args()
    
    result = run_offline_pipeline(
        pdf_path=args.pdf_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        drop_old=args.drop_old,
    )

    print("\n=== offline done ===")
    print(result)


if __name__ == "__main__":
    main()