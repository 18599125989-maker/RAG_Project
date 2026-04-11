import os
import json
import glob
import argparse
import numpy as np
import faiss


def load_embedding_files(input_dir):
    all_vectors = []
    all_metadata = []

    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            embedding = item.get("embedding")
            text = item.get("text", "")

            if not embedding or not text:
                continue

            all_vectors.append(embedding)
            all_metadata.append({
                "source_file": os.path.basename(json_file),
                "page_idx": item.get("page_idx"),
                "chunk_idx": item.get("chunk_idx"),
                "text": text
            })

    return all_vectors, all_metadata


def build_faiss_index(vectors):
    vectors = np.array(vectors, dtype="float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return index, vectors.shape[0], dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/embedding_json", help="默认读取 data/embedding_json")
    parser.add_argument("--output_dir", default="data/faiss_store", help="默认输出到 data/faiss_store")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    vectors, metadata = load_embedding_files(args.input_dir)

    if not vectors:
        raise ValueError(f"在 {args.input_dir} 中没有找到可用的 embedding 数据")

    index, total_count, dim = build_faiss_index(vectors)

    index_path = os.path.join(args.output_dir, "index.faiss")
    metadata_path = os.path.join(args.output_dir, "metadata.json")

    faiss.write_index(index, index_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved index to: {index_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Total vectors: {total_count}")
    print(f"Embedding dim: {dim}")


if __name__ == "__main__":
    main()