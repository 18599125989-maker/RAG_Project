import argparse
from pymilvus import connections, Collection
from src.config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="source filename in Milvus, e.g. tmpfqhp2wq9.pdf")
    args = parser.parse_args()

    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )

    collection = Collection(MILVUS_COLLECTION)
    collection.load()

    expr = f'source == "{args.source}"'
    result = collection.delete(expr)
    print("delete result:", result)

    collection.flush()
    print("flush done")


if __name__ == "__main__":
    main()