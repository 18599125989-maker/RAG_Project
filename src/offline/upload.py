from pathlib import Path
import argparse

from mineru.data.data_reader_writer import S3DataWriter
from src.config import AK, SK, ENDPOINT, BUCKET, INPUT_PREFIX


def run_upload(pdf_path: str):
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"Not a file: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files are supported, got: {pdf_path.name}")

    writer = S3DataWriter(
        default_prefix_without_bucket=INPUT_PREFIX,
        bucket=BUCKET,
        ak=AK,
        sk=SK,
        endpoint_url=ENDPOINT,
        addressing_style="path",
    )

    pdf_bytes = pdf_path.read_bytes()
    object_name = pdf_path.name

    writer.write(object_name, pdf_bytes)

    print(f"Uploaded: {pdf_path}")
    print(f"Target: s3://{BUCKET}/{INPUT_PREFIX}/{object_name}")

    return {
        "pdf_path": str(pdf_path),
        "object_name": object_name,
        "s3_path": f"s3://{BUCKET}/{INPUT_PREFIX}/{object_name}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Upload a local PDF file to MinIO bucket/input"
    )
    parser.add_argument(
        "pdf_path",
        help="Local PDF path, e.g. data/raw/4.pdf",
    )
    args = parser.parse_args()

    run_upload(args.pdf_path)


if __name__ == "__main__":
    main()