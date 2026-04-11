import tempfile
from pathlib import Path
from multiprocessing import freeze_support
import re
import argparse
from urllib.parse import quote

from mineru.cli.common import do_parse
from mineru.data.data_reader_writer import S3DataReader, S3DataWriter
from src.config import AK, SK, ENDPOINT, BUCKET, INPUT_PREFIX, OUTPUT_PREFIX


def build_http_object_url(endpoint: str, bucket_name: str, *parts: str) -> str:
    clean_parts = [part.strip("/") for part in parts if part]
    object_path = "/".join(quote(part, safe="/") for part in clean_parts)
    return f"{endpoint.rstrip('/')}/{bucket_name}/{object_path}"


def rewrite_relative_image_paths(
    file_text: str,
    relative_key: str,
    doc_stem: str,
    endpoint: str,
    bucket: str,
    output_prefix: str,
) -> str:
    parent_key = Path(relative_key).parent.as_posix()
    image_base_url = build_http_object_url(
        endpoint,
        bucket,
        output_prefix,
        doc_stem,
        parent_key,
        "images",
    )

    return re.sub(
        r'(?P<prefix>[\(\'"])images/',
        lambda m: f"{m.group('prefix')}{image_base_url}/",
        file_text,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read PDF from MinIO/S3, parse with MinerU, and upload outputs back."
    )
    parser.add_argument("--pdf", required=True, help="PDF file name under input prefix, e.g. 7.pdf")
    parser.add_argument("--language", default="ch", help="Document language, e.g. ch / en")
    parser.add_argument("--backend", default="pipeline", help="MinerU backend")
    parser.add_argument("--parse_method", default="auto", help="MinerU parse method")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ak = AK
    sk = SK
    endpoint = ENDPOINT
    bucket = BUCKET
    input_prefix = INPUT_PREFIX
    output_prefix = OUTPUT_PREFIX

    input_filename = args.pdf
    language = args.language
    backend = args.backend
    parse_method = args.parse_method

    reader = S3DataReader(
        default_prefix_without_bucket=input_prefix,
        bucket=bucket,
        ak=ak,
        sk=sk,
        endpoint_url=endpoint,
        addressing_style="path",
    )

    writer = S3DataWriter(
        default_prefix_without_bucket=output_prefix,
        bucket=bucket,
        ak=ak,
        sk=sk,
        endpoint_url=endpoint,
        addressing_style="path",
    )

    pdf_bytes = reader.read(input_filename)
    doc_stem = Path(input_filename).stem

    with tempfile.TemporaryDirectory(prefix="mineru_minio_") as tmp_dir:
        do_parse(
            output_dir=tmp_dir,
            pdf_file_names=[doc_stem],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[language],
            backend=backend,
            parse_method=parse_method,
            formula_enable=True,
            table_enable=True,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_model_output=True,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,
        )

        output_root = Path(tmp_dir) / doc_stem

        for local_file in output_root.rglob("*"):
            if not local_file.is_file():
                continue

            relative_key = local_file.relative_to(output_root).as_posix()
            payload = local_file.read_bytes()

            if local_file.suffix.lower() in {".md", ".json"}:
                text = payload.decode("utf-8")
                text = rewrite_relative_image_paths(
                    file_text=text,
                    relative_key=relative_key,
                    doc_stem=doc_stem,
                    endpoint=endpoint,
                    bucket=bucket,
                    output_prefix=output_prefix,
                )
                payload = text.encode("utf-8")

            writer.write(f"{doc_stem}/{relative_key}", payload)
            print(f"uploaded: s3://{bucket}/{output_prefix}/{doc_stem}/{relative_key}")


if __name__ == "__main__":
    freeze_support()
    main()