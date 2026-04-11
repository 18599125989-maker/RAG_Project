import argparse
import base64
import mimetypes
import re
import tempfile
from pathlib import Path

from openai import OpenAI
from mineru.data.data_reader_writer import S3DataReader, S3DataWriter
from src.config import (
    AK,
    SK,
    ENDPOINT,
    BUCKET,
    OUTPUT_PREFIX,
    SILICONFLOW_API_KEY,
    VLM_MODEL,
)


def guess_mime_type(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "image/jpeg"


def image_file_to_data_url(image_path: Path) -> str:
    mime = guess_mime_type(image_path.name)
    image_bytes = image_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def describe_image_with_siliconflow(
    client: OpenAI,
    model: str,
    image_path: Path,
    prompt: str,
) -> str:
    data_url = image_file_to_data_url(image_path)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        temperature=0.2,
        stream=False,
    )
    return resp.choices[0].message.content.strip()


def extract_absolute_image_refs_from_md(md_text: str) -> list[str]:
    image_urls = []

    md_pattern = re.compile(
        r'!\[[^\]]*\]\((https?://[^)\s]+/images/[^)\s]+)\)'
    )
    for m in md_pattern.finditer(md_text):
        image_urls.append(m.group(1))

    html_pattern = re.compile(
        r'src=["\'](https?://[^"\']+/images/[^"\']+)["\']'
    )
    for m in html_pattern.finditer(md_text):
        image_urls.append(m.group(1))

    seen = set()
    result = []
    for url in image_urls:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def build_image_url_to_name(image_urls: list[str]) -> dict[str, str]:
    return {url: Path(url).name for url in image_urls}


def insert_description_below_absolute_image_refs(
    md_text: str,
    image_name_to_desc: dict[str, str],
) -> str:
    lines = md_text.splitlines()
    new_lines = []

    md_pattern = re.compile(
        r'!\[[^\]]*\]\((https?://[^)\s]+/images/([^)\s]+))\)'
    )
    html_pattern = re.compile(
        r'src=["\'](https?://[^"\']+/images/([^"\']+))["\']'
    )

    for line in lines:
        new_lines.append(line)
        inserted = False

        md_match = md_pattern.search(line)
        if md_match:
            image_name = Path(md_match.group(2)).name
            desc = image_name_to_desc.get(image_name)
            if desc:
                new_lines.append("")
                new_lines.append(f"该图片表达了：{desc}")
                new_lines.append("")
                inserted = True

        if not inserted:
            html_match = html_pattern.search(line)
            if html_match:
                image_name = Path(html_match.group(2)).name
                desc = image_name_to_desc.get(image_name)
                if desc:
                    new_lines.append("")
                    new_lines.append(f"该图片表达了：{desc}")
                    new_lines.append("")

    return "\n".join(new_lines) + ("\n" if md_text.endswith("\n") else "")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download md/images from MinIO, describe images with SiliconFlow VLM, and update md."
    )
    parser.add_argument("--pdf", required=True, help="PDF filename, e.g. 7.pdf")
    parser.add_argument("--auto_dir", default="auto", help="MinerU subdir, usually auto")
    parser.add_argument(
        "--prompt",
        default=(
            "请用中文描述这张图片。"
            "如果是图表，请说明图表类型、横纵轴、主要趋势和关键结论；"
            "如果是表格，请概括字段和主要内容；"
            "如果是流程图或结构图，请说明模块、关系与核心含义；"
            "输出为一段简洁但信息充分的话。"
        ),
        help="Prompt for VLM",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    client = OpenAI(
        api_key=SILICONFLOW_API_KEY,
        base_url="https://api.siliconflow.cn/v1",
    )

    reader = S3DataReader(
        default_prefix_without_bucket=OUTPUT_PREFIX,
        bucket=BUCKET,
        ak=AK,
        sk=SK,
        endpoint_url=ENDPOINT,
        addressing_style="path",
    )
    writer = S3DataWriter(
        default_prefix_without_bucket=OUTPUT_PREFIX,
        bucket=BUCKET,
        ak=AK,
        sk=SK,
        endpoint_url=ENDPOINT,
        addressing_style="path",
    )

    pdf_stem = Path(args.pdf).stem
    md_key = f"{pdf_stem}/{args.auto_dir}/{pdf_stem}.md"
    images_dir_key = f"{pdf_stem}/{args.auto_dir}/images"

    with tempfile.TemporaryDirectory(prefix="vlm_md_enrich_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        local_md_path = tmp_dir / f"{pdf_stem}.md"
        local_images_dir = tmp_dir / "images"
        local_images_dir.mkdir(parents=True, exist_ok=True)

        print(f"[1/5] 下载 md: s3://{BUCKET}/{OUTPUT_PREFIX}/{md_key}")
        md_bytes = reader.read(md_key)
        local_md_path.write_bytes(md_bytes)
        md_text = md_bytes.decode("utf-8")

        print("[2/5] 从 md 中解析绝对路径图片链接...")
        image_urls = extract_absolute_image_refs_from_md(md_text)
        if not image_urls:
            print("没有在 md 中找到绝对路径图片引用，程序结束。")
            return

        image_url_to_name = build_image_url_to_name(image_urls)
        image_names = list(image_url_to_name.values())
        print(f"共找到 {len(image_names)} 张图片。")

        print("[3/5] 下载图片到本地临时目录...")
        local_image_paths = {}
        for image_name in image_names:
            image_key = f"{images_dir_key}/{image_name}"
            image_bytes = reader.read(image_key)
            local_image_path = local_images_dir / image_name
            local_image_path.write_bytes(image_bytes)
            local_image_paths[image_name] = local_image_path
            print(f"  已下载: {image_name}")

        print("[4/5] 调用 SiliconFlow VLM 生成描述...")
        image_name_to_desc = {}
        for image_name, local_image_path in local_image_paths.items():
            desc = describe_image_with_siliconflow(
                client=client,
                model=VLM_MODEL,
                image_path=local_image_path,
                prompt=args.prompt,
            )
            image_name_to_desc[image_name] = desc
            print(f"  已完成描述: {image_name}")

        print("[5/5] 修改 md 并上传覆盖...")
        updated_md = insert_description_below_absolute_image_refs(md_text, image_name_to_desc)
        local_md_path.write_text(updated_md, encoding="utf-8")
        writer.write(md_key, updated_md.encode("utf-8"))
        print(f"  已覆盖 md: s3://{BUCKET}/{OUTPUT_PREFIX}/{md_key}")

        print("全部完成。")


if __name__ == "__main__":
    main()