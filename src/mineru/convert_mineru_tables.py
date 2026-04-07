import json
import argparse
from pathlib import Path
from io import StringIO

import pandas as pd


def df_to_llm_text(df: pd.DataFrame) -> str:
    """
    把 DataFrame 转成更适合大模型读取的结构化文本。
    """
    lines = []

    header = [str(x).strip() for x in df.iloc[0].tolist()]
    lines.append("表头: " + " | ".join(header))

    for i in range(1, len(df)):
        row = [str(x).strip() for x in df.iloc[i].tolist()]
        row_text = " | ".join(row)
        lines.append(f"第{i}行: {row_text}")

    return "\n".join(lines)


def convert_table_body_html_to_text(table_html: str) -> str:
    """
    把 HTML table 转成 LLM 可读文本。
    """
    df = pd.read_html(StringIO(table_html))[0]
    return df_to_llm_text(df)


def process_content_list(content_list: list, drop_failed_html: bool = False) -> tuple[list, int, int]:
    """
    批量处理 content_list 中所有 type == 'table' 的 block。
    返回：
    - 修改后的 content_list
    - 成功转换数量
    - 失败数量
    """
    success_count = 0
    fail_count = 0

    for item in content_list:
        if not isinstance(item, dict):
            continue

        if item.get("type") != "table":
            continue

        table_html = item.get("table_body", "")
        if not table_html:
            fail_count += 1
            if drop_failed_html:
                item["table_body"] = ""
            continue

        try:
            llm_text = convert_table_body_html_to_text(table_html)
            item["table_body"] = llm_text
            success_count += 1
        except Exception as e:
            fail_count += 1
            item["table_body_error"] = str(e)
            if drop_failed_html:
                item["table_body"] = ""

    return content_list, success_count, fail_count


def main():
    parser = argparse.ArgumentParser(description="将 MinerU JSON 中的 table_body 从 HTML 转为 LLM 可读文本")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径")
    parser.add_argument(
        "--drop-failed-html",
        action="store_true",
        help="如果某个表格转换失败，则将其 table_body 置空；默认保留原内容"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, "r", encoding="utf-8") as f:
        content_list = json.load(f)

    if not isinstance(content_list, list):
        raise ValueError("输入 JSON 顶层必须是 list，当前脚本适用于 MinerU 的 content_list.json")

    content_list, success_count, fail_count = process_content_list(
        content_list,
        drop_failed_html=args.drop_failed_html
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(content_list, f, ensure_ascii=False, indent=2)

    print(f"处理完成")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"成功转换 table 数量: {success_count}")
    print(f"失败数量: {fail_count}")


if __name__ == "__main__":
    main()