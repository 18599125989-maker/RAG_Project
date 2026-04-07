import argparse
import subprocess
from pathlib import Path


def run_mineru_on_folder(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if not input_path.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_path.glob("*.pdf"))

    if not pdf_files:
        print(f"在 {input_path} 中没有找到 PDF 文件。")
        return

    print(f"共找到 {len(pdf_files)} 个 PDF 文件，开始解析...\n")

    for pdf_file in pdf_files:
        cmd = [
            "mineru",
            "-p",
            str(pdf_file),
            "-o",
            str(output_path),
            "-b",
            "pipeline"
        ]

        print(f"正在处理: {pdf_file.name}")
        print("执行命令:", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
            print(f"完成: {pdf_file.name}\n")
        except subprocess.CalledProcessError as e:
            print(f"处理失败: {pdf_file.name}")
            print(f"错误信息: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch parse PDFs with MinerU.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="输入PDF文件夹路径，例如: data/raw/test0403"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="输出结果文件夹路径，例如: data/mineru_out"
    )

    args = parser.parse_args()

    run_mineru_on_folder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()