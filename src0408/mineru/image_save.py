import json
import subprocess
from pathlib import Path
from urllib.parse import quote

# ======== 交互输入 ========
json_path = Path(input("请输入 json_path: ").strip()).expanduser()
images_dir = Path(input("请输入 images_dir: ").strip()).expanduser()

# ======== 可按需修改的固定参数 ========
mc_alias = "myminio"
bucket_name = "rag-data"
minio_endpoint = "http://120.24.119.246:19000"
# ====================================

if not json_path.exists():
    raise FileNotFoundError(f"JSON 不存在: {json_path}")

if not images_dir.exists():
    raise FileNotFoundError(f"images_dir 不存在: {images_dir}")

# 输出文件路径
output_json_path = json_path.with_name(json_path.stem + "_minio_public.json")

# 计算 object 前缀
# 如果路径里包含 mineru_out，就尽量保留层级结构，例如：
# mineru_out/4/auto/images/xxx.jpg
parts = json_path.parts
try:
    idx = parts.index("mineru_out")
    object_prefix = "/".join(parts[idx:-1]) + "/images"
except ValueError:
    object_prefix = images_dir.name

print(f"object_prefix = {object_prefix}")

# 读取 JSON
with open(json_path, "r", encoding="utf-8") as f:
    content_list = json.load(f)

# 收集需要上传的图片文件
needed_files = set()
for item in content_list:
    if item.get("type") in {"image", "table"}:
        img_path = item.get("img_path")
        if img_path:
            needed_files.add(Path(img_path).name)

print(f"JSON 中需要处理的图片数量: {len(needed_files)}")

uploaded = 0
missing = []
failed = []

# 上传图片到 MinIO
for filename in sorted(needed_files):
    local_file = images_dir / filename
    if not local_file.exists():
        missing.append(filename)
        continue

    object_key = f"{object_prefix}/{filename}"
    target = f"{mc_alias}/{bucket_name}/{object_key}"

    cmd = ["mc", "cp", str(local_file), target]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        failed.append((filename, result.stderr.strip()))
        continue

    uploaded += 1

print(f"成功上传: {uploaded}")

if missing:
    print("\n以下文件在 images_dir 中没有找到:")
    for x in missing:
        print(" -", x)

if failed:
    print("\n以下文件上传失败:")
    for name, err in failed:
        print(f" - {name}")
        print(f"   错误: {err}")

# 替换 JSON 中的 img_path
replaced = 0
for item in content_list:
    if item.get("type") in {"image", "table"}:
        old_img_path = item.get("img_path")
        if not old_img_path:
            continue

        filename = Path(old_img_path).name
        object_key = f"{object_prefix}/{filename}"
        public_url = f"{minio_endpoint}/{bucket_name}/{quote(object_key)}"

        item["img_path_local"] = old_img_path
        item["img_object_key"] = object_key
        item["img_path"] = public_url
        replaced += 1

print(f"\n成功替换 img_path 数量: {replaced}")

# 写出新 JSON
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(content_list, f, ensure_ascii=False, indent=2)

print(f"\n新 JSON 已保存到: {output_json_path}")