"""下载题目推荐的 Intel 场景分类数据集，只解压带标签的训练集和测试集。"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/puneet6060/intel-image-classification"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    root = args.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "intel-image-classification.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        print("Downloading Intel dataset (about 346 MiB)...", flush=True)
        request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            if "zip" not in response.headers.get("Content-Type", ""):
                raise RuntimeError("下载内容不是 ZIP，请从题目给出的 Kaggle 页面手动下载。")
            with temporary.open("wb") as output:
                total = 0
                milestone = 25 * 1024 * 1024
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    total += len(chunk)
                    if total >= milestone:
                        print(f"Downloaded {total / 1024 ** 2:.0f} MiB", flush=True)
                        milestone += 25 * 1024 * 1024
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            parts = Path(entry.filename).parts
            if not parts or parts[0] not in ("seg_train", "seg_test") or entry.is_dir():
                continue
            destination = (root / entry.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("压缩包包含不安全的路径。")
            if destination.exists() and destination.stat().st_size == entry.file_size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(entry) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"Dataset ready: {root}", flush=True)


if __name__ == "__main__":
    main()
