"""读取文件夹中的图片，并统一为 64×64 的 RGB 图像。"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_NAMES_ZH = ["建筑", "森林", "冰川", "山地", "海洋", "街道"]
IMAGE_SIZE = 64


class SceneDataset(Dataset):
    def __init__(self, data_dir, split):
        root = Path(data_dir) / split
        if (root / split).is_dir():
            root = root / split
        self.paths, self.labels, images = [], [], []
        for label, class_name in enumerate(CLASSES):
            folder = root / class_name
            files = sorted(folder.glob("*.jpg"))
            if not files:
                raise FileNotFoundError(f"没有找到图片：{folder}。请先运行 download_data.py。")
            for file in files:
                with Image.open(file) as image:
                    image = image.convert("RGB").resize(
                        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
                    )
                    images.append(np.array(image, dtype=np.uint8))
                self.paths.append(file)
                self.labels.append(label)
        # 只在启动时读取图片，训练时从内存取图，避免反复访问磁盘。
        self.images = np.stack(images)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # H×W×C 转成 PyTorch 需要的 C×H×W，并把像素缩放到 0～1。
        image = torch.from_numpy(self.images[index]).permute(2, 0, 1).float() / 255.0
        return image, self.labels[index]
