"""题三：读取 Parquet 数据并训练 MLP 完成 MNIST 分类。

运行 python 02_mnist_mlp.py，结果保存至 results/mnist。
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from io import BytesIO
import hashlib
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train-00000-of-00001.parquet"
OUTPUT = ROOT / "results" / "mnist"
SEED = 2026
BATCH_SIZE = 128
EPOCHS = 10  # 预先固定训练轮数，不按测试准确率选模型。
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}。请保留包内的 data 文件夹。")
    table = pd.read_parquet(path)
    if not {"image", "label"}.issubset(table.columns):
        raise ValueError("Parquet 必须包含 image 和 label 两列。")
    pixels = np.empty((len(table), 28, 28), dtype=np.uint8)
    for index, cell in enumerate(table["image"]):
        if not isinstance(cell, dict) or cell.get("bytes") is None:
            raise ValueError(f"第 {index} 行的 image 不是包含 bytes 的图片字典。")
        with Image.open(BytesIO(cell["bytes"])) as picture:  # Parquet 的 image 列保存图片字节。
            gray = picture.convert("L")
            if gray.size != (28, 28):
                raise ValueError(f"第 {index} 张图像尺寸为 {gray.size}，需要 28×28。")
            pixels[index] = np.asarray(gray, dtype=np.uint8)
    labels_np = table["label"].to_numpy(dtype=np.int64)
    if len(labels_np) < 20 or not np.isin(labels_np, np.arange(10)).all():
        raise ValueError("数据过少，或标签包含 0 到 9 之外的值。")
    images = torch.from_numpy(pixels).unsqueeze(1).float() / 255.0  # [N,1,28,28]，像素缩放至 [0,1]。
    labels = torch.from_numpy(labels_np.copy()).long()
    return images, labels


def split_data(images, labels):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(labels))  # 按固定随机种子划分互不重叠的 80%/10%/10% 子集。
    train_end = int(len(order) * 0.8)
    val_end = int(len(order) * 0.9)
    groups = {
        "train": order[:train_end],
        "validation": order[train_end:val_end],
        "test": order[val_end:],
    }
    datasets = {}
    for name, indices in groups.items():
        index_tensor = torch.from_numpy(indices)
        datasets[name] = TensorDataset(images[index_tensor], labels[index_tensor])
    return datasets, groups


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),  # 保留 batch 维，将每张图片展开为 784 维。
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 输出原始 logits，交叉熵前不做 Softmax。
        )

    def forward(self, x):
        return self.layers(x)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_count = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_count = labels.size(0)
        total_loss += loss.item() * batch_count  # 按样本数加权，兼容不足一个 batch 的末批。
        total_count += batch_count
    return total_loss / total_count


def evaluate(model, loader, criterion):
    model.eval()  # 关闭 Dropout；梯度记录由下面的 no_grad 关闭。
    total_loss = 0.0
    correct = 0
    total_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            predicted = logits.argmax(dim=1)
            count = labels.size(0)
            total_loss += loss.item() * count
            correct += (predicted == labels).sum().item()
            total_count += count
    return total_loss / total_count, correct / total_count


def save_figures(history, model, test_dataset):
    table = pd.DataFrame(history)
    table.to_csv(OUTPUT / "history.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(table["epoch"], table["train_loss"], "o-", label="Train")
    axes[0].plot(table["epoch"], table["val_loss"], "s-", label="Validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(table["epoch"], table["train_acc"] * 100, "o-", color="tab:green")
    axes[1].set(title="Train accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[2].plot(table["epoch"], table["test_acc"] * 100, "o-", color="tab:orange")
    axes[2].set(title="Held-out test accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xticks(table["epoch"])
    fig.tight_layout()
    fig.savefig(OUTPUT / "training_curves.png", dpi=180)
    plt.close(fig)

    for column, title, filename, scale, ylabel in [
        ("train_loss", "Training loss", "loss_curve.png", 1, "Cross-entropy loss"),
        ("train_acc", "Train accuracy", "train_accuracy.png", 100, "Accuracy (%)"),
        ("test_acc", "Held-out test accuracy", "test_accuracy.png", 100, "Accuracy (%)"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot(table["epoch"], table[column] * scale, "o-")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel, xticks=table["epoch"])
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT / filename, dpi=180)
        plt.close(fig)

    images, labels = test_dataset.tensors  # 固定展示测试集前九张图片。
    with torch.no_grad():
        logits = model(images[:9].to(DEVICE)).cpu()
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.5))
    for index, ax in enumerate(axes.flat):
        truth = int(labels[index])
        guess = int(predicted[index])
        ax.imshow(images[index, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"True: {truth} | Pred: {guess}\nScore: {confidence[index]:.1%}",
                     color="darkgreen" if truth == guess else "crimson")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT / "predictions_9.png", dpi=180)
    plt.close(fig)


def main():
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
    OUTPUT.mkdir(parents=True, exist_ok=True)
    images, labels = load_data(DATA_PATH)
    datasets, groups = split_data(images, labels)
    np.savez_compressed(OUTPUT / "split_indices.npz", **groups)
    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    eval_loaders = {name: DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
                    for name, dataset in datasets.items()}
    model = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    print(f"device={DEVICE}, split=" + str({k: len(v) for k, v in datasets.items()}), flush=True)
    print(model)

    for epoch in range(1, EPOCHS + 1):
        optimization_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        train_loss, train_acc = evaluate(model, eval_loaders["train"], criterion)  # 在轮末固定模型状态下重新评估训练集。
        val_loss, val_acc = evaluate(model, eval_loaders["validation"], criterion)
        test_loss, test_acc = evaluate(model, eval_loaders["test"], criterion)  # 按题目要求记录测试曲线，不用于调参。
        history.append({
            "epoch": epoch, "optimization_loss": optimization_loss,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "test_loss": test_loss, "test_acc": test_acc,
        })
        print(f"epoch={epoch:02d}/{EPOCHS}, train_loss={train_loss:.4f}, "
              f"train_acc={train_acc:.2%}, val_acc={val_acc:.2%}, test_acc={test_acc:.2%}", flush=True)

    torch.save(model.state_dict(), OUTPUT / "mnist_mlp.pt")
    save_figures(history, model, datasets["test"])
    summary = {
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "optimizer": "Adam", "dropout": 0.2,
        "architecture": "784-128-64-10", "device": str(DEVICE),
        "torch_version": torch.__version__, "pandas_version": pd.__version__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "source_name": DATA_PATH.name,
        "source_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "test_protocol": "Random 10% holdout from provided 60000-row training parquet; not official MNIST test split",
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "last_epoch": history[-1], "elapsed_seconds": time.perf_counter() - start,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"题三结果已保存到：{OUTPUT}")


if __name__ == "__main__":
    main()
