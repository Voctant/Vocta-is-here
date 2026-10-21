"""训练基础 CNN，保存模型、实验记录和分类结果图片。"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import CLASSES, IMAGE_SIZE, SceneDataset
from model import SimpleCNN
from visualize import plot_confusion_matrix, plot_history, plot_predictions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_one_epoch(model, loader, loss_function, optimizer):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_function):
    model.eval()
    total_loss, correct = 0.0, 0
    all_labels, all_predictions, all_scores = [], [], []
    for images, labels in loader:
        logits = model(images)
        total_loss += loss_function(logits, labels).item() * len(labels)
        scores, predictions = logits.softmax(dim=1).max(dim=1)
        correct += (predictions == labels).sum().item()
        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())
        all_scores.extend(scores.tolist())
    size = len(loader.dataset)
    return (total_loss / size, correct / size, np.array(all_labels),
            np.array(all_predictions), np.array(all_scores))


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("epochs、batch-size、threads 和 lr 必须为正数。")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    print("Loading training and test images...", flush=True)
    training = SceneDataset(args.data_dir, "seg_train")
    testing = SceneDataset(args.data_dir, "seg_test")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    train_eval_loader = DataLoader(training, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(testing, batch_size=args.batch_size, shuffle=False)
    model = SimpleCNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Train: {len(training)}, Test: {len(testing)}, Device: CPU", flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, loss_function, optimizer)
        train_loss, train_acc, _, _, _ = evaluate(model, train_eval_loader, loss_function)
        test_loss, test_acc, actual, predicted, scores = evaluate(model, test_loader, loss_function)
        history.append(dict(epoch=epoch, train_loss=loss, train_eval_loss=train_loss,
                            train_acc=train_acc, test_loss=test_loss, test_acc=test_acc))
        print(f"Epoch {epoch:02d}/{args.epochs} | Loss {loss:.4f} | "
              f"Train Acc {train_acc:.2%} | Test Acc {test_acc:.2%}", flush=True)
    # 固定采用最后一轮模型；测试集只用于计算指标，不参与反向传播或挑选模型。
    config = dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
                  seed=args.seed, threads=args.threads, image_size=IMAGE_SIZE, device="cpu")
    torch.save({"model_state_dict": model.state_dict(), "classes": CLASSES,
                "config": config}, args.output_dir / "cnn_model.pt")
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "true_label", "pred_label", "score"])
        for path, label, prediction, score in zip(testing.paths, actual, predicted, scores):
            writer.writerow([f"{path.parent.name}/{path.name}",
                             CLASSES[label], CLASSES[prediction], score])
    plot_history(history, args.output_dir)
    plot_predictions(testing, actual, predicted, scores, args.output_dir)
    matrix = plot_confusion_matrix(actual, predicted, args.output_dir)
    metrics = dict(
        dataset="Intel Image Classification", classes=CLASSES, config=config,
        train_count=len(training), test_count=len(testing),
        train_class_counts=torch.bincount(training.labels, minlength=6).tolist(),
        test_class_counts=torch.bincount(testing.labels, minlength=6).tolist(),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        final_epoch=history[-1], test_correct=int((actual == predicted).sum()),
        confusion_matrix=matrix.tolist(),
        per_class_recall=(matrix.diagonal() / matrix.sum(axis=1)).tolist(),
        elapsed_seconds=time.perf_counter() - start,
        python_version=platform.python_version(), torch_version=torch.__version__,
        numpy_version=np.__version__,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Results saved to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
