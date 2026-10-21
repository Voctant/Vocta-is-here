"""将真实训练记录、预测结果保存成 PNG。"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import CLASSES


def plot_history(history, output_dir):
    output_dir = Path(output_dir)
    epochs = [row["epoch"] for row in history]
    specifications = [
        ("loss.png", "Training loss", "Cross-entropy loss", "train_loss", 1),
        ("train_acc.png", "Training accuracy", "Accuracy (%)", "train_acc", 100),
        ("test_acc.png", "Test accuracy", "Accuracy (%)", "test_acc", 100),
    ]
    for filename, title, ylabel, key, scale in specifications:
        fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
        values = [row[key] * scale for row in history]
        ax.plot(epochs, values, marker="o", color="#2563a6", linewidth=2)
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel, xticks=epochs)
        ax.grid(alpha=0.25)
        if scale == 100:
            ax.set_ylim(0, 100)
        ax.annotate(f"{values[-1]:.2f}" if scale == 100 else f"{values[-1]:.4f}",
                    (epochs[-1], values[-1]), xytext=(-42, 12),
                    textcoords="offset points", fontsize=10)
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def plot_predictions(dataset, actual, predicted, scores, output_dir):
    output_dir = Path(output_dir)
    # 每类固定取排序最前的两张测试图，不根据预测是否正确挑图。
    indices = [i for label in range(len(CLASSES))
               for i in np.flatnonzero(actual == label)[:2]]
    selected = [(index, "class_samples") for index in indices]
    draw_grid(dataset, indices, actual, predicted, scores,
              output_dir / "classification_results.png", "Test images: two per class")
    wrong = np.flatnonzero(actual != predicted)[:12].tolist()
    if wrong:
        selected.extend((index, "first_errors") for index in wrong)
        draw_grid(dataset, wrong, actual, predicted, scores,
                  output_dir / "error_examples.png", "First misclassified test images")
    with (output_dir / "visualization_samples.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["selection", "test_index", "file", "true", "pred", "score"])
        for index, selection in selected:
            writer.writerow([selection, index, dataset.paths[index].name,
                             CLASSES[actual[index]], CLASSES[predicted[index]], scores[index]])


def draw_grid(dataset, indices, actual, predicted, scores, destination, title):
    rows = (len(indices) + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(12, 4.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, indices):
        ax.imshow(dataset.images[index])
        color = "#176c43" if actual[index] == predicted[index] else "#b43333"
        ax.set_title(f"True: {CLASSES[actual[index]]}\n"
                     f"Pred: {CLASSES[predicted[index]]}\n"
                     f"Score: {scores[index]:.1%}", color=color, fontsize=10)
    fig.suptitle(title, fontsize=16, y=0.97)
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.025, top=0.89,
                        wspace=0.10, hspace=0.50)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(actual, predicted, output_dir):
    matrix = np.zeros((6, 6), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    picture = ax.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(picture, ax=ax, label="Number of images")
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES,
           yticklabels=CLASSES, xlabel="Predicted class", ylabel="True class",
           title="Confusion matrix on the test set")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for row in range(6):
        for col in range(6):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    fig.savefig(Path(output_dir) / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    return matrix
