"""题二：基于 PyTorch 自动求导的非线性最小二乘拟合。

运行 python 01_fit_function.py，结果保存至 results/fit。
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

SEED = 2026
TRAIN_SIZE = 4000
TEST_SIZE = 2000
STEPS = 6000
LEARNING_RATE = 0.01
OUTPUT = Path(__file__).resolve().parent / "results" / "fit"
NAMES = ["a1", "a2", "a3", "a4", "a5", "a6", "b"]
TRUE_VALUES = [2.0, 1.5, 3.0, 0.5, -0.5, 0.8, 2026.0]  # 仅用于生成模拟观测和比较参数，不用于初始化模型。


def true_function(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (
        2 * x1**2 + 1.5 * x1 * x2 + 3 * np.sin(x1)
        + 0.5 * np.cos(2 * x2) + np.exp(-0.5 * x1)
        + 0.8 * x2 + 2026
    )


def make_samples(rng, count):
    x = rng.uniform(-3, 3, size=(count, 2))
    clean = true_function(x)
    noise = rng.uniform(-3, 3, size=count)
    noisy = clean + noise
    return x, clean, noisy


class FunctionModel(nn.Module):
    def __init__(self, y_mean):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(7, dtype=torch.float64))
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float64))  # 保存训练集均值，恢复原始输出尺度时使用。

    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        a1, a2, a3, a4, a5, a6, b_offset = self.params
        return (
            a1 * x1**2 + a2 * x1 * x2 + a3 * torch.sin(x1)
            + a4 * torch.cos(2 * x2) + torch.exp(a5 * x1)
            + a6 * x2 + b_offset
        )

    def predict_y(self, x):
        return self(x) + self.y_mean


def r2_score(y_true, y_pred):
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if total == 0:
        raise ValueError("真实值没有变化，不能按通常定义计算 R²。")
    return float(1 - residual / total)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_x, _, train_y = make_samples(rng, TRAIN_SIZE)
    test_x, test_clean, test_y = make_samples(rng, TEST_SIZE)
    x_tensor = torch.tensor(train_x, dtype=torch.float64)
    y_mean = float(train_y.mean())  # 只根据训练集中心化标签，减少大截距对优化的影响。
    target = torch.tensor(train_y - y_mean, dtype=torch.float64)
    model = FunctionModel(y_mean)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)  # 每步使用全部训练样本，执行全批量梯度下降。
    history = []

    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        prediction = model(x_tensor)
        loss = torch.mean((prediction - target) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError("损失出现 NaN 或无穷大，请降低学习率并检查输入范围。")
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0:
            history.append({"step": step, "mse_before_update": loss.item()})
        if step % 1000 == 0:
            print(f"step={step}/{STEPS}, MSE={loss.item():.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        test_prediction = model.predict_y(torch.tensor(test_x, dtype=torch.float64)).numpy()
        train_prediction = model.predict_y(x_tensor).numpy()
    learned = model.params.detach().numpy().copy()
    learned[-1] += y_mean  # 还原原模型截距：b = b_offset + y_mean。
    parameter_table = pd.DataFrame({
        "parameter": NAMES, "true_value": TRUE_VALUES, "fitted_value": learned,
        "absolute_error": np.abs(learned - np.array(TRUE_VALUES)),
    })
    parameter_table.to_csv(OUTPUT / "parameters.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(OUTPUT / "loss_history.csv", index=False)
    metrics = {
        "seed": SEED, "domain": [-3, 3], "noise": "Uniform(-3, 3)",
        "train_size": TRAIN_SIZE, "test_size": TEST_SIZE, "steps": STEPS,
        "optimizer": "SGD (full-batch gradient descent)", "learning_rate": LEARNING_RATE,
        "torch_version": torch.__version__, "dtype": "float64",
        "train_mse_noisy": float(np.mean((train_prediction - train_y) ** 2)),
        "test_mse_noisy": float(np.mean((test_prediction - test_y) ** 2)),
        "test_r2_noisy": r2_score(test_y, test_prediction),  # 带噪声独立测试集上的主要评价指标。
        "test_r2_clean": r2_score(test_clean, test_prediction),  # 与无噪声曲面比较的补充指标。
        "train_label_mean": y_mean, "parameters": dict(zip(NAMES, learned.tolist())),
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), OUTPUT / "fitted_parameters.pt")

    axis_values = np.linspace(-3, 3, 65)
    xx1, xx2 = np.meshgrid(axis_values, axis_values)
    grid = np.column_stack((xx1.ravel(), xx2.ravel()))
    true_surface = true_function(grid).reshape(xx1.shape)
    with torch.no_grad():
        fitted_surface = model.predict_y(torch.tensor(grid, dtype=torch.float64)).numpy()
    fitted_surface = fitted_surface.reshape(xx1.shape)
    fig = plt.figure(figsize=(13, 5.4))
    z_low = float(min(true_surface.min(), fitted_surface.min()))
    z_high = float(max(true_surface.max(), fitted_surface.max()))
    for index, (surface, title) in enumerate([
        (true_surface, "True function without noise"),
        (fitted_surface, "PyTorch fitted function"),
    ], start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        ax.plot_surface(xx1, xx2, surface, cmap="viridis", linewidth=0, alpha=0.9)
        ax.set(xlabel="x1", ylabel="x2", zlabel="y", title=title, zlim=(z_low, z_high))
        ax.ticklabel_format(axis="z", style="plain", useOffset=False)  # 保留公式中的 2026，不使用纵轴偏移显示。
        ax.view_init(elev=25, azim=-55)
    fig.suptitle(f"Independent noisy test R2 = {metrics['test_r2_noisy']:.6f}")
    fig.tight_layout()
    fig.savefig(OUTPUT / "surfaces.png", dpi=180, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogy([row["step"] for row in history], [row["mse_before_update"] for row in history])
    ax.set(xlabel="Update step", ylabel="Training MSE (log scale)", title="Least-squares training")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "loss_curve.png", dpi=180)
    plt.close(fig)
    print(parameter_table.to_string(index=False))
    print(json.dumps(metrics, indent=2))
    print(f"题二结果已保存到：{OUTPUT}")


if __name__ == "__main__":
    main()
