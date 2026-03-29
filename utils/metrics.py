from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_curve(history: dict, path: str | Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(epochs, history["train_acc"], label="train")
    ax[1].plot(epochs, history["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].legend()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def save_model_comparison_plots(results_root: str | Path, model_names: list[str]) -> None:
    """Save cross-model visualization figures under results root."""
    results_root = Path(results_root)

    compare_csv = results_root / "baseline_compare.csv"
    if not compare_csv.exists():
        return

    comp_df = pd.read_csv(compare_csv)
    if comp_df.empty:
        return

    metrics = ["accuracy", "f1_macro", "kappa"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, metric in enumerate(metrics):
        axes[i].bar(comp_df["model"], comp_df[metric], color="steelblue")
        axes[i].set_title(metric)
        axes[i].set_ylim(0.0, 1.0)
        axes[i].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(results_root / "baseline_compare_metrics.png", dpi=150)
    plt.close(fig)

    rows = []
    for model_name in model_names:
        summary_path = results_root / model_name.lower() / "summary.csv"
        if not summary_path.exists():
            continue

        df = pd.read_csv(summary_path)
        if "fold" not in df.columns:
            continue

        fold_df = df[df["fold"].astype(str) != "mean"].copy()
        if fold_df.empty:
            continue

        fold_df["model"] = model_name
        rows.append(fold_df)

    if not rows:
        return

    fold_df = pd.concat(rows, ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, metric in enumerate(metrics):
        grouped = [
            fold_df[fold_df["model"] == model][metric].dropna().values
            for model in model_names
            if model in fold_df["model"].unique()
        ]
        labels = [model for model in model_names if model in fold_df["model"].unique()]
        if grouped:
            axes[i].boxplot(grouped, tick_labels=labels)
        axes[i].set_title(f"Fold-wise {metric}")
        axes[i].set_ylim(0.0, 1.0)
        axes[i].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(results_root / "foldwise_metrics_boxplot.png", dpi=150)
    plt.close(fig)