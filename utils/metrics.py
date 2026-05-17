from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def _metric_labels(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> list[int]:
    if n_classes is not None:
        return list(range(n_classes))
    observed = np.unique(np.concatenate([y_true, y_pred]))
    return [int(label) for label in observed.tolist()]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> dict:
    labels = _metric_labels(y_true, y_pred, n_classes=n_classes)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    true_counts = np.bincount(y_true.astype(int), minlength=len(labels))
    pred_counts = np.bincount(y_pred.astype(int), minlength=len(labels))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    for idx, label in enumerate(labels):
        metrics[f"class_{label}_precision"] = float(precision[idx])
        metrics[f"class_{label}_recall"] = float(recall[idx])
        metrics[f"class_{label}_f1"] = float(per_class_f1[idx])
        metrics[f"class_{label}_support"] = int(support[idx])
        metrics[f"true_count_class_{label}"] = int(true_counts[idx])
        metrics[f"pred_count_class_{label}"] = int(pred_counts[idx])

    for true_idx, true_label in enumerate(labels):
        for pred_idx, pred_label in enumerate(labels):
            metrics[f"cm_true_{true_label}_pred_{pred_label}"] = int(cm[true_idx, pred_idx])

    return metrics


def save_confusion_matrix_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path,
    n_classes: int | None = None,
) -> None:
    labels = _metric_labels(y_true, y_pred, n_classes=n_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def save_predictions(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path) -> None:
    df = pd.DataFrame(
        {
            "trial_index": np.arange(len(y_true), dtype=int),
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "correct": (y_true == y_pred).astype(int),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_history_csv(history: dict, path: str | Path) -> None:
    df = pd.DataFrame(history)
    df.insert(0, "epoch", np.arange(1, len(df) + 1, dtype=int))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


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
