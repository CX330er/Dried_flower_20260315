from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.loso_npz import EEGDataset, build_loso_folds, load_subject_data, normalize_by_train_stats
from models.deepconvnet import DeepConvNet
from models.eegnet import EEGNet
from models.fbcnet import FBCNet
from models.msfbcnn import MSFBCNN
from models.shallowconvnet import ShallowConvNet
from utils.metrics import (
    compute_metrics,
    save_confusion_matrix,
    save_model_comparison_plots,
    save_training_curve,
)
from utils.seed import set_seed

MODEL_REGISTRY = {
    "ShallowConvNet": ShallowConvNet,
    "DeepConvNet": DeepConvNet,
    "EEGNet": EEGNet,
    "FBCNet": FBCNet,
    "MSFBCNN": MSFBCNN,
}

ALL_MODELS = ["ShallowConvNet", "DeepConvNet", "EEGNet", "FBCNet", "MSFBCNN"]

class EarlyStopping:
    def __init__(self, patience: int = 50):
        self.patience = patience
        self.best_loss = float("inf")
        self.wait = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
            self.best_state = deepcopy(model.state_dict())
            return False
        self.wait += 1
        return self.wait >= self.patience


def _run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * yb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

    return total_loss / total, correct / total


def train_and_evaluate_model(
    model_name: str,
    data_dir: str = "data/processed/bcic_iv_2a",
    results_root: str = "results",
    n_channels: int = 22,
    n_classes: int = 4,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 50,
    seed: int = 42,
):
    set_seed(seed)

    subject_data = load_subject_data(data_dir)
    folds = build_loso_folds(subject_data, val_ratio=0.2, seed=seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_metrics = []

    print(f"[{model_name}] start training on {len(folds)} LOSO folds")

    for fold_idx, (test_sid, fold_raw) in enumerate(folds, start=1):
        fold = normalize_by_train_stats(fold_raw)
        input_time = fold.x_train.shape[-1]

        print(
            f"[{model_name}] fold {fold_idx}/{len(folds)} | "
            f"test_subject={test_sid} | "
            f"train={len(fold.y_train)} val={len(fold.y_val)} test={len(fold.y_test)}"
        )

        model = MODEL_REGISTRY[model_name](
            n_channels=n_channels,
            n_classes=n_classes,
            input_time=input_time,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)
        early = EarlyStopping(patience=patience)

        train_loader = DataLoader(EEGDataset(fold.x_train, fold.y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(EEGDataset(fold.x_val, fold.y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(EEGDataset(fold.x_test, fold.y_test), batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        epoch_bar = tqdm(
            range(epochs),
            desc=f"{model_name} fold {fold_idx}",
            unit="epoch",
            leave=False,
        )
        for epoch_idx in epoch_bar:
            tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
            va_loss, va_acc = _run_epoch(model, val_loader, criterion, optimizer=None, device=device)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)

            epoch_bar.set_postfix(
                epoch=epoch_idx + 1,
                train_loss=f"{tr_loss:.4f}",
                val_loss=f"{va_loss:.4f}",
                train_acc=f"{tr_acc:.4f}",
                val_acc=f"{va_acc:.4f}",
                best_val=f"{min(history['val_loss']):.4f}",
            )

            if early.step(va_loss, model):
                print(
                    f"[{model_name}] fold {fold_idx} early stopped at epoch {epoch_idx + 1} "
                    f"(best_val_loss={early.best_loss:.4f})"
                )
                break

        if early.best_state is not None:
            model.load_state_dict(early.best_state)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1).cpu().numpy()
                y_pred.extend(pred.tolist())
                y_true.extend(yb.numpy().tolist())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        metrics = compute_metrics(y_true, y_pred)
        metrics.update({"fold": fold_idx, "test_subject": test_sid})
        all_metrics.append(metrics)

        print(
            f"[{model_name}] fold {fold_idx} done | "
            f"accuracy={metrics['accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f} "
            f"kappa={metrics['kappa']:.4f}"
        )

        fold_dir = Path(results_root) / model_name.lower() / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        save_confusion_matrix(y_true, y_pred, fold_dir / "confusion_matrix.png")
        save_training_curve(history, fold_dir / "training_curve.png")

    summary_path = Path(results_root) / model_name.lower() / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(all_metrics)
    avg_row = {
        "fold": "mean",
        "test_subject": "all",
        "accuracy": df["accuracy"].mean(),
        "f1_macro": df["f1_macro"].mean(),
        "kappa": df["kappa"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    df.to_csv(summary_path, index=False)

    print(
        f"[{model_name}] summary | "
        f"accuracy={avg_row['accuracy']:.4f} "
        f"f1_macro={avg_row['f1_macro']:.4f} "
        f"kappa={avg_row['kappa']:.4f}"
    )

    return df


def run_all_baselines(data_dir: str = "data/processed/bcic_iv_2a", results_root: str = "results"):
    import pandas as pd

    compare_rows = []
    for model_name in ALL_MODELS:
        df = train_and_evaluate_model(model_name=model_name, data_dir=data_dir, results_root=results_root)
        mean_row = df[df["fold"] == "mean"].iloc[0]
        compare_rows.append(
            {
                "model": model_name,
                "accuracy": mean_row["accuracy"],
                "f1_macro": mean_row["f1_macro"],
                "kappa": mean_row["kappa"],
            }
        )

    comp_df = pd.DataFrame(compare_rows)
    comp_path = Path(results_root) / "baseline_compare.csv"
    comp_df.to_csv(comp_path, index=False)

    save_model_comparison_plots(results_root=results_root, model_names=ALL_MODELS)

    print(f"[baseline] comparison saved to {comp_path}")
    return comp_df
