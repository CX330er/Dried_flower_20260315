from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from datasets.loso_npz import EEGDataset, build_loso_folds, load_subject_data, normalize_by_train_stats
from models.deepconvnet import DeepConvNet
from models.eegnet import EEGNet
from models.fbcnet import FBCNet
from models.msfbcnn import MSFBCNN
from models.shallowconvnet import ShallowConvNet
from models.eegnet_fsfe import EEGNetFSFE
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
    "EEGNetFSFE": EEGNetFSFE,
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

class EEGSubjectDataset(EEGDataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, sid: np.ndarray):
        super().__init__(x, y)
        self.sid = torch.from_numpy(sid).long()

    def __getitem__(self, idx: int):
        xb, yb = super().__getitem__(idx)
        return xb, yb, self.sid[idx]


def _supervised_center_loss(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Lightweight supervised center-style loss for cross-subject class aggregation."""
    if features.ndim != 2:
        raise ValueError(f"Expected features [B, D], got {tuple(features.shape)}")

    unique_labels = labels.unique()
    losses = []
    for cls in unique_labels:
        mask = labels == cls
        cls_feats = features[mask]
        if cls_feats.size(0) < 2:
            continue
        center = cls_feats.mean(dim=0, keepdim=True)
        losses.append(((cls_feats - center) ** 2).mean())

    if not losses:
        return features.new_tensor(0.0)
    return torch.stack(losses).mean()


def _subject_coral_loss(features: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
    """Invariance regularization among source subjects via pairwise CORAL covariance alignment."""
    uniq_subjects = subject_ids.unique()
    if uniq_subjects.numel() < 2:
        return features.new_tensor(0.0)

    covs = []
    for sid in uniq_subjects:
        mask = subject_ids == sid
        sub_feats = features[mask]
        if sub_feats.size(0) < 2:
            continue
        centered = sub_feats - sub_feats.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(sub_feats.size(0) - 1, 1)
        covs.append(cov)

    if len(covs) < 2:
        return features.new_tensor(0.0)

    pair_losses = []
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            pair_losses.append(((covs[i] - covs[j]) ** 2).mean())
    return torch.stack(pair_losses).mean()


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cpu",
    aux_mode: str = "none",
    lambda_aux: float = 0.0,
    max_time_shift: int = 0,
    noise_std: float = 0.0,
    grad_clip_norm: float = 0.0,
):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    # Auxiliary DG regularization is optimization-only and should not affect
    # validation/test objective used for model selection.
    effective_lambda_aux = lambda_aux if train else 0.0

    with torch.set_grad_enabled(train):
        for batch in loader:
            if len(batch) == 3:
                xb, yb, sid = batch
                sid = sid.to(device)
            else:
                xb, yb = batch
                sid = None

            xb, yb = xb.to(device), yb.to(device)

            if train and max_time_shift > 0:
                shift = int(np.random.randint(-max_time_shift, max_time_shift + 1))
                if shift != 0:
                    xb = torch.roll(xb, shifts=shift, dims=-1)
            if train and noise_std > 0.0:
                xb = xb + noise_std * torch.randn_like(xb)

            if aux_mode in {"center", "coral"}:
                logits, features = model(xb, return_features=True)
            else:
                logits = model(xb)
                features = None

            cls_loss = criterion(logits, yb)
            aux_loss = cls_loss.new_tensor(0.0)
            if aux_mode == "center" and features is not None:
                aux_loss = _supervised_center_loss(features, yb)
            elif aux_mode == "coral" and features is not None and sid is not None:
                aux_loss = _subject_coral_loss(features, sid)

            loss = cls_loss + lambda_aux * aux_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
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
    aux_mode: str = "none",
    lambda_aux: float = 0.02,
    label_smoothing: float = 0.1,
    use_class_weights: bool = True,
    use_weighted_sampler: bool = True,
    max_time_shift: int = 25,
    noise_std: float = 0.01,
    grad_clip_norm: float = 1.0,
    aux_warmup_epochs: int = 30,
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

        cls_weights = None
        if use_class_weights:
            bincount = np.bincount(fold.y_train, minlength=n_classes).astype(np.float32)
            inv = 1.0 / np.clip(bincount, a_min=1.0, a_max=None)
            cls_weights = (inv / inv.sum()) * float(n_classes)
            cls_weights = torch.tensor(cls_weights, dtype=torch.float32, device=device)

        criterion = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=label_smoothing)
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        early = EarlyStopping(patience=patience)

        if aux_mode in {"center", "coral"}:
            if fold.sid_train is None:
                raise ValueError("sid_train is required when aux_mode is center/coral.")
            train_dataset = EEGSubjectDataset(fold.x_train, fold.y_train, fold.sid_train)
            val_dataset = EEGSubjectDataset(fold.x_val, fold.y_val, fold.sid_val)
        else:
            train_dataset = EEGDataset(fold.x_train, fold.y_train)
            val_dataset = EEGDataset(fold.x_val, fold.y_val)

        sampler = None
        if use_weighted_sampler:
            class_counts = np.bincount(fold.y_train, minlength=n_classes).astype(np.float64)
            class_counts = np.clip(class_counts, a_min=1.0, a_max=None)
            sample_weights = 1.0 / class_counts[fold.y_train]
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(sample_weights),
                replacement=True,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(EEGDataset(fold.x_test, fold.y_test), batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        epoch_bar = tqdm(
            range(epochs),
            desc=f"{model_name} fold {fold_idx}",
            unit="epoch",
            leave=False,
        )
        for epoch_idx in epoch_bar:
            lambda_epoch = lambda_aux
            if aux_mode in {"center", "coral"} and aux_warmup_epochs > 0:
                lambda_epoch = lambda_aux * min(1.0, float(epoch_idx + 1) / float(aux_warmup_epochs))

            tr_loss, tr_acc = _run_epoch(
                model,
                train_loader,
                criterion,
                optimizer=optimizer,
                device=device,
                aux_mode=aux_mode,
                lambda_aux=lambda_epoch,
                max_time_shift=max_time_shift,
                noise_std=noise_std,
                grad_clip_norm=grad_clip_norm,
            )
            va_loss, va_acc = _run_epoch(
                model,
                val_loader,
                criterion,
                optimizer=None,
                device=device,
                aux_mode=aux_mode,
                lambda_aux=0.0,
            )

            scheduler.step()

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
