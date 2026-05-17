from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from datasets.loso_npz import (
    EEGDataset,
    build_loso_folds,
    build_loso_train_eval_folds,
    build_subject_dependent_te_folds,
    build_subject_dependent_te_final_folds,
    load_subject_data,
    normalize_by_train_stats,
)
from models.deepconvnet import DeepConvNet
from models.eegnet import EEGNet
from models.fbcnet import FBCNet
from models.msfbcnn import MSFBCNN
from models.shallowconvnet import ShallowConvNet
from models.eegnet_fsfe import EEGNetFSFE
from utils.metrics import (
    compute_metrics,
    save_confusion_matrix,
    save_confusion_matrix_values,
    save_history_csv,
    save_model_comparison_plots,
    save_predictions,
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

ALL_MODELS = ["ShallowConvNet", "DeepConvNet", "EEGNet", "FBCNet", "MSFBCNN", "EEGNetFSFE"]
PROTOCOL_CHOICES = ("loso_t", "subject_dependent_te", "subject_dependent_te_final", "loso_te")


def _build_protocol_folds(subject_data, protocol: str, val_ratio: float, seed: int):
    if protocol == "loso_t":
        return build_loso_folds(subject_data, val_ratio=val_ratio, seed=seed)
    if protocol == "subject_dependent_te":
        return build_subject_dependent_te_folds(subject_data, val_ratio=val_ratio, seed=seed)
    if protocol == "subject_dependent_te_final":
        return build_subject_dependent_te_final_folds(subject_data, val_ratio=val_ratio, seed=seed)
    if protocol == "loso_te":
        return build_loso_train_eval_folds(subject_data, val_ratio=val_ratio, seed=seed)
    raise ValueError(f"Unknown protocol={protocol}. Expected one of {PROTOCOL_CHOICES}.")


def _normalize_train_and_targets(
    x_train: np.ndarray,
    *target_arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Normalize arrays with statistics computed from x_train only."""
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    normalized = [(x_train - mean) / std]
    normalized.extend((arr - mean) / std for arr in target_arrays)
    return tuple(arr.astype(np.float32, copy=False) for arr in normalized)

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


def _class_count_fields(prefix: str, labels: np.ndarray, n_classes: int) -> dict:
    counts = np.bincount(labels.astype(int), minlength=n_classes)
    return {f"{prefix}_class_{idx}_count": int(counts[idx]) for idx in range(n_classes)}


def _value_counts(values: np.ndarray | None) -> dict:
    if values is None:
        return {}
    uniq, counts = np.unique(values.astype(int), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(uniq, counts)}


def _load_completed_fold_metrics(metrics_path: Path) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, (dict, list))
    }


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


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).T
    dists = x_norm + y_norm - 2.0 * (x @ y.T)
    return torch.clamp(dists, min=0.0)


def _multi_kernel_mmd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> torch.Tensor:
    if x.size(0) == 0 or y.size(0) == 0:
        return x.new_tensor(0.0)

    xy = torch.cat([x, y], dim=0)
    with torch.no_grad():
        all_dists = _pairwise_sq_dists(xy, xy)
        nonzero = all_dists[all_dists > 0.0]
        if nonzero.numel() == 0:
            bandwidth = x.new_tensor(1.0)
        else:
            bandwidth = torch.median(nonzero).clamp_min(1e-6)

    offsets = torch.arange(kernel_num, device=x.device, dtype=x.dtype) - kernel_num // 2
    bandwidths = bandwidth * (kernel_mul ** offsets)

    loss = x.new_tensor(0.0)
    xx = _pairwise_sq_dists(x, x)
    yy = _pairwise_sq_dists(y, y)
    xy_dists = _pairwise_sq_dists(x, y)
    for bw in bandwidths:
        k_xx = torch.exp(-xx / bw)
        k_yy = torch.exp(-yy / bw)
        k_xy = torch.exp(-xy_dists / bw)
        loss = loss + k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return loss / float(kernel_num)


def _subject_mmd_loss(
    features: torch.Tensor,
    subject_ids: torch.Tensor,
    min_samples: int = 2,
) -> torch.Tensor:
    domains = []
    for sid in subject_ids.unique():
        domain_feats = features[subject_ids == sid]
        if domain_feats.size(0) >= min_samples:
            domains.append(domain_feats)

    if len(domains) < 2:
        return features.new_tensor(0.0)

    pair_losses = []
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            pair_losses.append(_multi_kernel_mmd_loss(domains[i], domains[j]))
    return torch.stack(pair_losses).mean()


def _subject_conditional_mmd_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: torch.Tensor,
    min_samples: int = 2,
) -> torch.Tensor:
    class_losses = []
    for cls in labels.unique():
        cls_mask = labels == cls
        cls_feats = features[cls_mask]
        cls_subjects = subject_ids[cls_mask]
        cls_loss = _subject_mmd_loss(cls_feats, cls_subjects, min_samples=min_samples)
        if cls_loss.detach().abs().item() > 0.0:
            class_losses.append(cls_loss)

    if not class_losses:
        return features.new_tensor(0.0)
    return torch.stack(class_losses).mean()


def _eegdg_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: torch.Tensor,
    conditional_weight: float = 1.0,
) -> torch.Tensor:
    """EEG-DG-style source-only marginal and conditional distribution alignment."""
    features = F.normalize(features, p=2, dim=1)
    marginal = _subject_mmd_loss(features, subject_ids)
    conditional = _subject_conditional_mmd_loss(features, labels, subject_ids)
    return marginal + conditional_weight * conditional


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cpu",
    aux_mode: str = "none",
    lambda_aux: float = 0.0,
    eegdg_conditional_weight: float = 1.0,
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

            if aux_mode in {"center", "coral", "eegdg"}:
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
            elif aux_mode == "eegdg" and features is not None and sid is not None:
                aux_loss = _eegdg_loss(
                    features,
                    yb,
                    sid,
                    conditional_weight=eegdg_conditional_weight,
                )

            loss = cls_loss + effective_lambda_aux * aux_loss

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
    eegdg_conditional_weight: float = 1.0,
    label_smoothing: float = 0.1,
    use_class_weights: bool = True,
    use_weighted_sampler: bool = True,
    max_time_shift: int = 25,
    noise_std: float = 0.01,
    grad_clip_norm: float = 1.0,
    aux_warmup_epochs: int = 30,
    resume_existing: bool = False,
    protocol: str = "loso_t",
):
    set_seed(seed)

    subject_data = load_subject_data(data_dir)
    folds = _build_protocol_folds(subject_data, protocol=protocol, val_ratio=0.2, seed=seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_metrics = []

    print(f"[{model_name}] start training on {len(folds)} folds | protocol={protocol}")

    for fold_idx, (test_sid, fold_raw) in enumerate(folds, start=1):
        fold_dir = Path(results_root) / model_name.lower() / f"fold_{fold_idx}"
        metrics_path = fold_dir / "metrics.json"
        if resume_existing and metrics_path.exists():
            metrics = _load_completed_fold_metrics(metrics_path)
            all_metrics.append(metrics)
            print(
                f"[{model_name}] fold {fold_idx}/{len(folds)} | "
                f"test_subject={test_sid} | resume skip existing metrics"
            )
            continue

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

        if aux_mode in {"center", "coral", "eegdg"}:
            if fold.sid_train is None:
                raise ValueError("sid_train is required when aux_mode is center/coral/eegdg.")
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
            if aux_mode in {"center", "coral", "eegdg"} and aux_warmup_epochs > 0:
                lambda_epoch = lambda_aux * min(1.0, float(epoch_idx + 1) / float(aux_warmup_epochs))

            tr_loss, tr_acc = _run_epoch(
                model,
                train_loader,
                criterion,
                optimizer=optimizer,
                device=device,
                aux_mode=aux_mode,
                lambda_aux=lambda_epoch,
                eegdg_conditional_weight=eegdg_conditional_weight,
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
                eegdg_conditional_weight=eegdg_conditional_weight,
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

        history_for_plot = history
        selection_history = None
        selection_best_epoch = None
        selection_best_val_loss = None
        final_train_trials = None

        if protocol == "subject_dependent_te_final":
            if fold_raw.x_train_full is None or fold_raw.y_train_full is None:
                raise ValueError("subject_dependent_te_final requires full AxxT training data in FoldData.")

            selection_history = history
            selection_best_epoch = int(np.argmin(history["val_loss"]) + 1)
            selection_best_val_loss = float(np.min(history["val_loss"]))
            final_train_trials = int(len(fold_raw.y_train_full))

            print(
                f"[{model_name}] fold {fold_idx} final refit | "
                f"best_epoch={selection_best_epoch} "
                f"selection_best_val_loss={selection_best_val_loss:.4f} "
                f"full_train={final_train_trials}"
            )

            x_train_full, x_test_final = _normalize_train_and_targets(
                fold_raw.x_train_full,
                fold_raw.x_test,
            )
            y_train_full = fold_raw.y_train_full

            model = MODEL_REGISTRY[model_name](
                n_channels=n_channels,
                n_classes=n_classes,
                input_time=x_train_full.shape[-1],
            ).to(device)

            cls_weights_final = None
            if use_class_weights:
                bincount = np.bincount(y_train_full, minlength=n_classes).astype(np.float32)
                inv = 1.0 / np.clip(bincount, a_min=1.0, a_max=None)
                cls_weights_final = (inv / inv.sum()) * float(n_classes)
                cls_weights_final = torch.tensor(cls_weights_final, dtype=torch.float32, device=device)

            criterion_final = nn.CrossEntropyLoss(weight=cls_weights_final, label_smoothing=label_smoothing)
            optimizer_final = Adam(model.parameters(), lr=lr)
            scheduler_final = CosineAnnealingLR(optimizer_final, T_max=max(selection_best_epoch, 1), eta_min=1e-5)

            if aux_mode in {"center", "coral", "eegdg"}:
                if fold_raw.sid_train_full is None:
                    raise ValueError("sid_train_full is required when aux_mode is center/coral/eegdg.")
                train_dataset_final = EEGSubjectDataset(x_train_full, y_train_full, fold_raw.sid_train_full)
            else:
                train_dataset_final = EEGDataset(x_train_full, y_train_full)

            sampler_final = None
            if use_weighted_sampler:
                class_counts = np.bincount(y_train_full, minlength=n_classes).astype(np.float64)
                class_counts = np.clip(class_counts, a_min=1.0, a_max=None)
                sample_weights = 1.0 / class_counts[y_train_full]
                sampler_final = WeightedRandomSampler(
                    weights=torch.from_numpy(sample_weights).double(),
                    num_samples=len(sample_weights),
                    replacement=True,
                )

            train_loader_final = DataLoader(
                train_dataset_final,
                batch_size=batch_size,
                shuffle=sampler_final is None,
                sampler=sampler_final,
            )
            test_loader = DataLoader(
                EEGDataset(x_test_final, fold_raw.y_test),
                batch_size=batch_size,
                shuffle=False,
            )

            final_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
            final_epoch_bar = tqdm(
                range(selection_best_epoch),
                desc=f"{model_name} fold {fold_idx} final",
                unit="epoch",
                leave=False,
            )
            for final_epoch_idx in final_epoch_bar:
                lambda_epoch = lambda_aux
                if aux_mode in {"center", "coral", "eegdg"} and aux_warmup_epochs > 0:
                    lambda_epoch = lambda_aux * min(
                        1.0,
                        float(final_epoch_idx + 1) / float(aux_warmup_epochs),
                    )

                tr_loss, tr_acc = _run_epoch(
                    model,
                    train_loader_final,
                    criterion_final,
                    optimizer=optimizer_final,
                    device=device,
                    aux_mode=aux_mode,
                    lambda_aux=lambda_epoch,
                    eegdg_conditional_weight=eegdg_conditional_weight,
                    max_time_shift=max_time_shift,
                    noise_std=noise_std,
                    grad_clip_norm=grad_clip_norm,
                )
                scheduler_final.step()

                final_history["train_loss"].append(tr_loss)
                final_history["val_loss"].append(np.nan)
                final_history["train_acc"].append(tr_acc)
                final_history["val_acc"].append(np.nan)

                final_epoch_bar.set_postfix(
                    epoch=final_epoch_idx + 1,
                    train_loss=f"{tr_loss:.4f}",
                    train_acc=f"{tr_acc:.4f}",
                )

            history_for_plot = final_history
        elif early.best_state is not None:
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

        selection_epochs_ran = int(len(history["train_loss"]))
        selection_best_idx = int(np.argmin(history["val_loss"])) if history["val_loss"] else 0
        fold_best_epoch = int(selection_best_idx + 1)
        fold_best_val_loss = float(history["val_loss"][selection_best_idx]) if history["val_loss"] else float("nan")
        fold_best_val_acc = float(history["val_acc"][selection_best_idx]) if history["val_acc"] else float("nan")
        eval_epochs_ran = int(len(history_for_plot["train_loss"]))

        metrics = compute_metrics(y_true, y_pred, n_classes=n_classes)
        metrics.update(
            {
                "model": model_name,
                "seed": int(seed),
                "fold": fold_idx,
                "test_subject": test_sid,
                "protocol": protocol,
                "train_trials": int(len(fold.y_train)),
                "val_trials": int(len(fold.y_val)),
                "test_trials": int(len(fold.y_test)),
                "input_time": int(input_time),
                "n_channels": int(n_channels),
                "n_classes": int(n_classes),
                "epochs_config": int(epochs),
                "epochs_ran": eval_epochs_ran,
                "selection_epochs_ran": selection_epochs_ran,
                "best_epoch": fold_best_epoch,
                "best_val_loss": fold_best_val_loss,
                "best_val_acc": fold_best_val_acc,
                "stopped_early": bool(selection_epochs_ran < epochs),
                "lr": float(lr),
                "batch_size": int(batch_size),
                "patience": int(patience),
                "aux_mode": aux_mode,
                "lambda_aux": float(lambda_aux),
                "eegdg_conditional_weight": float(eegdg_conditional_weight),
                "label_smoothing": float(label_smoothing),
                "use_class_weights": bool(use_class_weights),
                "use_weighted_sampler": bool(use_weighted_sampler),
                "max_time_shift": int(max_time_shift),
                "noise_std": float(noise_std),
                "grad_clip_norm": float(grad_clip_norm),
                "aux_warmup_epochs": int(aux_warmup_epochs),
            }
        )
        metrics.update(_class_count_fields("train", fold.y_train, n_classes))
        metrics.update(_class_count_fields("val", fold.y_val, n_classes))
        if protocol == "subject_dependent_te_final":
            metrics.update(
                {
                    "selection_best_epoch": selection_best_epoch,
                    "selection_best_val_loss": selection_best_val_loss,
                    "selection_train_trials": int(len(fold_raw.y_train)),
                    "selection_val_trials": int(len(fold_raw.y_val)),
                    "final_train_trials": final_train_trials,
                    "final_epochs_ran": eval_epochs_ran,
                }
            )
        all_metrics.append(metrics)

        print(
            f"[{model_name}] fold {fold_idx} done | "
            f"accuracy={metrics['accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f} "
            f"kappa={metrics['kappa']:.4f}"
        )

        fold_dir.mkdir(parents=True, exist_ok=True)

        metrics_json = dict(metrics)
        metrics_json["split_subject_counts"] = {
            "train": _value_counts(fold.sid_train),
            "val": _value_counts(fold.sid_val),
            "test": _value_counts(fold.sid_test),
        }

        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, indent=2)

        save_confusion_matrix(y_true, y_pred, fold_dir / "confusion_matrix.png")
        save_confusion_matrix_values(y_true, y_pred, fold_dir / "confusion_matrix.csv", n_classes=n_classes)
        save_predictions(y_true, y_pred, fold_dir / "predictions.csv")
        save_training_curve(history_for_plot, fold_dir / "training_curve.png")
        save_history_csv(history_for_plot, fold_dir / "history.csv")
        if selection_history is not None:
            save_training_curve(selection_history, fold_dir / "selection_curve.png")
            save_history_csv(selection_history, fold_dir / "selection_history.csv")

    summary_path = Path(results_root) / model_name.lower() / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(all_metrics)
    avg_row = {
        "model": model_name,
        "seed": int(seed),
        "fold": "mean",
        "test_subject": "all",
        "protocol": protocol,
    }
    for col in df.columns:
        if col in {"fold", "test_subject", "protocol", "model"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            avg_row[col] = df[col].mean()
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    df.to_csv(summary_path, index=False)

    print(
        f"[{model_name}] summary | "
        f"accuracy={avg_row['accuracy']:.4f} "
        f"f1_macro={avg_row['f1_macro']:.4f} "
        f"kappa={avg_row['kappa']:.4f}"
    )

    return df


def run_all_baselines(
    data_dir: str = "data/processed/bcic_iv_2a",
    results_root: str = "results",
    protocol: str = "loso_t",
    **train_kwargs,
):
    import pandas as pd

    compare_rows = []
    fold_tables = []
    for model_name in ALL_MODELS:
        df = train_and_evaluate_model(
            model_name=model_name,
            data_dir=data_dir,
            results_root=results_root,
            protocol=protocol,
            **train_kwargs,
        )
        fold_tables.append(df)
        mean_row = df[df["fold"] == "mean"].iloc[0]
        compare_rows.append(
            {
                "model": model_name,
                "seed": mean_row.get("seed"),
                "accuracy": mean_row["accuracy"],
                "balanced_accuracy": mean_row.get("balanced_accuracy"),
                "precision_macro": mean_row.get("precision_macro"),
                "recall_macro": mean_row.get("recall_macro"),
                "f1_macro": mean_row["f1_macro"],
                "f1_weighted": mean_row.get("f1_weighted"),
                "kappa": mean_row["kappa"],
            }
        )

    comp_df = pd.DataFrame(compare_rows)
    comp_path = Path(results_root) / "baseline_compare.csv"
    comp_df.to_csv(comp_path, index=False)

    if fold_tables:
        all_fold_df = pd.concat(fold_tables, ignore_index=True)
        all_fold_df.to_csv(Path(results_root) / "all_fold_metrics.csv", index=False)

    save_model_comparison_plots(results_root=results_root, model_names=ALL_MODELS)

    print(f"[baseline] comparison saved to {comp_path}")
    return comp_df
