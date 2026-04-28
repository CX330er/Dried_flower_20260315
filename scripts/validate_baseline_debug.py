"""Baseline diagnosis utilities for BCIC-IV-2a.

Implements four requested checks:
1) Label alignment / sanity check.
2) Small-sample overfitting test.
3) Subject-dependent vs mixed-subject vs LOSO protocols.
4) Training-curve export (train/val loss + train/val acc).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.loso_npz import build_loso_folds, load_subject_data, normalize_by_train_stats
from trainers.baseline_trainer import MODEL_REGISTRY
from utils.metrics import save_training_curve
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline diagnostics for MI EEG project")
    parser.add_argument("--model", type=str, default="EEGNet", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_dir", type=str, default="data/processed/bcic_iv_2a")
    parser.add_argument("--results_root", type=str, default="results/debug")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--overfit_trials", type=int, default=40)
    parser.add_argument("--run_loso", action="store_true", help="Run full LOSO diagnostic (slower)")
    return parser.parse_args()


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _fit_once(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> Tuple[dict, float]:
    model = MODEL_REGISTRY[model_name](
        n_channels=x_train.shape[1],
        n_classes=int(np.max(y_train)) + 1,
        input_time=x_train.shape[-1],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    tr_loader = _make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    va_loader = _make_loader(x_val, y_val, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for _ in tqdm(range(epochs), leave=False):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_n += yb.size(0)

        train_loss = total_loss / max(total_n, 1)
        train_acc = total_correct / max(total_n, 1)

        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_total += loss.item() * yb.size(0)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_n += yb.size(0)

        val_loss = val_loss_total / max(val_n, 1)
        val_acc = val_correct / max(val_n, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

    return history, history["val_acc"][-1]


def check_label_alignment(subject_data: Dict[str, Tuple[np.ndarray, np.ndarray]], out_dir: Path) -> dict:
    per_subject = {}
    x_all, y_all = [], []

    for sid, (x, y) in sorted(subject_data.items()):
        uniq, cnt = np.unique(y, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
        per_subject[sid] = {
            "n_trials": int(len(y)),
            "label_set": [int(v) for v in uniq.tolist()],
            "label_distribution": dist,
            "is_label_range_valid": bool(np.all(np.isin(y, [0, 1, 2, 3]))),
        }
        x_all.append(x)
        y_all.append(y)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Shift-sanity: if labels are badly misaligned, true-label accuracy may fail to
    # outperform a 1-step shifted-label baseline on simple linear probe.
    x_flat = x_all.reshape(x_all.shape[0], -1)
    x_tr, x_te, y_tr, y_te = train_test_split(x_flat, y_all, test_size=0.3, random_state=42, stratify=y_all)
    clf = LogisticRegression(max_iter=500, n_jobs=1, multi_class="auto")
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_te)
    acc_true = float(accuracy_score(y_te, y_pred))

    y_shift = np.roll(y_all, 1)
    x_tr2, x_te2, y_tr2, y_te2 = train_test_split(x_flat, y_shift, test_size=0.3, random_state=42, stratify=y_shift)
    clf2 = LogisticRegression(max_iter=500, n_jobs=1, multi_class="auto")
    clf2.fit(x_tr2, y_tr2)
    y_pred2 = clf2.predict(x_te2)
    acc_shift = float(accuracy_score(y_te2, y_pred2))

    report = {
        "per_subject": per_subject,
        "linear_probe_true_label_acc": acc_true,
        "linear_probe_shifted_label_acc": acc_shift,
        "alignment_warning": bool(acc_true <= acc_shift + 0.02),
    }

    out_path = out_dir / "label_alignment_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def overfit_small_sample(
    model_name: str,
    subject_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    overfit_trials: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    out_dir: Path,
) -> dict:
    sid = sorted(subject_data.keys())[0]
    x, y = subject_data[sid]

    n = min(overfit_trials, len(y))
    x_sub = x[:n]
    y_sub = y[:n]

    history, _ = _fit_once(
        model_name=model_name,
        x_train=x_sub,
        y_train=y_sub,
        x_val=x_sub,
        y_val=y_sub,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )

    save_training_curve(history, out_dir / "overfit_small_sample_curve.png")
    result = {
        "subject": sid,
        "n_trials": int(n),
        "final_train_acc": float(history["train_acc"][-1]),
        "final_train_loss": float(history["train_loss"][-1]),
        "can_overfit_95": bool(history["train_acc"][-1] >= 0.95),
    }
    (out_dir / "overfit_small_sample_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_protocols(
    model_name: str,
    subject_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    out_dir: Path,
    run_loso: bool,
) -> dict:
    results = {}

    # A) subject-dependent: train/test split inside each subject, then average.
    sd_acc = []
    for sid, (x, y) in sorted(subject_data.items()):
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
        history, acc = _fit_once(model_name, x[tr], y[tr], x[te], y[te], epochs, lr, batch_size, device)
        save_training_curve(history, out_dir / f"subject_dependent_{sid}_curve.png")
        sd_acc.append(acc)
    results["subject_dependent_mean_acc"] = float(np.mean(sd_acc))

    # B) mixed-subject random split.
    x_all = np.concatenate([v[0] for v in subject_data.values()], axis=0)
    y_all = np.concatenate([v[1] for v in subject_data.values()], axis=0)
    idx = np.arange(len(y_all))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_all)
    history, acc = _fit_once(model_name, x_all[tr], y_all[tr], x_all[te], y_all[te], epochs, lr, batch_size, device)
    save_training_curve(history, out_dir / "mixed_subject_curve.png")
    results["mixed_subject_random_split_acc"] = float(acc)

    # C) LOSO
    if run_loso:
        folds = build_loso_folds(subject_data, val_ratio=0.2, seed=42)
        loso_acc = []
        for i, (_test_sid, fold_raw) in enumerate(folds, start=1):
            fold = normalize_by_train_stats(fold_raw)
            history, acc = _fit_once(
                model_name,
                fold.x_train,
                fold.y_train,
                fold.x_test,
                fold.y_test,
                epochs,
                lr,
                batch_size,
                device,
            )
            save_training_curve(history, out_dir / f"loso_fold{i}_curve.png")
            loso_acc.append(acc)
        results["loso_mean_acc"] = float(np.mean(loso_acc))
        results["loso_fold_acc"] = [float(v) for v in loso_acc]

    (out_dir / "protocol_comparison_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.results_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_data = load_subject_data(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_report = check_label_alignment(subject_data, out_dir)
    overfit_report = overfit_small_sample(
        model_name=args.model,
        subject_data=subject_data,
        overfit_trials=args.overfit_trials,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        out_dir=out_dir,
    )
    protocol_report = evaluate_protocols(
        model_name=args.model,
        subject_data=subject_data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        out_dir=out_dir,
        run_loso=args.run_loso,
    )

    summary = {
        "label_alignment": label_report,
        "overfit_small_sample": overfit_report,
        "protocols": protocol_report,
    }
    (out_dir / "debug_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[debug] label true vs shifted acc:", label_report["linear_probe_true_label_acc"], label_report["linear_probe_shifted_label_acc"])
    print("[debug] overfit final train acc:", overfit_report["final_train_acc"])
    print("[debug] protocol metrics:", protocol_report)
    print("[debug] outputs saved to", out_dir)


if __name__ == "__main__":
    main()
