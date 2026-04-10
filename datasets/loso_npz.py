from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


AGGREGATED_SUBJECT_STEM_KEYWORDS = {
    "all_subject",
    "all_subjects",
    "allsubjects",
    "all-subject",
    "all-subjects",
}


@dataclass
class FoldData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class EEGDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _read_subject_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    x_key = "x" if "x" in d.files else "X"
    y_key = "y" if "y" in d.files else "Y"
    x = d[x_key]
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0, :, :]
    if x.ndim != 3:
        raise ValueError(f"Expected x shape [N, C, T] or [N, 1, C, T], got {x.shape} in {path}")
    y = d[y_key]
    if y.ndim > 1:
        y = y.squeeze()
    return x.astype(np.float32), y.astype(np.int64)


def _candidate_processed_dirs(processed_dir: str | Path) -> List[Path]:
    raw_path = Path(processed_dir)
    repo_root = Path(__file__).resolve().parents[1]
    candidates: List[Path] = []

    for candidate in [raw_path, repo_root / raw_path]:
        if candidate not in candidates:
            candidates.append(candidate)

    current_default = Path("data/processed/bcic_iv_2a")
    legacy_default = Path("scripts/data/processed/bcic_iv_2a")

    if raw_path in {current_default, legacy_default} or raw_path.as_posix() in {
        current_default.as_posix(),
        legacy_default.as_posix(),
    }:
        for candidate in [repo_root / current_default, repo_root / legacy_default]:
            if candidate not in candidates:
                candidates.append(candidate)

    return candidates


def load_subject_data(processed_dir: str | Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    checked_paths = []
    for candidate_dir in _candidate_processed_dirs(processed_dir):
        checked_paths.append(str(candidate_dir))
        files = sorted(candidate_dir.glob("*.npz"))
        if not files:
            continue

        subject_files = []
        for file_path in files:
            stem_lower = file_path.stem.lower()
            if any(keyword in stem_lower for keyword in AGGREGATED_SUBJECT_STEM_KEYWORDS):
                print(f"[data] skip aggregated file: {file_path.name}")
                continue
            subject_files.append(file_path)

        if not subject_files:
            continue

        subject_data = {}
        for file_path in subject_files:
            sid = file_path.stem
            subject_data[sid] = _read_subject_npz(file_path)
        print(f"[data] loaded {len(subject_data)} subject files from {candidate_dir}")
        return subject_data

    checked = "\n".join(f"- {path}" for path in checked_paths)
    raise FileNotFoundError(
        "No .npz files found. Checked these directories:\n"
        f"{checked}\n"
        "Please pass --data_dir explicitly if your processed files are stored elsewhere."
    )


def build_loso_folds(
    subject_data: Dict[str, Tuple[np.ndarray, np.ndarray]], val_ratio: float = 0.2, seed: int = 42
) -> List[Tuple[str, FoldData]]:
    folds = []
    subjects = sorted(subject_data.keys())

    for test_sid in subjects:
        x_test, y_test = subject_data[test_sid]

        train_x_list, train_y_list = [], []
        for sid in subjects:
            if sid == test_sid:
                continue
            x, y = subject_data[sid]
            train_x_list.append(x)
            train_y_list.append(y)

        x_train_all = np.concatenate(train_x_list, axis=0)
        y_train_all = np.concatenate(train_y_list, axis=0)

        idx = np.arange(len(y_train_all))
        train_idx, val_idx = train_test_split(
            idx, test_size=val_ratio, random_state=seed, stratify=y_train_all
        )

        fold_data = FoldData(
            x_train=x_train_all[train_idx],
            y_train=y_train_all[train_idx],
            x_val=x_train_all[val_idx],
            y_val=y_train_all[val_idx],
            x_test=x_test,
            y_test=y_test,
        )
        folds.append((test_sid, fold_data))
    return folds


def normalize_by_train_stats(fold: FoldData) -> FoldData:
    mean = fold.x_train.mean(axis=(0, 2), keepdims=True)
    std = fold.x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    return FoldData(
        x_train=(fold.x_train - mean) / std,
        y_train=fold.y_train,
        x_val=(fold.x_val - mean) / std,
        y_val=fold.y_val,
        x_test=(fold.x_test - mean) / std,
        y_test=fold.y_test,
    )
