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
    sid_train: np.ndarray | None = None
    sid_val: np.ndarray | None = None
    sid_test: np.ndarray | None = None
    x_train_full: np.ndarray | None = None
    y_train_full: np.ndarray | None = None
    sid_train_full: np.ndarray | None = None


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

        train_x_list, train_y_list, train_sid_list = [], [], []
        for sid_idx, sid in enumerate(subjects):
            if sid == test_sid:
                continue
            x, y = subject_data[sid]
            train_x_list.append(x)
            train_y_list.append(y)
            train_sid_list.append(np.full((len(y),), sid_idx, dtype=np.int64))

        x_train_all = np.concatenate(train_x_list, axis=0)
        y_train_all = np.concatenate(train_y_list, axis=0)
        sid_train_all = np.concatenate(train_sid_list, axis=0)

        idx = np.arange(len(y_train_all))
        train_idx, val_idx = train_test_split(
            idx, test_size=val_ratio, random_state=seed, stratify=y_train_all
        )

        fold_data = FoldData(
            x_train=x_train_all[train_idx],
            y_train=y_train_all[train_idx],
            sid_train=sid_train_all[train_idx],
            x_val=x_train_all[val_idx],
            y_val=y_train_all[val_idx],
            sid_val=sid_train_all[val_idx],
            x_test=x_test,
            y_test=y_test,
            sid_test=np.full((len(y_test),), subjects.index(test_sid), dtype=np.int64),
        )
        folds.append((test_sid, fold_data))
    return folds


def _split_train_eval_sessions(
    subject_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    train_sessions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    eval_sessions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for sid, data in subject_data.items():
        sid_upper = sid.upper()
        subject = sid_upper[:-1]
        if sid_upper.endswith("T"):
            train_sessions[subject] = data
        elif sid_upper.endswith("E"):
            eval_sessions[subject] = data

    missing_train = sorted(set(eval_sessions) - set(train_sessions))
    missing_eval = sorted(set(train_sessions) - set(eval_sessions))
    if missing_train or missing_eval:
        raise ValueError(
            "Expected matched AxxT/AxxE processed files. "
            f"missing_train={missing_train}, missing_eval={missing_eval}"
        )
    if not train_sessions:
        raise ValueError("No matched AxxT/AxxE sessions found in processed data.")

    return train_sessions, eval_sessions


def build_subject_dependent_te_folds(
    subject_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> List[Tuple[str, FoldData]]:
    """Official-style per-subject protocol: train on AxxT, test on AxxE."""
    train_sessions, eval_sessions = _split_train_eval_sessions(subject_data)
    folds = []

    for subject in sorted(train_sessions):
        x_train_all, y_train_all = train_sessions[subject]
        x_test, y_test = eval_sessions[subject]

        idx = np.arange(len(y_train_all))
        train_idx, val_idx = train_test_split(
            idx, test_size=val_ratio, random_state=seed, stratify=y_train_all
        )

        folds.append(
            (
                f"{subject}E",
                FoldData(
                    x_train=x_train_all[train_idx],
                    y_train=y_train_all[train_idx],
                    x_val=x_train_all[val_idx],
                    y_val=y_train_all[val_idx],
                    x_test=x_test,
                    y_test=y_test,
                    sid_train=np.zeros((len(train_idx),), dtype=np.int64),
                    sid_val=np.zeros((len(val_idx),), dtype=np.int64),
                    sid_test=np.zeros((len(y_test),), dtype=np.int64),
                ),
            )
        )

    return folds


def build_subject_dependent_te_final_folds(
    subject_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> List[Tuple[str, FoldData]]:
    """Subject-dependent final protocol.

    The split AxxT train/val subset is kept for selecting the best epoch.
    The full AxxT session is also stored so the trainer can refit on all
    288 training trials before evaluating on AxxE.
    """
    train_sessions, eval_sessions = _split_train_eval_sessions(subject_data)
    folds = []

    for subject in sorted(train_sessions):
        x_train_all, y_train_all = train_sessions[subject]
        x_test, y_test = eval_sessions[subject]

        idx = np.arange(len(y_train_all))
        train_idx, val_idx = train_test_split(
            idx, test_size=val_ratio, random_state=seed, stratify=y_train_all
        )

        folds.append(
            (
                f"{subject}E",
                FoldData(
                    x_train=x_train_all[train_idx],
                    y_train=y_train_all[train_idx],
                    x_val=x_train_all[val_idx],
                    y_val=y_train_all[val_idx],
                    x_test=x_test,
                    y_test=y_test,
                    sid_train=np.zeros((len(train_idx),), dtype=np.int64),
                    sid_val=np.zeros((len(val_idx),), dtype=np.int64),
                    sid_test=np.zeros((len(y_test),), dtype=np.int64),
                    x_train_full=x_train_all,
                    y_train_full=y_train_all,
                    sid_train_full=np.zeros((len(y_train_all),), dtype=np.int64),
                ),
            )
        )

    return folds


def build_loso_train_eval_folds(
    subject_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> List[Tuple[str, FoldData]]:
    """Strict subject-independent protocol: source AxxT subjects -> target AxxE."""
    train_sessions, eval_sessions = _split_train_eval_sessions(subject_data)
    subjects = sorted(train_sessions)
    folds = []

    for test_subject in subjects:
        x_test, y_test = eval_sessions[test_subject]

        train_x_list, train_y_list, train_sid_list = [], [], []
        for sid_idx, subject in enumerate(subjects):
            if subject == test_subject:
                continue
            x, y = train_sessions[subject]
            train_x_list.append(x)
            train_y_list.append(y)
            train_sid_list.append(np.full((len(y),), sid_idx, dtype=np.int64))

        x_train_all = np.concatenate(train_x_list, axis=0)
        y_train_all = np.concatenate(train_y_list, axis=0)
        sid_train_all = np.concatenate(train_sid_list, axis=0)

        idx = np.arange(len(y_train_all))
        train_idx, val_idx = train_test_split(
            idx, test_size=val_ratio, random_state=seed, stratify=y_train_all
        )

        folds.append(
            (
                f"{test_subject}E",
                FoldData(
                    x_train=x_train_all[train_idx],
                    y_train=y_train_all[train_idx],
                    sid_train=sid_train_all[train_idx],
                    x_val=x_train_all[val_idx],
                    y_val=y_train_all[val_idx],
                    sid_val=sid_train_all[val_idx],
                    x_test=x_test,
                    y_test=y_test,
                    sid_test=np.full((len(y_test),), subjects.index(test_subject), dtype=np.int64),
                ),
            )
        )

    return folds


def normalize_by_train_stats(fold: FoldData) -> FoldData:
    mean = fold.x_train.mean(axis=(0, 2), keepdims=True)
    std = fold.x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    return FoldData(
        x_train=(fold.x_train - mean) / std,
        y_train=fold.y_train,
        sid_train=fold.sid_train,
        x_val=(fold.x_val - mean) / std,
        y_val=fold.y_val,
        sid_val=fold.sid_val,
        x_test=(fold.x_test - mean) / std,
        y_test=fold.y_test,
        sid_test=fold.sid_test,
        x_train_full=None if fold.x_train_full is None else (fold.x_train_full - mean) / std,
        y_train_full=fold.y_train_full,
        sid_train_full=fold.sid_train_full,
    )
