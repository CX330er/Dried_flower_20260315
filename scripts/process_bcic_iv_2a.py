"""Process BCIC-IV-2a .gdf files into unified numpy datasets.

Usage example:
python scripts/process_bcic_iv_2a.py \
  --raw-dir "D:\\PycharmProjects\\Dried_Flower\\data\\raw" \
  --out-dir "data/processed/bcic_iv_2a"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import shutil
import sys
import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.bcic_iv_2a_reader import (
    DEFAULT_REJECT_MARKERS,
    NoCueEventsError,
    ProcessConfig,
    collect_subject_session_id,
    iter_raw_files,
    preprocess_one_file,
)


def _time_tag(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def _value_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _window_tag(tmin: float, tmax: float) -> str:
    return f"window_{_time_tag(tmin)}_{_time_tag(tmax)}"


def _band_tag(l_freq: float, h_freq: float) -> str:
    return f"band_{_value_tag(l_freq)}_{_value_tag(h_freq)}hz"


def _session_tag(session_mode: str) -> str:
    return {
        "train": "train_only",
        "eval": "eval_only",
        "all": "train_eval",
    }[session_mode]


def _reject_tag(reject_bad_trials: str) -> str:
    if reject_bad_trials == "trial_start_to_tmax":
        return "_reject1023"
    if reject_bad_trials == "cue_to_tmax":
        return "_reject1023_cue_window"
    return ""


def default_processed_dir(
    session_mode: str,
    tmin: float,
    tmax: float,
    l_freq: float,
    h_freq: float,
    reject_bad_trials: str,
) -> Path:
    name = (
        f"bcic_iv_2a_{_session_tag(session_mode)}_{_window_tag(tmin, tmax)}_"
        f"{_band_tag(l_freq, h_freq)}{_reject_tag(reject_bad_trials)}"
    )
    return Path("data") / "processed" / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BCIC-IV-2a preprocessing pipeline")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=r"D:\PycharmProjects\Dried_Flower\data\raw",
        help="Directory containing BCIC-IV-2a .gdf files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save processed .npz and stats report. Defaults to a descriptive protocol/window/band name.",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default="data/raw/true_labels",
        help="Directory containing official .mat labels for evaluation sessions",
    )
    parser.add_argument(
        "--session-mode",
        choices=["train", "eval", "all"],
        default="train",
        help="Process training sessions only, evaluation sessions only, or both.",
    )
    parser.add_argument("--l-freq", type=float, default=4.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--sfreq", type=int, default=250)
    parser.add_argument("--tmin", type=float, default=0.5)
    parser.add_argument("--tmax", type=float, default=4.5)
    parser.add_argument("--butter-order", type=int, default=4)
    parser.add_argument(
        "--reject-bad-trials",
        choices=["none", "cue_to_tmax", "trial_start_to_tmax"],
        default="trial_start_to_tmax",
        help=(
            "How to reject BCIC-IV-2a 1023 bad-trial markers. "
            "trial_start_to_tmax drops a trial if 1023 occurs from event 768 "
            "to cue+tmax, matching the usual rejected-trial interpretation."
        ),
    )
    parser.add_argument(
        "--replace-out-dir",
        action="store_true",
        help="If set, delete existing out-dir before writing newly processed files.",
    )
    return parser.parse_args()


def _session_allowed(sid: str, session_mode: str) -> bool:
    if session_mode == "all":
        return sid.endswith(("T", "E"))
    if session_mode == "train":
        return sid.endswith("T")
    if session_mode == "eval":
        return sid.endswith("E")
    raise ValueError(f"Unsupported session_mode={session_mode}")


def _load_external_labels(label_dir: Path, sid: str) -> np.ndarray:
    label_path = label_dir / f"{sid}.mat"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing official label file for {sid}: {label_path}")
    data = loadmat(label_path)
    if "classlabel" not in data:
        keys = sorted(k for k in data if not k.startswith("__"))
        raise KeyError(f"Expected 'classlabel' in {label_path}; available keys={keys}")
    return np.asarray(data["classlabel"]).squeeze()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    label_dir = Path(args.label_dir)
    if args.out_dir is None:
        args.out_dir = str(
            default_processed_dir(
                args.session_mode,
                args.tmin,
                args.tmax,
                args.l_freq,
                args.h_freq,
                args.reject_bad_trials,
            )
        )
    out_dir = Path(args.out_dir)

    if args.replace_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = ProcessConfig(
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        resample_sfreq=args.sfreq,
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=None,
        butter_order=args.butter_order,
        reject_bad_trials=args.reject_bad_trials,
    )

    all_files = iter_raw_files(raw_dir)
    report: Dict[str, object] = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "config": {
            "channels": 22,
            "l_freq": config.l_freq,
            "h_freq": config.h_freq,
            "resample_sfreq": config.resample_sfreq,
            "tmin": config.tmin,
            "tmax": config.tmax,
            "butter_order": config.butter_order,
            "reject_markers": sorted(DEFAULT_REJECT_MARKERS),
            "reject_bad_trials": args.reject_bad_trials,
            "session_mode": args.session_mode,
            "label_dir": str(label_dir),
            "target_shape": "B x 1 x C x T",
        },
        "files": [],
        "skipped_files": [],
        "total_trials": 0,
    }

    merged_x: List[np.ndarray] = []
    merged_y: List[np.ndarray] = []
    merged_group: List[str] = []

    for file_path in all_files:
        sid = collect_subject_session_id(file_path)
        if not _session_allowed(sid, args.session_mode):
            continue

        external_labels = _load_external_labels(label_dir, sid) if sid.endswith("E") else None
        try:
            x, y, meta = preprocess_one_file(
                file_path=file_path,
                config=config,
                external_labels=external_labels,
            )
        except NoCueEventsError as err:
            report["skipped_files"].append(
                {
                    "file": file_path.name,
                    "subject_session": sid,
                    "reason": str(err),
                }
            )
            print(f"[SKIP] {file_path.name}: {err}")
            continue

        np.savez_compressed(
            out_dir / f"{sid}.npz",
            X=x,
            y=y,
            subject_session=np.array([sid] * len(y)),
        )

        merged_x.append(x)
        merged_y.append(y)
        merged_group.extend([sid] * len(y))

        file_record = {
            **meta,
            "subject_session": sid,
            "label_distribution": {str(k): int((y == k).sum()) for k in np.unique(y)},
        }
        report["files"].append(file_record)
        report["total_trials"] = int(report["total_trials"]) + int(len(y))

        print(
            f"Processed {file_path.name}: trials={meta['n_trials']}, "
            f"shape={x.shape}, labels={file_record['label_distribution']}"
        )

    if not merged_x:
        raise RuntimeError(
            "No labeled files were processed. Please check raw files or provide label events (769-772)."
        )

    x_all = np.concatenate(merged_x, axis=0)
    y_all = np.concatenate(merged_y, axis=0)
    groups = np.asarray(merged_group)

    np.savez_compressed(
        out_dir / "all_subjects.npz",
        X=x_all,
        y=y_all,
        subject_session=groups,
    )

    report["merged"] = {
        "n_trials": int(x_all.shape[0]),
        "n_channels": int(x_all.shape[2] if x_all.ndim == 4 else x_all.shape[1]),
        "n_times": int(x_all.shape[3] if x_all.ndim == 4 else x_all.shape[2]),
        "label_distribution": {str(k): int((y_all == k).sum()) for k in np.unique(y_all)},
    }

    report_path = out_dir / "data_stats.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved merged dataset to: {out_dir / 'all_subjects.npz'}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
