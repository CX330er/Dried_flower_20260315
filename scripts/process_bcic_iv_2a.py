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

import numpy as np

from datasets.bcic_iv_2a_reader import (
    DEFAULT_REJECT_MARKERS,
    NoCueEventsError,
    ProcessConfig,
    collect_subject_session_id,
    iter_raw_files,
    preprocess_one_file,
)


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
        default="data/processed/bcic_iv_2a",
        help="Directory to save processed .npz and stats report",
    )
    parser.add_argument("--l-freq", type=float, default=5.0)
    parser.add_argument("--h-freq", type=float, default=30.0)
    parser.add_argument("--sfreq", type=int, default=250)
    parser.add_argument("--tmin", type=float, default=2.0)
    parser.add_argument("--tmax", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = ProcessConfig(
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        resample_sfreq=args.sfreq,
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=None,
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
            "reject_markers": sorted(DEFAULT_REJECT_MARKERS),
        },
        "files": [],
        "skipped_files": [],
        "total_trials": 0,
    }

    merged_x: List[np.ndarray] = []
    merged_y: List[np.ndarray] = []
    merged_group: List[str] = []

    for file_path in all_files:
        try:
            x, y, meta = preprocess_one_file(file_path=file_path, config=config)
        except NoCueEventsError as err:
            sid = collect_subject_session_id(file_path)
            report["skipped_files"].append(
                {
                    "file": file_path.name,
                    "subject_session": sid,
                    "reason": str(err),
                }
            )
            print(f"[SKIP] {file_path.name}: {err}")
            continue

        sid = collect_subject_session_id(file_path)

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
        "n_channels": int(x_all.shape[1]),
        "n_times": int(x_all.shape[2]),
        "label_distribution": {str(k): int((y_all == k).sum()) for k in np.unique(y_all)},
    }

    report_path = out_dir / "data_stats.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved merged dataset to: {out_dir / 'all_subjects.npz'}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
