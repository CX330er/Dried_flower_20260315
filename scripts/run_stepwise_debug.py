"""Run the recommended stepwise baseline-debug workflow.

Steps:
1) Overfit check (force small-sample memorization).
2) Clean protocol check (subject-dependent / mixed / optional LOSO).
3) Optional preprocessing window sweep table.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], execute: bool) -> int:
    print("$", " ".join(cmd))
    if not execute:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def _load_protocol_metrics(debug_dir: Path) -> dict:
    p = debug_dir / "protocol_comparison_metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stepwise baseline debugging runner")
    parser.add_argument("--model", type=str, default="EEGNet")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--base_processed_dir", type=str, default="data/processed/bcic_iv_2a")
    parser.add_argument("--results_root", type=str, default="results/stepwise_debug")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default only prints commands.")
    parser.add_argument("--run_loso", action="store_true")
    parser.add_argument("--run_window_sweep", action="store_true")
    args = parser.parse_args()

    results_root = REPO_ROOT / args.results_root
    results_root.mkdir(parents=True, exist_ok=True)

    # Step 1: strict overfit on tiny sample.
    step1_dir = results_root / "step1_overfit"
    step1_dir.mkdir(parents=True, exist_ok=True)
    step1_cmd = [
        "python",
        "scripts/validate_baseline_debug.py",
        "--model",
        args.model,
        "--data_dir",
        args.base_processed_dir,
        "--results_root",
        str(step1_dir),
        "--epochs",
        "200",
        "--lr",
        "0.002",
        "--overfit_trials",
        "40",
        "--batch_size",
        "16",
    ]
    if args.run_loso:
        step1_cmd.append("--run_loso")
    _run(step1_cmd, execute=args.execute)

    # Step 2: clean protocol check (same data, same model) for reproducible baseline.
    step2_dir = results_root / "step2_protocol"
    step2_dir.mkdir(parents=True, exist_ok=True)
    step2_cmd = [
        "python",
        "scripts/validate_baseline_debug.py",
        "--model",
        args.model,
        "--data_dir",
        args.base_processed_dir,
        "--results_root",
        str(step2_dir),
        "--epochs",
        "120",
        "--lr",
        "0.001",
        "--overfit_trials",
        "40",
        "--batch_size",
        "32",
    ]
    if args.run_loso:
        step2_cmd.append("--run_loso")
    _run(step2_cmd, execute=args.execute)

    # Step 3: optional time-window sweep table.
    if args.run_window_sweep:
        windows = [(0.5, 4.5), (1.0, 5.0), (1.5, 5.5), (2.0, 6.0)]
        csv_path = results_root / "window_sweep_protocols.csv"
        rows = []

        for tmin, tmax in windows:
            tag = f"t{tmin:.1f}_{tmax:.1f}".replace(".", "p")
            out_processed = f"data/processed/bcic_iv_2a_{tag}"
            debug_dir = results_root / f"step3_{tag}"
            debug_dir.mkdir(parents=True, exist_ok=True)

            preprocess_cmd = [
                "python",
                "scripts/process_bcic_iv_2a.py",
                "--raw-dir",
                args.raw_dir,
                "--out-dir",
                out_processed,
                "--tmin",
                str(tmin),
                "--tmax",
                str(tmax),
                "--replace-out-dir",
            ]
            _run(preprocess_cmd, execute=args.execute)

            debug_cmd = [
                "python",
                "scripts/validate_baseline_debug.py",
                "--model",
                args.model,
                "--data_dir",
                out_processed,
                "--results_root",
                str(debug_dir),
                "--epochs",
                "120",
                "--lr",
                "0.001",
                "--batch_size",
                "32",
                "--run_loso",
            ]
            _run(debug_cmd, execute=args.execute)

            if args.execute:
                metrics = _load_protocol_metrics(debug_dir)
                rows.append(
                    {
                        "window": f"{tmin:.1f}-{tmax:.1f}",
                        "subject_dependent_mean_acc": metrics.get("subject_dependent_mean_acc", ""),
                        "mixed_subject_random_split_acc": metrics.get("mixed_subject_random_split_acc", ""),
                        "loso_mean_acc": metrics.get("loso_mean_acc", ""),
                    }
                )

        if args.execute and rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[done] window sweep table: {csv_path}")


if __name__ == "__main__":
    main()
