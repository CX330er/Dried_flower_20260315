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
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.run_naming import make_dated_results_root, write_run_config


def _time_tag(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def _value_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _window_tag(tmin: float, tmax: float) -> str:
    return f"window_{_time_tag(tmin)}_{_time_tag(tmax)}"


def _band_tag(l_freq: float, h_freq: float) -> str:
    return f"band_{_value_tag(l_freq)}_{_value_tag(h_freq)}hz"


def _processed_dir_name(tmin: float, tmax: float, l_freq: float, h_freq: float) -> str:
    return f"bcic_iv_2a_train_only_{_window_tag(tmin, tmax)}_{_band_tag(l_freq, h_freq)}"


def _python_cmd(*args: str) -> list[str]:
    return [sys.executable, *args]


def _run(cmd: list[str], execute: bool) -> int:
    print("$", " ".join(cmd))
    if not execute:
        return 0
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return completed.returncode


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {description} at {path}, but it was not created.")


def _load_protocol_metrics(debug_dir: Path) -> dict:
    p = debug_dir / "protocol_comparison_metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stepwise baseline debugging runner")
    parser.add_argument("--model", type=str, default="EEGNet")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--base_processed_dir", type=str, default=None)
    parser.add_argument("--results_root", type=str, default=None)
    parser.add_argument("--run_timestamp", type=str, default=None, help="Optional date tag for automatic result names, e.g. 2026-05-13_1125.")
    parser.add_argument("--l_freq", type=float, default=4.0, help="Band-pass low cutoff used in window sweep preprocessing.")
    parser.add_argument("--h_freq", type=float, default=40.0, help="Band-pass high cutoff used in window sweep preprocessing.")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default only prints commands.")
    parser.add_argument("--run_loso", action="store_true")
    parser.add_argument("--run_window_sweep", action="store_true")
    args = parser.parse_args()

    if args.base_processed_dir is None:
        args.base_processed_dir = str(REPO_ROOT / "data" / "processed" / _processed_dir_name(0.5, 4.5, args.l_freq, args.h_freq))
    if args.results_root is None:
        args.results_root = str(
            make_dated_results_root(
                "results",
                name_parts=[args.model, "stepwise_debug"],
                run_timestamp=args.run_timestamp,
            )
        )

    results_root = REPO_ROOT / args.results_root
    results_root.mkdir(parents=True, exist_ok=True)
    write_run_config(
        results_root,
        {
            "entrypoint": "scripts/run_stepwise_debug.py",
            "argv": sys.argv,
            "model": args.model,
            "raw_dir": args.raw_dir,
            "base_processed_dir": args.base_processed_dir,
            "l_freq": args.l_freq,
            "h_freq": args.h_freq,
            "run_loso": args.run_loso,
            "run_window_sweep": args.run_window_sweep,
            "execute": args.execute,
        },
    )
    print(f"[results] {results_root}")

    # Step 1: strict overfit on tiny sample.
    step1_dir = results_root / "step1_overfit"
    step1_dir.mkdir(parents=True, exist_ok=True)
    step1_cmd = _python_cmd(
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
    )
    if args.run_loso:
        step1_cmd.append("--run_loso")
    _run(step1_cmd, execute=args.execute)
    if args.execute:
        _require_file(step1_dir / "debug_summary.json", "step1 debug summary")

    # Step 2: clean protocol check (same data, same model) for reproducible baseline.
    step2_dir = results_root / "step2_protocol"
    step2_dir.mkdir(parents=True, exist_ok=True)
    step2_cmd = _python_cmd(
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
    )
    if args.run_loso:
        step2_cmd.append("--run_loso")
    _run(step2_cmd, execute=args.execute)
    if args.execute:
        _require_file(step2_dir / "debug_summary.json", "step2 debug summary")

    # Step 3: optional time-window sweep table.
    if args.run_window_sweep:
        windows = [(0.5, 4.5), (1.0, 5.0), (1.5, 5.5), (2.0, 6.0)]
        csv_path = results_root / "window_sweep_protocols.csv"
        rows = []

        for tmin, tmax in windows:
            tag = _window_tag(tmin, tmax)
            out_processed = f"data/processed/{_processed_dir_name(tmin, tmax, args.l_freq, args.h_freq)}"
            debug_dir = results_root / f"step3_{tag}"
            debug_dir.mkdir(parents=True, exist_ok=True)

            preprocess_cmd = _python_cmd(
                "scripts/process_bcic_iv_2a.py",
                "--raw-dir",
                args.raw_dir,
                "--out-dir",
                out_processed,
                "--l-freq",
                str(args.l_freq),
                "--h-freq",
                str(args.h_freq),
                "--tmin",
                str(tmin),
                "--tmax",
                str(tmax),
                "--replace-out-dir",
            )
            _run(preprocess_cmd, execute=args.execute)
            if args.execute:
                _require_file(REPO_ROOT / out_processed / "data_stats.json", f"preprocessing stats for {tag}")

            debug_cmd = _python_cmd(
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
            )
            _run(debug_cmd, execute=args.execute)

            if args.execute:
                _require_file(debug_dir / "debug_summary.json", f"window-sweep debug summary for {tag}")
                metrics = _load_protocol_metrics(debug_dir)
                if not metrics:
                    raise FileNotFoundError(
                        f"Expected protocol metrics for {tag} in {debug_dir}, but the file is missing or empty."
                    )
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
