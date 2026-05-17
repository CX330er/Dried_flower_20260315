from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainers.baseline_trainer import PROTOCOL_CHOICES, run_all_baselines
from utils.run_naming import make_dated_results_root, write_run_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all baseline models with date-based result naming.")
    parser.add_argument("--data_dir", type=str, default="data/processed/bcic_iv_2a")
    parser.add_argument("--results_root", type=str, default=None)
    parser.add_argument("--run_timestamp", type=str, default=None, help="Optional date tag for automatic result names, e.g. 2026-05-13_1125.")
    parser.add_argument("--protocol", type=str, default="loso_t", choices=PROTOCOL_CHOICES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Skip fold runs that already have metrics.json in results_root.")
    args = parser.parse_args()

    if args.results_root is None:
        args.results_root = str(
            make_dated_results_root(
                "results",
                name_parts=["all", args.protocol],
                run_timestamp=args.run_timestamp,
            )
        )

    write_run_config(
        args.results_root,
        {
            "entrypoint": "scripts/train_baselines.py",
            "argv": sys.argv,
            "model": "all",
            "protocol": args.protocol,
            "data_dir": args.data_dir,
            "seed": args.seed,
            "resume": args.resume,
        },
    )
    print(f"[results] {args.results_root}")
    run_all_baselines(
        data_dir=args.data_dir,
        results_root=args.results_root,
        protocol=args.protocol,
        seed=args.seed,
        resume_existing=args.resume,
    )
