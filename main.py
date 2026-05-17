import argparse
import sys

from trainers.baseline_trainer import PROTOCOL_CHOICES, run_all_baselines, train_and_evaluate_model
from utils.run_naming import make_dated_results_root, write_run_config

DEFAULT_DATA_DIR = "data/processed/bcic_iv_2a"

def _add_argument_once(parser: argparse.ArgumentParser, *name_or_flags, **kwargs):
    """Avoid argparse duplicate-option crashes when local code has repeated flags."""
    existing = set(parser._option_string_actions.keys())
    option_flags = [flag for flag in name_or_flags if isinstance(flag, str) and flag.startswith("-")]
    if any(flag in existing for flag in option_flags):
        return
    parser.add_argument(*name_or_flags, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG baseline training entrypoint")
    _add_argument_once(parser, "--model", type=str, default="all", choices=["all", "ShallowConvNet", "DeepConvNet", "EEGNet", "FBCNet", "MSFBCNN", "EEGNetFSFE"])
    _add_argument_once(parser, "--data_dir", type=str, default=DEFAULT_DATA_DIR)
    _add_argument_once(parser, "--results_root", type=str, default=None)
    _add_argument_once(parser, "--run_timestamp", type=str, default=None, help="Optional date tag for automatic result names, e.g. 2026-05-13_1125.")
    _add_argument_once(parser, "--protocol", type=str, default="loso_t", choices=PROTOCOL_CHOICES)
    _add_argument_once(parser, "--seed", type=int, default=42)
    _add_argument_once(parser, "--aux_mode", type=str, default="none", choices=["none", "center", "coral", "eegdg"])
    _add_argument_once(parser, "--lambda_aux", type=float, default=0.02)
    _add_argument_once(parser, "--eegdg_conditional_weight", type=float, default=1.0)
    _add_argument_once(parser, "--label_smoothing", type=float, default=0.1)
    _add_argument_once(parser, "--disable_class_weights", action="store_true")
    _add_argument_once(parser, "--disable_weighted_sampler", action="store_true")
    _add_argument_once(parser, "--max_time_shift", type=int, default=25)
    _add_argument_once(parser, "--noise_std", type=float, default=0.01)
    _add_argument_once(parser, "--grad_clip_norm", type=float, default=1.0)
    _add_argument_once(parser, "--aux_warmup_epochs", type=int, default=30)
    _add_argument_once(parser, "--resume", action="store_true", help="Skip fold runs that already have metrics.json in results_root.")
    args = parser.parse_args()

    if args.results_root is None:
        args.results_root = str(
            make_dated_results_root(
                "results",
                name_parts=[args.model, args.protocol],
                run_timestamp=args.run_timestamp,
            )
        )

    write_run_config(
        args.results_root,
        {
            "entrypoint": "main.py",
            "argv": sys.argv,
            "model": args.model,
            "protocol": args.protocol,
            "data_dir": args.data_dir,
            "seed": args.seed,
            "aux_mode": args.aux_mode,
            "lambda_aux": args.lambda_aux,
            "eegdg_conditional_weight": args.eegdg_conditional_weight,
            "label_smoothing": args.label_smoothing,
            "use_class_weights": not args.disable_class_weights,
            "use_weighted_sampler": not args.disable_weighted_sampler,
            "max_time_shift": args.max_time_shift,
            "noise_std": args.noise_std,
            "grad_clip_norm": args.grad_clip_norm,
            "aux_warmup_epochs": args.aux_warmup_epochs,
            "resume": args.resume,
        },
    )
    print(f"[results] {args.results_root}")

    if args.model == "all":
        run_all_baselines(
            data_dir=args.data_dir,
            results_root=args.results_root,
            protocol=args.protocol,
            seed=args.seed,
            aux_mode=args.aux_mode,
            lambda_aux=args.lambda_aux,
            eegdg_conditional_weight=args.eegdg_conditional_weight,
            label_smoothing=args.label_smoothing,
            use_class_weights=not args.disable_class_weights,
            use_weighted_sampler=not args.disable_weighted_sampler,
            max_time_shift=args.max_time_shift,
            noise_std=args.noise_std,
            grad_clip_norm=args.grad_clip_norm,
            aux_warmup_epochs=args.aux_warmup_epochs,
            resume_existing=args.resume,
        )
    else:
        train_and_evaluate_model(
            model_name=args.model,
            data_dir=args.data_dir,
            results_root=args.results_root,
            protocol=args.protocol,
            seed=args.seed,
            aux_mode=args.aux_mode,
            lambda_aux=args.lambda_aux,
            eegdg_conditional_weight=args.eegdg_conditional_weight,
            label_smoothing=args.label_smoothing,
            use_class_weights=not args.disable_class_weights,
            use_weighted_sampler=not args.disable_weighted_sampler,
            max_time_shift=args.max_time_shift,
            noise_std=args.noise_std,
            grad_clip_norm=args.grad_clip_norm,
            aux_warmup_epochs=args.aux_warmup_epochs,
            resume_existing=args.resume,
        )
