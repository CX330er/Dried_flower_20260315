import argparse
import sys

from trainers.eegdg_full_trainer import train_eegdg_full_loso_te
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
    _add_argument_once(parser, "--model", type=str, default="all", choices=["all", "ShallowConvNet", "DeepConvNet", "EEGNet", "EEGDG", "FBCNet", "MSFBCNN", "EEGNetFSFE"])
    _add_argument_once(parser, "--data_dir", type=str, default=DEFAULT_DATA_DIR)
    _add_argument_once(parser, "--results_root", type=str, default=None)
    _add_argument_once(parser, "--run_timestamp", type=str, default=None, help="Optional date tag for automatic result names, e.g. 2026-05-13_1125.")
    _add_argument_once(parser, "--protocol", type=str, default="loso_t", choices=PROTOCOL_CHOICES)
    _add_argument_once(parser, "--seed", type=int, default=42)
    _add_argument_once(parser, "--aux_mode", type=str, default="none", choices=["none", "center", "coral", "eegdg", "dg_supcon", "eegdg_full"])
    _add_argument_once(parser, "--lambda_aux", type=float, default=0.02)
    _add_argument_once(parser, "--eegdg_conditional_weight", type=float, default=1.0)
    _add_argument_once(parser, "--eegdg_mmd_weight", type=float, default=0.1)
    _add_argument_once(parser, "--eegdg_domain_weight", type=float, default=0.1)
    _add_argument_once(parser, "--eegdg_mcd_weight", type=float, default=0.1)
    _add_argument_once(parser, "--eegdg_mcd_alpha", type=float, default=0.1)
    _add_argument_once(parser, "--eegdg_domain_batch_size", type=int, default=8)
    _add_argument_once(parser, "--eegdg_full_epochs", type=int, default=500)
    _add_argument_once(parser, "--eegdg_full_patience", type=int, default=80)
    _add_argument_once(parser, "--eegdg_full_lr", type=float, default=5e-4)
    _add_argument_once(parser, "--eegdg_full_weight_decay", type=float, default=0.05)
    _add_argument_once(parser, "--supcon_temperature", type=float, default=0.2)
    _add_argument_once(parser, "--label_smoothing", type=float, default=0.1)
    _add_argument_once(parser, "--disable_class_weights", action="store_true")
    _add_argument_once(parser, "--disable_weighted_sampler", action="store_true")
    _add_argument_once(parser, "--same_class_mixup_alpha", type=float, default=0.0)
    _add_argument_once(parser, "--same_class_mixup_prob", type=float, default=0.0)
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
            "eegdg_mmd_weight": args.eegdg_mmd_weight,
            "eegdg_domain_weight": args.eegdg_domain_weight,
            "eegdg_mcd_weight": args.eegdg_mcd_weight,
            "eegdg_mcd_alpha": args.eegdg_mcd_alpha,
            "eegdg_domain_batch_size": args.eegdg_domain_batch_size,
            "eegdg_full_epochs": args.eegdg_full_epochs,
            "eegdg_full_patience": args.eegdg_full_patience,
            "eegdg_full_lr": args.eegdg_full_lr,
            "eegdg_full_weight_decay": args.eegdg_full_weight_decay,
            "supcon_temperature": args.supcon_temperature,
            "label_smoothing": args.label_smoothing,
            "use_class_weights": not args.disable_class_weights,
            "use_weighted_sampler": not args.disable_weighted_sampler,
            "same_class_mixup_alpha": args.same_class_mixup_alpha,
            "same_class_mixup_prob": args.same_class_mixup_prob,
            "max_time_shift": args.max_time_shift,
            "noise_std": args.noise_std,
            "grad_clip_norm": args.grad_clip_norm,
            "aux_warmup_epochs": args.aux_warmup_epochs,
            "resume": args.resume,
        },
    )
    print(f"[results] {args.results_root}")

    if args.aux_mode == "eegdg_full" and args.model != "EEGDG":
        raise ValueError("aux_mode=eegdg_full is only valid with --model EEGDG.")

    if args.model == "EEGDG" and args.aux_mode == "eegdg_full":
        if args.protocol not in {"loso_te", "eegdg_paper_te"}:
            raise ValueError("aux_mode=eegdg_full supports --protocol loso_te or eegdg_paper_te.")
        train_eegdg_full_loso_te(
            data_dir=args.data_dir,
            results_root=args.results_root,
            seed=args.seed,
            epochs=args.eegdg_full_epochs,
            lr=args.eegdg_full_lr,
            weight_decay=args.eegdg_full_weight_decay,
            domain_batch_size=args.eegdg_domain_batch_size,
            patience=args.eegdg_full_patience,
            lambda_mmd=args.eegdg_mmd_weight,
            beta_domain=args.eegdg_domain_weight,
            gamma_mcd=args.eegdg_mcd_weight,
            mcd_alpha=args.eegdg_mcd_alpha,
            resume_existing=args.resume,
            protocol=args.protocol,
        )
    elif args.model == "all":
        run_all_baselines(
            data_dir=args.data_dir,
            results_root=args.results_root,
            protocol=args.protocol,
            seed=args.seed,
            aux_mode=args.aux_mode,
            lambda_aux=args.lambda_aux,
            eegdg_conditional_weight=args.eegdg_conditional_weight,
            supcon_temperature=args.supcon_temperature,
            label_smoothing=args.label_smoothing,
            use_class_weights=not args.disable_class_weights,
            use_weighted_sampler=not args.disable_weighted_sampler,
            same_class_mixup_alpha=args.same_class_mixup_alpha,
            same_class_mixup_prob=args.same_class_mixup_prob,
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
            supcon_temperature=args.supcon_temperature,
            label_smoothing=args.label_smoothing,
            use_class_weights=not args.disable_class_weights,
            use_weighted_sampler=not args.disable_weighted_sampler,
            same_class_mixup_alpha=args.same_class_mixup_alpha,
            same_class_mixup_prob=args.same_class_mixup_prob,
            max_time_shift=args.max_time_shift,
            noise_std=args.noise_std,
            grad_clip_norm=args.grad_clip_norm,
            aux_warmup_epochs=args.aux_warmup_epochs,
            resume_existing=args.resume,
        )
