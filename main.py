import argparse

from trainers.baseline_trainer import run_all_baselines, train_and_evaluate_model

DEFAULT_DATA_DIR = "data/processed/bcic_iv_2a"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG baseline training entrypoint")
    parser.add_argument("--model", type=str, default="all", choices=["all", "ShallowConvNet", "DeepConvNet", "EEGNet", "FBCNet", "MSFBCNN", "EEGNetFSFE"])
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--aux_mode", type=str, default="none", choices=["none", "center", "coral"])
    parser.add_argument("--lambda_aux", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--disable_class_weights", action="store_true")
    parser.add_argument("--disable_weighted_sampler", action="store_true")
    parser.add_argument("--max_time_shift", type=int, default=25)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--aux_warmup_epochs", type=int, default=30)
    parser.add_argument("--aux_mode", type=str, default="none", choices=["none", "center", "coral"])
    parser.add_argument("--lambda_aux", type=float, default=0.05)
    args = parser.parse_args()

    if args.model == "all":
        run_all_baselines(data_dir=args.data_dir, results_root=args.results_root)
    else:
        train_and_evaluate_model(
            model_name=args.model,
            data_dir=args.data_dir,
            results_root=args.results_root,
            aux_mode=args.aux_mode,
            lambda_aux=args.lambda_aux,
            label_smoothing=args.label_smoothing,
            use_class_weights=not args.disable_class_weights,
            use_weighted_sampler=not args.disable_weighted_sampler,
            max_time_shift=args.max_time_shift,
            noise_std=args.noise_std,
            grad_clip_norm=args.grad_clip_norm,
            aux_warmup_epochs=args.aux_warmup_epochs,
        )
