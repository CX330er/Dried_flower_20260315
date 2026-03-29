import argparse

from trainers.baseline_trainer import run_all_baselines, train_and_evaluate_model

DEFAULT_DATA_DIR = "data/processed/bcic_iv_2a"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG baseline training entrypoint")
    parser.add_argument("--model", type=str, default="all", choices=["all", "ShallowConvNet", "DeepConvNet", "EEGNet", "FBCNet", "MSFBCNN"])
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results_root", type=str, default="results")
    args = parser.parse_args()

    if args.model == "all":
        run_all_baselines(data_dir=args.data_dir, results_root=args.results_root)
    else:
        train_and_evaluate_model(model_name=args.model, data_dir=args.data_dir, results_root=args.results_root)
