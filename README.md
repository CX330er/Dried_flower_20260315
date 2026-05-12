# Dried Flower EEG Project

Research code for BCIC-IV-2a four-class motor imagery EEG recognition, with a focus on subject-dependent baselines and subject-independent cross-subject generalization.

## Current Data Setting

- Dataset: BCIC-IV-2a.
- Channels: 22 EEG channels.
- Classes: 4 motor imagery classes, mapped to labels `0, 1, 2, 3`.
- Current preprocessing band: `5-30Hz`.
- Current selected time window: `0.5-4.5s`.
- Processed trial shape: `[N, 1, 22, T]`; loaders also accept legacy `[N, 22, T]`.

Training sessions `A01T-A09T` use event labels `769/770/771/772`. Evaluation sessions `A01E-A09E` use cue event `783` plus official labels from `data/raw/true_labels/AxxE.mat`.

## Naming Convention

Processed data directories should include session scope, window, and band:

```text
data/processed/bcic_iv_2a_train_only_window_0p5_4p5_band_5_30hz
data/processed/bcic_iv_2a_train_eval_window_0p5_4p5_band_5_30hz
```

Result directories should include model, task/protocol, and data setting:

```text
results/eegnet_stepwise_debug_train_only_window_sweep_band_5_30hz
results/protocol1_subject_dependent_te_train_eval_window_0p5_4p5_band_5_30hz
results/protocol2_loso_te_train_eval_window_0p5_4p5_band_5_30hz
```

## Protocols

`loso_t`: legacy/debug protocol. Uses processed `AxxT` sessions only and performs LOSO across training sessions.

`subject_dependent_te`: official-style subject-dependent protocol. For each subject, train on `AxxT`, validate on a split of `AxxT`, and test on `AxxE`.

`loso_te`: strict subject-independent protocol. For each target subject, train on the other subjects' `AxxT`, validate on a split of the source `T` trials, and test on the target subject's `AxxE`.

## Preprocessing

Generate train-only data for window sweep or legacy LOSO debug:

```powershell
D:\anaconda3\envs\weed_out\python.exe scripts\process_bcic_iv_2a.py --raw-dir data/raw --session-mode train --l-freq 5 --h-freq 30 --tmin 0.5 --tmax 4.5 --replace-out-dir
```

Generate matched T/E data for protocol 1 and protocol 2:

```powershell
D:\anaconda3\envs\weed_out\python.exe scripts\process_bcic_iv_2a.py --raw-dir data/raw --label-dir data/raw/true_labels --session-mode all --l-freq 5 --h-freq 30 --tmin 0.5 --tmax 4.5 --replace-out-dir
```

When `--out-dir` is omitted, the preprocessing script now creates a descriptive directory name automatically.

## Training

Run protocol 1 with all baseline models:

```powershell
D:\anaconda3\envs\weed_out\python.exe main.py --model all --data_dir data/processed/bcic_iv_2a_train_eval_window_0p5_4p5_band_5_30hz --results_root results/protocol1_subject_dependent_te_train_eval_window_0p5_4p5_band_5_30hz --protocol subject_dependent_te
```

Run protocol 2 with all baseline models:

```powershell
D:\anaconda3\envs\weed_out\python.exe main.py --model all --data_dir data/processed/bcic_iv_2a_train_eval_window_0p5_4p5_band_5_30hz --results_root results/protocol2_loso_te_train_eval_window_0p5_4p5_band_5_30hz --protocol loso_te
```

Run one model:

```powershell
D:\anaconda3\envs\weed_out\python.exe main.py --model EEGNet --data_dir data/processed/bcic_iv_2a_train_eval_window_0p5_4p5_band_5_30hz --results_root results/eegnet_protocol2_loso_te_train_eval_window_0p5_4p5_band_5_30hz --protocol loso_te
```

## Stepwise Debug

Run EEGNet window sweep diagnostics with descriptive output names:

```powershell
D:\anaconda3\envs\weed_out\python.exe scripts\run_stepwise_debug.py --model EEGNet --run_loso --run_window_sweep --execute
```

Default output:

```text
results/eegnet_stepwise_debug_train_only_window_sweep_band_5_30hz
```

## Output Files

For each model/fold:

```text
results/<experiment>/<model_name_lower>/fold_<idx>/metrics.json
results/<experiment>/<model_name_lower>/fold_<idx>/confusion_matrix.png
results/<experiment>/<model_name_lower>/fold_<idx>/training_curve.png
```

Per-model summary:

```text
results/<experiment>/<model_name_lower>/summary.csv
```

Cross-model comparison:

```text
results/<experiment>/baseline_compare.csv
```
