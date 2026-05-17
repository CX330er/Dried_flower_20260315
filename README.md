# Dried Flower EEG Project

Research code for BCIC-IV-2a four-class motor imagery EEG recognition, with a focus on subject-dependent baselines and subject-independent cross-subject generalization.

## Current Data Setting

- Dataset: BCIC-IV-2a.
- Channels: 22 EEG channels.
- Classes: 4 motor imagery classes, mapped to labels `0, 1, 2, 3`.
- Current preprocessing band: `4-40Hz`.
- Current selected time window: `0.5-4.5s`.
- Processed trial shape: `[N, 1, 22, T]`; loaders also accept legacy `[N, 22, T]`.

Training sessions `A01T-A09T` use event labels `769/770/771/772`. Evaluation sessions `A01E-A09E` use cue event `783` plus official labels from `data/raw/true_labels/AxxE.mat`.

## Naming Convention

Processed data directories should include session scope, window, and band:

```text
data/processed/bcic_iv_2a_train_only_window_0p5_4p5_band_4_40hz
data/processed/bcic_iv_2a_train_eval_window_0p5_4p5_band_4_40hz
```

Result directories should use the change-log date/time as the run id, plus a short model/protocol suffix. Do not include preprocessing tags such as `window_0p5_4p5_band_4_40hz` in new result directory names; record those details in `PROJECT_CHANGE_LOG.md` and the generated `run_config.json` instead.

```text
results/2026-05-13_1130_all_subject_dependent_te
results/2026-05-13_1145_all_loso_te
results/2026-05-13_1200_eegnet_stepwise_debug
```

## Protocols

`loso_t`: legacy/debug protocol. Uses processed `AxxT` sessions only and performs LOSO across training sessions.

`subject_dependent_te`: official-style subject-dependent protocol. For each subject, train on `AxxT`, validate on a split of `AxxT`, and test on `AxxE`.

`subject_dependent_te_final`: final subject-dependent protocol. For each subject, first use the `AxxT` train/validation split to select the best epoch, then refit from scratch on the full `AxxT=288` trials and test on `AxxE=288`.

`loso_te`: strict subject-independent protocol. For each target subject, train on the other subjects' `AxxT`, validate on a split of the source `T` trials, and test on the target subject's `AxxE`.

## Preprocessing

Generate train-only data for window sweep or legacy LOSO debug:

```powershell
D:\anaconda3\envs\weed_out\python.exe scripts\process_bcic_iv_2a.py --raw-dir data/raw --session-mode train --l-freq 4 --h-freq 40 --tmin 0.5 --tmax 4.5 --replace-out-dir
```

Generate matched T/E data for protocol 1 and protocol 2:

```powershell
D:\anaconda3\envs\weed_out\python.exe scripts\process_bcic_iv_2a.py --raw-dir data/raw --label-dir data/raw/true_labels --session-mode all --l-freq 4 --h-freq 40 --tmin 0.5 --tmax 4.5 --replace-out-dir
```

When `--out-dir` is omitted, the preprocessing script now creates a descriptive directory name automatically.

## Training

Run protocol 1 with all baseline models:

```powershell
D:\anaconda3\envs\weed_out\python.exe main.py --model all --data_dir data/processed/bcic_iv_2a_train_eval_window_0p5_4p5_band_4_40hz --protocol subject_dependent_te
```

Run protocol 1 final baseline with all baseline models:

```powershell
D:\anaconda3\envs\weed_out\python.exe main.py --model all --data_dir data/processed/bcic_iv_2a_train_eval_window_0p5_4p5_band_4_40hz --protocol subject_dependent_te_final
```

Run protocol 2 with all baseline models plus EEGNetFSFE:

```powershell
D:\anaconda3\envs\weed_out\python.exe main.py --model all --data_dir data/processed/bcic_iv_2a_train_eval_window_0p5_4p5_band_4_40hz --protocol loso_te --seed 42
```

Resume an interrupted run by reusing the same timestamp and adding `--resume`:

```powershell
D:\anaconda3\envs\weed_out\python.exe main.py --model all --data_dir data/processed/bcic_iv_2a_train_eval_window_0p5_4p5_band_4_40hz --protocol loso_te --seed 42 --run_timestamp 2026-05-14_2000 --aux_mode none --resume
```

Run one model:

```powershell
D:\anaconda3\envs\weed_out\python.exe main.py --model EEGNet --data_dir data/processed/bcic_iv_2a_train_eval_window_0p5_4p5_band_4_40hz --protocol loso_te
```

Run EEGNetFSFE with EEG-DG-style source-domain alignment:

```powershell
D:\anaconda3\envs\weed_out\python.exe main.py --model EEGNetFSFE --data_dir data/processed/bcic_iv_2a_train_eval_window_0p5_4p5_band_4_40hz --protocol loso_te --seed 42 --run_timestamp 2026-05-16_2000 --aux_mode eegdg --lambda_aux 0.05 --eegdg_conditional_weight 1.0
```

`aux_mode=eegdg` uses only source subjects in each LOSO fold. It adds
feature-level marginal MMD across source subjects and class-conditional MMD
within each available class.

When `--results_root` is omitted, `main.py` creates a date-based directory automatically. To align a run with a change-log entry, pass the same date tag explicitly:

```powershell
D:\anaconda3\envs\weed_out\python.exe main.py --model all --data_dir data/processed/bcic_iv_2a_train_eval_window_0p5_4p5_band_4_40hz --protocol loso_te --run_timestamp 2026-05-14_1900
```

## Stepwise Debug

Run EEGNet window sweep diagnostics with date-based output names:

```powershell
D:\anaconda3\envs\weed_out\python.exe scripts\run_stepwise_debug.py --model EEGNet --run_loso --run_window_sweep --execute
```

Default output:

```text
results/<date>_<model>_stepwise_debug
```

## Output Files

For each model/fold:

```text
results/<experiment>/<model_name_lower>/fold_<idx>/metrics.json
results/<experiment>/<model_name_lower>/fold_<idx>/predictions.csv
results/<experiment>/<model_name_lower>/fold_<idx>/confusion_matrix.csv
results/<experiment>/<model_name_lower>/fold_<idx>/confusion_matrix.png
results/<experiment>/<model_name_lower>/fold_<idx>/history.csv
results/<experiment>/<model_name_lower>/fold_<idx>/training_curve.png
results/<experiment>/<model_name_lower>/fold_<idx>/selection_history.csv  # subject_dependent_te_final only
results/<experiment>/<model_name_lower>/fold_<idx>/selection_curve.png  # subject_dependent_te_final only
```

Per-model summary:

```text
results/<experiment>/<model_name_lower>/summary.csv
```

Cross-model comparison:

```text
results/<experiment>/baseline_compare.csv
results/<experiment>/all_fold_metrics.csv
```

Top-level run metadata:

```text
results/<experiment>/run_config.json
```

## Change Log Workflow

Project-level changes should be recorded in `PROJECT_CHANGE_LOG.md`. Each entry should include the timestamp, purpose/目的, changed items/改动, added items/增加, removed items/减少, and optional notes/备注.

Append a new entry with:

```powershell
D:\anaconda3\envs\weed_out\python.exe scripts\append_change_log.py --title "Short change title" --purpose "Why this change was made" --changed "What changed" --added "What was added" --removed "What was removed"
```

Repeat `--changed`, `--added`, `--removed`, or `--notes` when one entry needs multiple bullet points.
