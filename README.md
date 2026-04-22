# Dried_flower_20260315
My research on enhancing the generalization ability of EEG signal recognition models

## research stages
1. Data processing and baseline model establishment.
2. Structural innovation and improvement on existing models.
3. Advanced methods including domain adaptation or domain generalization.

## Raw dataset note
The `data/raw` directory stores the original EEG signals from **BCIC-IV 2a**:
- 9 subjects.
- Four-class motor imagery classification.
- 22 EEG channels and 3 EOG channels.
- Each subject includes training session **T** and evaluation session **E** such as 'A01T' and 'A01E'.
- Files are stored in `.gdf` format.

## Project structure：
```text
Dried_flower/
├─ .gitignore
├─ environment.yml
├─ main.py
├─ configs/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ splits/
├─ datasets/
├─ models/
├─ trainers/
├─ utils/
├─ scripts/
├─ results/
├─notebooks/
└─ main.py
```

## Stage 1.2 baseline scope
This repository now contains five baseline models for four-class motor imagery EEG classification:
- `ShallowConvNet`
- `DeepConvNet`
- `EEGNet`
- `FBCNet`
- `MSFBCNN`

Shared training setup:
-LOSO(Leave-One-Subject-Out)across`.npz`subjects.
-validation split in training subjects with fixed seed.
-Epochs:300
-Optimizer:Adam
-Learning rate:1e-3
-Batch size:64
-Loss:CrossEntropy
-Early stopping:patience 50

## Data format assumptions
Processed per-subject files are expected at:
- `data/processed/bcic_iv_2a/*.npz`

Each`.npz`contains:
-`x`or`X`:EEG trials with shape `[N, 1, 22, T]` after preprocessing (training loader also supports legacy `[N, 22, T]`)
-`y`or`Y`:labels in `{0, 1, 2, 3}`

Preprocessing (current):
- retain 22 EEG channels
- map motor imagery events 769/770/771/772 -> labels 0/1/2/3
- drop bad trials by reject markers
- epoch window: 2s to 6s
- Butterworth band-pass filtering: 4-40Hz
- z-score normalization

## Run training
Train all baselines:
```bash
python main.py --model all --data_dir data/processed/bcic_iv_2a --results_root results
```

Train one model:
```bash
python main.py --model EEGNet
```

## Output files
For each model/fold
-`results/<model_name_lower>/fold_x/metrics.json`
-`results/<model_name_lower>/fold_x/confusion_matrix.png`
-`results/<model_name_lower>/fold_x/training_curve.png`

Per-model summary:
-`results/<model_name_lower>/summary.csv`

Cross-model comparison:
-`results/baselinecompare.csv`
