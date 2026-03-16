# Dried_flower_20260315
My research on enhancing the generalization ability of EEG signal recognition models

## Current research stages
1. Data processing and baseline model establishment.
2. Structural innovation and improvement on existing models.
3. Advanced methods including domain adaptation and meta-learning.

## Raw dataset note
The `data/raw` directory stores the original EEG signals from **BCIC-IV 2a**:
- 9 subjects.
- Four-class motor imagery classification.
- 22 EEG channels and 3 EOG channels.
- Each subject includes training session **T** and evaluation session **E** such as 'A01T' and 'A01E'.
- Files are stored in `.gdf` format.

## This is the structure of my current project：
```text
Dried_flower/
├─ .gitignore
├─ environment.yml
├─ main.py
├─ configs/
├─ data/
│  ├─ true_label/
│  ├─ raw/
│  ├─ processed/
│  └─ splits/
├─ datasets/
├─ models/
├─ trainers/
├─ utils/
├─ scripts/
├─ results/
└─ notebooks/
```

`environment.yml` file is used to record the Conda environment for easy replication on Github and Codex

`main.py` serves as the unified entry point

`configs` stores model configuration files, including data path, batch size, learning rate, epoch, dropout, etc

`data` stores evaluate data's true label, raw data, processed data, and partitioned data

`datasets` contains the code for reading and processing data

`models` is used to store the baseline and core model code

`trainers` for placing training and evaluation logic code

`utils` contains general tools including fixing random seeds, logging, result evaluation metrics, and model saving

`scripts` for quick running scripts

`results` for output results

`Notebooks` for exploratory analysis code
