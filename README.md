# Dried_flower_20260315
My research on enhancing the generalization ability of EEG signal recognition models

This is the structure of my current project：
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
└─ notebooks/

--"environment.yml" file is used to record the Conda environment for easy replication on Github and Codex

--"main.py" serves as the unified entry point

--"configs" stores model configuration files, including data path, batch size, learning rate, epoch, dropout, etc

--"data" stores raw data, processed data, and partitioned data

--"datasets" contains the code for reading and processing data

--"models" is used to store the baseline and core model code

--"trainers" for placing training and evaluation logic code

--"utils" contains general tools including fixing random seeds, logging, result evaluation metrics, and model saving

"scripts" for quick running scripts

"results" for output results

"Notebooks" for exploratory analysis code
