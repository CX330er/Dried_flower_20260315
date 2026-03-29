import argparse
from pathlib import Path

import mne
import numpy as np

BCIC2A_EEG_CHANNELS = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]


def _load_npz(npz_path: Path):
    data = np.load(npz_path)
    x_key = "x" if "x" in data.files else "X"
    y_key = "y" if "y" in data.files else "Y"
    x = data[x_key].astype(np.float32)
    y = data[y_key].squeeze().astype(np.int64)
    return x, y


def _build_channel_names(n_channels: int):
    if n_channels == 22:
        return BCIC2A_EEG_CHANNELS
    return [f"EEG{i:02d}" for i in range(1, n_channels + 1)]


def visualize_trial(npz_path: Path, trial_idx: int, sfreq: int):
    x, y = _load_npz(npz_path)
    if x.ndim != 3:
        raise ValueError(f"Expected x shape [N, C, T], got {x.shape}")

    if trial_idx < 0 or trial_idx >= x.shape[0]:
        raise IndexError(f"trial_idx {trial_idx} out of range [0, {x.shape[0]-1}]")

    trial = x[trial_idx]
    ch_names = _build_channel_names(trial.shape[0])
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(trial, info, verbose="ERROR")

    title = f"{npz_path.name} | trial={trial_idx} | label={int(y[trial_idx]) if len(y) > trial_idx else 'NA'}"
    print(f"[viz] {title}")
    raw.plot(
        title=title,
        duration=5.0,
        n_channels=min(22, trial.shape[0]),
        scalings="auto",
        show=True,
        block=True,
    )


def visualize_epochs(npz_path: Path, sfreq: int, n_epochs: int = 10):
    x, y = _load_npz(npz_path)
    if x.ndim != 3:
        raise ValueError(f"Expected x shape [N, C, T], got {x.shape}")

    ch_names = _build_channel_names(x.shape[1])
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    events = np.column_stack([
        np.arange(len(y), dtype=np.int64),
        np.zeros(len(y), dtype=np.int64),
        y.astype(np.int64),
    ])

    event_id = {str(int(k)): int(k) for k in np.unique(y)}
    epochs = mne.EpochsArray(x, info, events=events, event_id=event_id, tmin=0.0, verbose="ERROR")
    print(f"[viz] {npz_path.name} | epochs={len(epochs)} | channels={x.shape[1]} | samples={x.shape[2]}")
    epochs.plot(
        n_epochs=min(n_epochs, len(epochs)),
        n_channels=min(22, x.shape[1]),
        scalings="auto",
        show=True,
        block=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize processed .npz EEG with MNE GUI")
    parser.add_argument("--npz", type=str, required=True, help="Path to one processed .npz file")
    parser.add_argument("--sfreq", type=int, default=250, help="Sampling rate for visualization")
    parser.add_argument("--mode", choices=["trial", "epochs"], default="trial")
    parser.add_argument("--trial_idx", type=int, default=0, help="Used in trial mode")
    parser.add_argument("--n_epochs", type=int, default=10, help="Used in epochs mode")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    if args.mode == "trial":
        visualize_trial(npz_path=npz_path, trial_idx=args.trial_idx, sfreq=args.sfreq)
    else:
        visualize_epochs(npz_path=npz_path, sfreq=args.sfreq, n_epochs=args.n_epochs)


if __name__ == "__main__":
    main()
