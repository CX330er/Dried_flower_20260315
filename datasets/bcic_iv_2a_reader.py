"""BCIC-IV-2a raw data reader and preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from scipy.signal import butter, filtfilt

import mne
import numpy as np

# Official four-class motor imagery cue events in BCIC-IV-2a.
EVENT_ID_MAP: Dict[str, int] = {
    "769": 0,  # left hand
    "770": 1,  # right hand
    "771": 2,  # feet
    "772": 3,  # tongue
}

# Canonical 22 EEG channels in BCIC-IV-2a.
BCIC_IV_2A_EEG_CHANNELS: List[str] = [
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

# Annotation descriptions that usually represent rejected/artefact trials.
DEFAULT_REJECT_MARKERS = {
    "1023",  # rejected trial marker in BCIC-IV-2a
    "BAD",
    "BAD_ACQ_SKIP",
    "bad",
}


class NoCueEventsError(RuntimeError):
    """Raised when a file has no labeled MI cue events (769-772)."""


@dataclass
class ProcessConfig:
    """Configuration for preprocessing BCIC-IV-2a data."""

    l_freq: float = 4.0
    h_freq: float = 40.0
    resample_sfreq: int = 250
    tmin: float = 2.0
    tmax: float = 6.0
    baseline: Tuple[float, float] | None = None
    butter_order: int = 4

def _normalize_annotation_desc(desc: str) -> str:
    """Extract digit-based code from annotation text when possible."""
    desc = str(desc).strip()
    digits = "".join(ch for ch in desc if ch.isdigit())
    return digits if digits else desc


def _gdf_files(raw_dir: Path) -> List[Path]:
    files = sorted(raw_dir.glob("*.gdf"))
    if not files:
        raise FileNotFoundError(f"No .gdf files found in: {raw_dir}")
    return files


def _pick_22_eeg_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Pick the 22 EEG channels from BCIC-IV-2a and drop EOG channels."""
    canonical_present = [ch for ch in BCIC_IV_2A_EEG_CHANNELS if ch in raw.ch_names]
    if len(canonical_present) == 22:
        raw.pick(canonical_present)
        return raw

    non_eog_channels = [ch for ch in raw.ch_names if "EOG" not in ch.upper()]
    if len(non_eog_channels) == 22:
        raw.pick(non_eog_channels)
        return raw

    raise RuntimeError(
        "Unable to select exactly 22 EEG channels from file. "
        f"Found channels={len(raw.ch_names)}, non-EOG channels={len(non_eog_channels)}."
    )


def read_raw_gdf(file_path: Path) -> mne.io.BaseRaw:
    """Read one BCIC-IV-2a .gdf file and keep only 22 EEG channels."""
    raw = mne.io.read_raw_gdf(file_path, preload=True, verbose="ERROR")
    raw = _pick_22_eeg_channels(raw)
    return raw


def _extract_bad_event_times(raw: mne.io.BaseRaw, reject_markers: Iterable[str]) -> np.ndarray:
    reject_set = {m for m in reject_markers}
    bad_times = []
    for ann in raw.annotations:
        code = _normalize_annotation_desc(ann["description"])
        if code in reject_set:
            bad_times.append(float(ann["onset"]))
    return np.asarray(bad_times, dtype=float)


def _extract_cue_events(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, np.ndarray]:
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    code_by_event_value = {value: _normalize_annotation_desc(key) for key, value in event_id.items()}

    cue_rows: List[np.ndarray] = []
    cue_labels: List[int] = []

    for ev in events:
        event_value = int(ev[2])
        code = code_by_event_value.get(event_value, "")
        if code in EVENT_ID_MAP:
            cue_rows.append(ev)
            cue_labels.append(EVENT_ID_MAP[code])

    if not cue_rows:
        available_codes = sorted(set(code_by_event_value.values()))
        raise NoCueEventsError(
            "No labeled motor imagery cue events (769-772) found in file. "
            f"Available annotation codes include: {available_codes[:20]}"
        )

    return np.asarray(cue_rows, dtype=int), np.asarray(cue_labels, dtype=int)


def _drop_bad_trials(
    cue_events: np.ndarray,
    cue_labels: np.ndarray,
    sfreq: float,
    bad_times: np.ndarray,
    trial_window_sec: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop trials if a bad marker occurs during [cue, cue + trial_window_sec]."""
    if bad_times.size == 0:
        return cue_events, cue_labels

    kept_events = []
    kept_labels = []
    for ev, label in zip(cue_events, cue_labels):
        cue_time = float(ev[0]) / sfreq
        in_trial = np.logical_and(bad_times >= cue_time, bad_times <= cue_time + trial_window_sec)
        if not np.any(in_trial):
            kept_events.append(ev)
            kept_labels.append(label)

    if not kept_events:
        raise RuntimeError("All trials were removed after artifact/reject filtering.")

    return np.asarray(kept_events, dtype=int), np.asarray(kept_labels, dtype=int)

def _butter_bandpass_filter(
    x: np.ndarray,
    l_freq: float,
    h_freq: float,
    sfreq: float,
    order: int,
) -> np.ndarray:
    """Apply zero-phase Butterworth band-pass filtering on the time axis.

    Args:
        x: EEG array with shape [B, C, T].
    """
    nyquist = 0.5 * sfreq
    low = l_freq / nyquist
    high = h_freq / nyquist
    if not (0 < low < high < 1):
        raise ValueError(
            f"Invalid band-pass range for Butterworth: l_freq={l_freq}, h_freq={h_freq}, sfreq={sfreq}."
        )
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x, axis=-1).astype(np.float32, copy=False)


def _zscore_per_trial_channel(x: np.ndarray) -> np.ndarray:
    """Z-score normalize each (trial, channel) sequence along time axis."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32, copy=False)

def preprocess_one_file(
    file_path: Path,
    config: ProcessConfig,
    reject_markers: Iterable[str] = DEFAULT_REJECT_MARKERS,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int | str]]:
    """Load and preprocess one file into (X, y, metadata).

    Returns:
        X: (n_trials, 22, n_times)
        y: (n_trials,)
        metadata: summary fields for logging/reporting
    """
    raw = read_raw_gdf(file_path)

    if len(raw.ch_names) != 22:
        raise RuntimeError(
            f"Expected 22 EEG channels after selection, got {len(raw.ch_names)} in {file_path.name}."
        )

    raw.resample(config.resample_sfreq, npad="auto", verbose="ERROR")

    cue_events, cue_labels = _extract_cue_events(raw)
    bad_times = _extract_bad_event_times(raw, reject_markers=reject_markers)
    cue_events, cue_labels = _drop_bad_trials(
        cue_events,
        cue_labels,
        sfreq=raw.info["sfreq"],
        bad_times=bad_times,
        trial_window_sec=config.tmax,
    )

    epoch_events = cue_events.copy()
    epoch_events[:, 2] = cue_labels + 1

    epochs = mne.Epochs(
        raw,
        events=epoch_events,
        event_id=None,
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=config.baseline,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    x = epochs.get_data(copy=True).astype(np.float32)  # [B, C, T]
    x = _butter_bandpass_filter(
        x,
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        sfreq=float(config.resample_sfreq),
        order=config.butter_order,
    )
    x = _zscore_per_trial_channel(x)
    x = np.expand_dims(x, axis=1)  # [B, 1, C, T]
    y = (epochs.events[:, 2] - 1).astype(np.int64)

    meta = {
        "file": file_path.name,
        "n_trials": int(x.shape[0]),
        "n_channels": int(x.shape[2]),
        "n_times": int(x.shape[3]),
        "sfreq": int(config.resample_sfreq),
        "shape": [int(v) for v in x.shape],
    }
    return x, y, meta


def collect_subject_session_id(file_path: Path) -> str:
    """Infer ID like A01T/A01E from file stem."""
    return file_path.stem.upper()


def iter_raw_files(raw_dir: Path) -> List[Path]:
    """Public file list helper."""
    return _gdf_files(raw_dir)
