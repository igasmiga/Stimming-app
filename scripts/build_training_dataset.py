"""Buduje tabelę cech do uczenia modelu z danych dwóch czujników x-IMU3.

Przykład:
python scripts/build_training_dataset.py --manifest data/manifest.csv \
  --annotations data/annotations.csv --output data/training_windows.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from imu_pipeline import fuse_sensor_features, read_ximu_csv, sampling_rate_hz


FEATURE_SIGNALS = [
    "wrist_acc_variability_g", "wrist_gyro_mean_dps",
    "lumbar_acc_variability_g", "lumbar_gyro_mean_dps", "motion_score",
]


def label_for_window(labels: pd.DataFrame, start: float, end: float, fully_annotated: bool) -> str:
    """Przypisuje klasę o największym pokryciu okna (co najmniej 50%)."""
    window_length = end - start
    candidates = labels.copy()
    candidates["overlap"] = (
        np.minimum(candidates["end_s"], end) - np.maximum(candidates["start_s"], start)
    ).clip(lower=0)
    best = candidates.loc[candidates["overlap"].idxmax()] if not candidates.empty else None
    if best is not None and best["overlap"] >= 0.5 * window_length:
        return str(best["label"])
    return "background" if fully_annotated else "unknown"


def window_features(frame: pd.DataFrame, session: pd.Series, labels: pd.DataFrame, window_s: float, hop_s: float) -> list[dict]:
    rate = sampling_rate_hz(frame)
    window_samples = round(window_s * rate)
    hop_samples = round(hop_s * rate)
    rows: list[dict] = []
    for start_index in range(0, len(frame) - window_samples + 1, hop_samples):
        chunk = frame.iloc[start_index : start_index + window_samples]
        start_s, end_s = float(chunk["time_s"].iloc[0]), float(chunk["time_s"].iloc[-1])
        row = {
            "participant_id": session["participant_id"],
            "session_id": session["session_id"],
            "window_start_s": round(start_s, 3),
            "window_end_s": round(end_s, 3),
            "label": label_for_window(labels, start_s, end_s, bool(session["fully_annotated"])),
        }
        for signal in FEATURE_SIGNALS:
            values = chunk[signal]
            row[f"{signal}_mean"] = values.mean()
            row[f"{signal}_std"] = values.std()
            row[f"{signal}_min"] = values.min()
            row[f"{signal}_max"] = values.max()
            row[f"{signal}_energy"] = float(np.mean(np.square(values)))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV opisujący pary plików dla sesji")
    parser.add_argument("--annotations", required=True, help="CSV z ręcznymi etykietami przedziałów")
    parser.add_argument("--output", required=True, help="Docelowy plik CSV z oknami treningowymi")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    annotations = pd.read_csv(args.annotations)
    required_manifest = {"participant_id", "session_id", "wrist_path", "lumbar_path", "fully_annotated"}
    required_annotations = {"session_id", "start_s", "end_s", "label"}
    if missing := required_manifest - set(manifest.columns):
        raise ValueError(f"Brak kolumn w manifeście: {', '.join(sorted(missing))}")
    if missing := required_annotations - set(annotations.columns):
        raise ValueError(f"Brak kolumn w etykietach: {', '.join(sorted(missing))}")

    rows: list[dict] = []
    for _, session in manifest.iterrows():
        wrist_path, lumbar_path = Path(session["wrist_path"]), Path(session["lumbar_path"])
        if not wrist_path.exists() or not lumbar_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pary plików sesji {session['session_id']}")
        wrist = read_ximu_csv(wrist_path.read_bytes())
        lumbar = read_ximu_csv(lumbar_path.read_bytes())
        # Opcjonalna kolumna pozwala zachować przesunięcie ustalone przy
        # etykietowaniu. Bez niej zakładamy, że czujniki zaczęły równocześnie.
        lumbar_offset = float(session.get("lumbar_time_offset_s", 0.0))
        fused = fuse_sensor_features(wrist, lumbar, lumbar_time_offset_s=lumbar_offset)
        session_labels = annotations[annotations["session_id"] == session["session_id"]]
        rows.extend(window_features(fused, session, session_labels, args.window_seconds, args.hop_seconds))

    dataset = pd.DataFrame(rows)
    dataset = dataset[dataset["label"] != "unknown"].reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Zapisano {len(dataset)} okien w: {args.output}")
    print(dataset["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
