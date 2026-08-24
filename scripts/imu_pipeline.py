"""Przetwarzanie danych inercyjnych eksportowanych przez x-IMU3.

Ten moduł nie stawia diagnozy i nie klasyfikuje automatycznie ruchu jako
objawu klinicznego. W wersji bazowej wykrywa kandydatów na dynamiczne,
powtarzalne epizody ruchu. Wynik ma służyć do ręcznej weryfikacji i do
budowania późniejszego, uczonego modelu klasyfikacyjnego.
"""

from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import numpy as np
import pandas as pd


TIMESTAMP = "Timestamp (us)"
ACC_COLUMNS = [
    "Accelerometer X (g)",
    "Accelerometer Y (g)",
    "Accelerometer Z (g)",
]
GYRO_COLUMNS = ["Gyroscope X (deg/s)", "Gyroscope Y (deg/s)", "Gyroscope Z (deg/s)"]
REQUIRED_COLUMNS = [TIMESTAMP, *ACC_COLUMNS, *GYRO_COLUMNS]


def read_ximu_csv(source: BinaryIO | bytes) -> pd.DataFrame:
    """Wczytuje CSV x-IMU3 i zwraca dane w jednej, sprawdzonej postaci."""
    if isinstance(source, bytes):
        source = BytesIO(source)
    frame = pd.read_csv(source)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "To nie wygląda na plik Inertial.csv z x-IMU3. "
            f"Brakuje kolumn: {', '.join(missing)}"
        )

    frame = frame[REQUIRED_COLUMNS].copy()
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame.sort_values(TIMESTAMP).drop_duplicates(TIMESTAMP).reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("Plik zawiera zbyt mało poprawnych próbek do analizy.")

    frame["time_s"] = (frame[TIMESTAMP] - frame[TIMESTAMP].iloc[0]) / 1_000_000
    return frame


def sampling_rate_hz(frame: pd.DataFrame) -> float:
    deltas = frame["time_s"].diff().dropna()
    median_delta = float(deltas.median())
    if median_delta <= 0:
        raise ValueError("Nieprawidłowe znaczniki czasu w pliku.")
    return 1 / median_delta


def add_motion_features(frame: pd.DataFrame, window_seconds: float = 1.0) -> pd.DataFrame:
    """Dodaje cechy ruchu niezależne od orientacji czujnika."""
    result = frame.copy()
    rate = sampling_rate_hz(result)
    window = max(3, round(window_seconds * rate))

    acc = result[ACC_COLUMNS].to_numpy(dtype=float)
    gyro = result[GYRO_COLUMNS].to_numpy(dtype=float)
    result["acc_magnitude_g"] = np.linalg.norm(acc, axis=1)
    result["gyro_magnitude_dps"] = np.linalg.norm(gyro, axis=1)
    # Odchylenie w oknie usuwa wpływ grawitacji i orientacji czujnika.
    result["acc_variability_g"] = (
        result["acc_magnitude_g"].rolling(window, center=True, min_periods=1).std().fillna(0)
    )
    result["gyro_mean_dps"] = (
        result["gyro_magnitude_dps"].rolling(window, center=True, min_periods=1).mean()
    )
    return result


def fuse_sensor_features(
    wrist: pd.DataFrame, lumbar: pd.DataFrame, lumbar_time_offset_s: float = 0.0
) -> pd.DataFrame:
    """Synchronizuje sygnały z nadgarstka i odcinka lędźwiowego.

    ``lumbar_time_offset_s`` przesuwa czas czujnika lędźwiowego do czasu
    nadgarstka. Przykład: jeżeli skok synchronizacyjny widać w 3.0 s na
    nadgarstku i w 1.5 s na lędźwiach, offset wynosi 3.0 - 1.5 = 1.5 s.
    Następnie dane są interpolowane na wspólnej osi czasu; nie łączymy
    wierszy po numerze próbki.
    """
    wrist_featured = add_motion_features(wrist).set_index("time_s")
    lumbar_featured = add_motion_features(lumbar).copy()
    lumbar_featured["time_s"] += lumbar_time_offset_s
    lumbar_featured = lumbar_featured.set_index("time_s")
    shared_start = max(float(wrist_featured.index.min()), float(lumbar_featured.index.min()))
    shared_end = min(float(wrist_featured.index.max()), float(lumbar_featured.index.max()))
    if shared_end <= shared_start:
        raise ValueError("Po synchronizacji pliki nie mają wspólnego zakresu czasu.")
    rate = min(sampling_rate_hz(wrist), sampling_rate_hz(lumbar))
    step = 1 / rate
    common_time = np.arange(shared_start, shared_end, step)

    selected = ["acc_variability_g", "gyro_mean_dps"]
    wrist_part = wrist_featured[selected].reindex(wrist_featured.index.union(common_time)).interpolate().reindex(common_time)
    lumbar_part = lumbar_featured[selected].reindex(lumbar_featured.index.union(common_time)).interpolate().reindex(common_time)
    # W aplikacji i w etykietach 0 s oznacza pierwszy moment, w którym oba
    # czujniki mają jednocześnie dane.
    fused = pd.DataFrame({"time_s": common_time - shared_start})
    for column in selected:
        fused[f"wrist_{column}"] = wrist_part[column].to_numpy()
        fused[f"lumbar_{column}"] = lumbar_part[column].to_numpy()

    # Wspólny wynik wykorzystuje dynamikę obu części ciała. Model ML w
    # kolejnym etapie będzie uczył się z pełnego zestawu tych cech.
    fused["motion_score"] = sum(
        _robust_zscore(fused[column]).clip(lower=0)
        for column in [
            "wrist_acc_variability_g", "wrist_gyro_mean_dps",
            "lumbar_acc_variability_g", "lumbar_gyro_mean_dps",
        ]
    )
    fused.attrs["shared_start_s"] = shared_start
    fused.attrs["lumbar_time_offset_s"] = lumbar_time_offset_s
    return fused


def _robust_zscore(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median()
    if mad < 1e-9:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return 0.6745 * (values - median) / mad


def detect_motion_episodes(
    featured_frame: pd.DataFrame,
    threshold: float = 3.5,
    min_duration_seconds: float = 1.0,
    merge_gap_seconds: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wykrywa zwarte okresy podwyższonej dynamiki ruchu.

    Próg jest wyrażony w odpornych odchyleniach od mediany sesji. To detektor
    bazowy do przeglądu danych; nie jest modelem stymowania.
    """
    frame = featured_frame.copy()
    if "motion_score" in frame.columns:
        score = frame["motion_score"]
    else:
        score = _robust_zscore(frame["acc_variability_g"]).clip(lower=0) + _robust_zscore(
            frame["gyro_mean_dps"]
        ).clip(lower=0)
        frame["motion_score"] = score
    active = score >= threshold

    rate = sampling_rate_hz(frame)
    gap_samples = max(1, round(merge_gap_seconds * rate))
    # Krótkie luki wewnątrz jednego ruchu nie tworzą osobnych epizodów.
    inactive_runs = (~active).astype(int).groupby(active.ne(active.shift()).cumsum()).transform("sum")
    active = active | ((~active) & (inactive_runs <= gap_samples))
    frame["candidate_motion"] = active

    groups = active.ne(active.shift()).cumsum()
    episodes: list[dict[str, float | int | str]] = []
    for _, segment in frame[active].groupby(groups[active]):
        duration = float(segment["time_s"].iloc[-1] - segment["time_s"].iloc[0])
        if duration >= min_duration_seconds:
            episodes.append(
                {
                    "episode_id": len(episodes) + 1,
                    "start_s": round(float(segment["time_s"].iloc[0]), 2),
                    "end_s": round(float(segment["time_s"].iloc[-1]), 2),
                    "duration_s": round(duration, 2),
                    "peak_motion_score": round(float(segment["motion_score"].max()), 2),
                    "classification": "do weryfikacji",
                }
            )
    return frame, pd.DataFrame(episodes)


def session_summary(frame: pd.DataFrame, episodes: pd.DataFrame) -> dict[str, float | int]:
    duration = float(frame["time_s"].iloc[-1])
    candidate_duration = float(episodes["duration_s"].sum()) if not episodes.empty else 0.0
    return {
        "samples": len(frame),
        "sampling_rate_hz": round(sampling_rate_hz(frame), 2),
        "duration_s": round(duration, 2),
        "episode_count": len(episodes),
        "candidate_duration_s": round(candidate_duration, 2),
        "candidate_share_percent": round(100 * candidate_duration / duration, 2) if duration else 0.0,
    }
