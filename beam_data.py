"""Data helpers for the GPS-based beam prediction workshop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CALIBRATED_SCENARIOS = {3, 4, 8, 9}


def _find_one(data_root: str | Path, pattern: str) -> Path:
    """Return exactly one file matching a scenario pattern."""
    data_root = Path(data_root)
    matches = sorted(data_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file found for pattern {pattern!r} in {data_root}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Expected one file for pattern {pattern!r}, found: {names}")
    return matches[0]


def scenario_files(
    data_root: str | Path,
    scenario_id: int,
    use_calibrated: bool = True,
) -> dict[str, Path]:
    """Locate the GPS, power, and sequence files for one DeepSense scenario."""
    loc_name = "loc_cal" if use_calibrated and scenario_id in CALIBRATED_SCENARIOS else "loc"
    return {
        "user_gps": _find_one(data_root, f"scenario{scenario_id}_unit2_{loc_name}_1-*.npy"),
        "base_gps": _find_one(data_root, f"scenario{scenario_id}_unit1_loc_1-*.npy"),
        "power": _find_one(data_root, f"scenario{scenario_id}_unit1_pwr_60ghz_1-*.npy"),
        "seq_index": _find_one(data_root, f"scenario{scenario_id}_seq_index_1-*.npy"),
    }


def load_scenario(
    data_root: str | Path,
    scenario_id: int,
    use_calibrated: bool = True,
) -> dict[str, np.ndarray]:
    """Load one scenario and return arrays plus the paths used."""
    files = scenario_files(data_root, scenario_id, use_calibrated=use_calibrated)
    user_gps = np.load(files["user_gps"])
    base_gps = np.load(files["base_gps"])
    power = np.load(files["power"])
    seq_index = np.load(files["seq_index"])

    if user_gps.shape[0] != power.shape[0] or user_gps.shape[0] != seq_index.shape[0]:
        raise ValueError(
            "Scenario arrays have inconsistent sample counts: "
            f"user_gps={user_gps.shape}, power={power.shape}, seq_index={seq_index.shape}"
        )

    return {
        "scenario_id": scenario_id,
        "files": files,
        "user_gps": user_gps[:, :2],
        "base_gps": base_gps[:, :2],
        "power": power,
        "seq_index": seq_index,
    }


def build_beam_dataframe(
    user_gps: np.ndarray,
    power: np.ndarray,
    seq_index: np.ndarray,
    scenario_id: int | None = None,
) -> pd.DataFrame:
    """Build a tidy table where the label is the beam with maximum power."""
    best_beam = np.argmax(power, axis=1).astype(int)
    best_power = np.max(power, axis=1)
    df = pd.DataFrame(
        {
            "sample_id": np.arange(len(best_beam), dtype=int),
            "seq_index": seq_index.astype(int),
            "latitude": user_gps[:, 0],
            "longitude": user_gps[:, 1],
            "beam_index": best_beam,
            "best_power": best_power,
        }
    )
    if scenario_id is not None:
        df.insert(0, "scenario_id", int(scenario_id))
    return df


def train_test_split_beam_data(
    df: pd.DataFrame,
    power: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, np.ndarray | pd.DataFrame]:
    """Split GPS features, labels, powers, and dataframe rows together.

    A fully stratified split fails when a class appears only once. In that
    beginner-unfriendly corner case, rare samples stay in train and the rest is
    stratified normally.
    """
    features = df[["latitude", "longitude"]].to_numpy()
    labels = df["beam_index"].to_numpy(dtype=int)
    indices = np.arange(len(df))
    counts = pd.Series(labels).value_counts()
    rare_classes = set(counts[counts < 2].index)
    rare_mask = np.array([label in rare_classes for label in labels])

    if rare_mask.any():
        common_idx = indices[~rare_mask]
        rare_idx = indices[rare_mask]
        common_train_idx, test_idx = train_test_split(
            common_idx,
            test_size=test_size,
            random_state=random_state,
            stratify=labels[common_idx],
        )
        train_idx = np.concatenate([common_train_idx, rare_idx])
    else:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train": features[train_idx],
        "X_test": features[test_idx],
        "y_train": labels[train_idx],
        "y_test": labels[test_idx],
        "power_train": power[train_idx],
        "power_test": power[test_idx],
        "df_train": df.iloc[train_idx].copy(),
        "df_test": df.iloc[test_idx].copy(),
    }
