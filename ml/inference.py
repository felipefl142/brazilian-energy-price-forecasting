"""
Inference helper: load model + fetch online features from Feast → run prediction.
Used by the FastAPI serving layer.

Usage (standalone):
    python -m ml.inference --subsystem "SE/CO" --week 2024-W05
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FEATURE_STORE_REPO = BASE_DIR / "feature_store"

SUBSYSTEMS = ["SE/CO", "S", "NE", "N"]


def load_model():
    model_path = MODELS_DIR / "lgbm_multioutput.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run ml/train.py first.")
    return joblib.load(str(model_path))


def get_features_from_feast(subsystem: str, prediction_week: str, store=None) -> pd.DataFrame:
    """
    Retrieve online features from Feast for the given entity.

    Args:
        subsystem:       "SE/CO" | "S" | "NE" | "N"
        prediction_week: ISO week string, e.g. "2024-W05"

    Returns:
        DataFrame with FEATURE_COLS columns (1 row).
    """
    from feast import FeatureStore
    from ml.train import FEATURE_COLS

    if store is None:
        store = FeatureStore(repo_path=str(FEATURE_STORE_REPO))

    feature_refs = [
        f"pld_lag_features:{col}"
        for col in [
            "pld_lag_1w", "pld_lag_2w", "pld_lag_4w", "pld_lag_13w", "pld_lag_52w",
            "pld_roll_4w", "pld_roll_13w",
            "pld_seco_lag_1w", "pld_s_lag_1w", "pld_ne_lag_1w", "pld_n_lag_1w",
            "pld_seco_lag_4w", "pld_s_lag_4w", "pld_ne_lag_4w", "pld_n_lag_4w",
        ]
    ] + [
        f"hydro_features:{col}"
        for col in [
            "reservoir_seco_lag_1w", "reservoir_s_lag_1w", "reservoir_ne_lag_1w", "reservoir_n_lag_1w",
            "reservoir_seco_lag_2w", "reservoir_s_lag_2w", "reservoir_ne_lag_2w", "reservoir_n_lag_2w",
            "reservoir_seco_lag_4w", "reservoir_s_lag_4w", "reservoir_ne_lag_4w", "reservoir_n_lag_4w",
            "reservoir_seco_roll_4w", "reservoir_s_roll_4w", "reservoir_ne_roll_4w", "reservoir_n_roll_4w",
            "ena_lag_1w", "ena_lag_4w", "ena_roll_4w", "ena_anomaly",
        ]
    ] + [
        f"generation_features:{col}"
        for col in ["hydro_share_lag_1w", "thermal_share_lag_1w", "wind_share_lag_1w", "solar_share_lag_1w"]
    ] + [
        f"load_weather_features:{col}"
        for col in ["load_lag_1w", "load_roll_4w", "precip_lag_1w", "precip_lag_2w", "precip_roll_4w", "temp_avg_lag_1w"]
    ] + [
        f"calendar_features:{col}"
        for col in ["week_sin", "week_cos", "month_sin", "month_cos", "is_dry_season",
                    "has_holiday_week", "year", "week_of_year", "is_seco", "is_s", "is_ne", "is_n"]
    ]

    result = store.get_online_features(
        features=feature_refs,
        entity_rows=[{"subsystem": subsystem, "prediction_week": prediction_week}],
    ).to_df()

    for col in FEATURE_COLS:
        if col not in result.columns:
            result[col] = float("nan")

    return result[FEATURE_COLS]


def predict(subsystem: str, prediction_week: str, model=None) -> dict:
    """
    Run inference for a given subsystem and week.

    Returns:
        Dict with 4 weekly horizon forecasts (pld_t_plus_1w through pld_t_plus_4w).
    """
    from ml.train import TARGET_COLS

    if model is None:
        model = load_model()

    features = get_features_from_feast(subsystem, prediction_week)
    preds = model.predict(features)[0]

    return {
        "subsystem": subsystem,
        "prediction_week": prediction_week,
        "forecasts": [
            {"horizon": col.replace("pld_", ""), "pld_brl_mwh": round(float(v), 2)}
            for col, v in zip(TARGET_COLS, preds)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsystem", default="SE/CO", choices=SUBSYSTEMS)
    parser.add_argument("--week", default="2024-W01",
                        help="ISO week string, e.g. '2024-W05'")
    args = parser.parse_args()

    result = predict(args.subsystem, args.week)
    print(json.dumps(result, indent=2))
