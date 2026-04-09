"""
Model training: LightGBM MultiOutputRegressor (4 weekly horizons × 4 subsystems = long format).
Tracks all experiments with Aim for superior run comparison UI.

The FEATURE_COLS list is the single source of truth — imported by serving and evaluation.

Usage:
    python -m ml.train                     # train with default config
    python -m ml.train --n-estimators 300  # override hyperparameters

Aim UI:
    aim init --repo ./aim_logs             # once, to initialize repo
    aim up --repo ./aim_logs --port 43800  # then open localhost:43800
"""

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

from ml.imputer import MixedImputer

PLD_STATS_PATH_NAME = "pld_stats.json"

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_DIR = BASE_DIR / "data" / "gold"
MODELS_DIR = BASE_DIR / "models"
AIM_REPO = str(BASE_DIR / "aim_logs")

ABT_TRAIN_PATH = str(GOLD_DIR / "abt_train.parquet")

# ---------------------------------------------------------------------------
# Feature and target column definitions
# Single source of truth — imported by serving/api.py and ml/evaluate.py
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Subsystem identity (one-hot)
    "is_seco", "is_s", "is_ne", "is_n",

    # PLD lags — this subsystem
    "pld_lag_1w", "pld_lag_2w", "pld_lag_4w", "pld_lag_13w", "pld_lag_52w",
    "pld_roll_4w", "pld_roll_13w",

    # PLD — cross-subsystem (1-week and 4-week lags, all 4)
    "pld_seco_lag_1w", "pld_s_lag_1w", "pld_ne_lag_1w", "pld_n_lag_1w",
    "pld_seco_lag_4w", "pld_s_lag_4w", "pld_ne_lag_4w", "pld_n_lag_4w",

    # Reservoir % — all 4 subsystems, 3 lags each
    "reservoir_seco_lag_1w", "reservoir_s_lag_1w", "reservoir_ne_lag_1w", "reservoir_n_lag_1w",
    "reservoir_seco_lag_2w", "reservoir_s_lag_2w", "reservoir_ne_lag_2w", "reservoir_n_lag_2w",
    "reservoir_seco_lag_4w", "reservoir_s_lag_4w", "reservoir_ne_lag_4w", "reservoir_n_lag_4w",
    "reservoir_seco_roll_4w", "reservoir_s_roll_4w", "reservoir_ne_roll_4w", "reservoir_n_roll_4w",

    # Reservoir absolute (MWmonth) — all 4 subsystems, 2 lags each
    # Captures scale that % ignores: 30% of 200k MWmonth ≠ 30% of 50k MWmonth
    "reservoir_mwmonth_seco_lag_1w", "reservoir_mwmonth_s_lag_1w",
    "reservoir_mwmonth_ne_lag_1w", "reservoir_mwmonth_n_lag_1w",
    "reservoir_mwmonth_seco_lag_4w", "reservoir_mwmonth_s_lag_4w",
    "reservoir_mwmonth_ne_lag_4w", "reservoir_mwmonth_n_lag_4w",

    # ENA — all 4 subsystems
    "ena_seco_lag_1w", "ena_s_lag_1w", "ena_ne_lag_1w", "ena_n_lag_1w",
    "ena_seco_roll_4w", "ena_s_roll_4w", "ena_ne_roll_4w", "ena_n_roll_4w",

    # ENA anomaly — this subsystem (ENA vs. historical avg for same week-of-year)
    # KEY FEATURE: values < 1 = drought year → high PLD
    "ena_anomaly",

    # Thermal dispatch stress signals (national, 2022+; pre-2022 rows filled with 0)
    # thermal_emergency: GFOM dispatch = ONS emergency override of merit order → leading PLD indicator
    # thermal_nonmerit_share: (GFOM + inflexibility) / total generation → stress ratio
    # NaN = no data collected (pre-2022), which semantically means no emergency dispatch → fill 0
    "thermal_emergency_mwmed_lag_1w", "thermal_nonmerit_share_lag_1w",
    "thermal_emergency_mwmed_roll_4w", "thermal_nonmerit_share_roll_4w",

    # Generation mix — national
    "hydro_share_lag_1w", "thermal_share_lag_1w", "wind_share_lag_1w", "solar_share_lag_1w",

    # Load — this subsystem
    "load_lag_1w", "load_roll_4w",

    # Weather — this subsystem's representative city
    "precip_lag_1w", "precip_lag_2w", "precip_roll_4w", "temp_avg_lag_1w",

    # Calendar
    "week_sin", "week_cos", "month_sin", "month_cos",
    "is_dry_season", "has_holiday_week", "year",
]

# 4 horizons (1w–4w ahead) — same subsystem as the feature row
TARGET_COLS = ["pld_t_plus_1w", "pld_t_plus_2w", "pld_t_plus_3w", "pld_t_plus_4w"]

# Default LightGBM hyperparameters
# Conservative regularization is crucial given ~1,400 training rows (small dataset)
DEFAULT_CONFIG = {
    "n_estimators": 300,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


# Thermal dispatch features are NaN for pre-2022 rows because the dataset didn't exist yet.
# NaN here means "no emergency dispatch occurred" (the correct interpretation), so fill with 0.
# All other features use median imputation for the rare structural gaps (early ENA years, etc.).
ZERO_FILL_FEATURES = [
    "thermal_emergency_mwmed_lag_1w",
    "thermal_nonmerit_share_lag_1w",
    "thermal_emergency_mwmed_roll_4w",
    "thermal_nonmerit_share_roll_4w",
]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_pipeline(config: dict) -> Pipeline:
    imputer = MixedImputer(zero_fill_cols=ZERO_FILL_FEATURES)
    return Pipeline([
        ("imputer", imputer),
        ("model", MultiOutputRegressor(LGBMRegressor(**config), n_jobs=4)),
    ])


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(train_df: pd.DataFrame, config: dict | None = None, aim_run=None) -> Pipeline:
    """
    Train a LightGBM MultiOutputRegressor on the training ABT.

    Args:
        train_df: Gold ABT training split.
        config:   LightGBM hyperparameter dict (defaults to DEFAULT_CONFIG).
        aim_run:  Optional aim.Run instance for experiment tracking.

    Returns:
        Fitted sklearn Pipeline (imputer + MultiOutputRegressor).
    """
    if config is None:
        config = DEFAULT_CONFIG

    X = train_df[FEATURE_COLS]
    Y = train_df[TARGET_COLS]

    pipeline = _build_pipeline(config)
    print(f"  Training: {len(X):,} rows × {len(FEATURE_COLS)} features × {len(TARGET_COLS)} targets")
    pipeline.fit(X, Y)

    if aim_run is not None:
        aim_run["n_features"] = len(FEATURE_COLS)
        aim_run["n_targets"] = len(TARGET_COLS)
        aim_run["n_train_rows"] = len(X)
        aim_run["subsystems"] = ["SE/CO", "S", "NE", "N"]

        # Log feature importances (mean across all 4 target estimators)
        estimators = pipeline.named_steps["model"].estimators_
        all_importances = np.stack([e.feature_importances_ for e in estimators])
        mean_importances = all_importances.mean(axis=0)
        # Store per-horizon importances as well
        horizon_importances = {
            f"t_plus_{h}w": dict(zip(FEATURE_COLS, all_importances[i].tolist()))
            for i, h in enumerate([1, 2, 3, 4])
        }
        aim_run["feature_importances"] = dict(zip(FEATURE_COLS, mean_importances.tolist()))
        aim_run["feature_importances_per_horizon"] = horizon_importances

    return pipeline


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM PLD forecast model")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_CONFIG["n_estimators"])
    parser.add_argument("--num-leaves", type=int, default=DEFAULT_CONFIG["num_leaves"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--no-aim", action="store_true", help="Disable Aim tracking")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    config = {**DEFAULT_CONFIG, **{
        "n_estimators": args.n_estimators,
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
    }}

    print("Loading training data...")
    con = duckdb.connect()
    train_df = con.execute(f"SELECT * FROM read_parquet('{ABT_TRAIN_PATH}')").fetchdf()
    con.close()
    print(f"  → {len(train_df):,} rows loaded")

    # Save PLD distribution stats for evaluation thresholds
    all_pld_train = train_df[TARGET_COLS].values.flatten()
    pld_stats = {
        "p50": float(np.percentile(all_pld_train, 50)),
        "p75": float(np.percentile(all_pld_train, 75)),
        "p90": float(np.percentile(all_pld_train, 90)),
    }
    (MODELS_DIR / PLD_STATS_PATH_NAME).write_text(json.dumps(pld_stats, indent=2))
    print(f"  PLD stats saved (p50={pld_stats['p50']:.1f}, p75={pld_stats['p75']:.1f}, p90={pld_stats['p90']:.1f})")

    aim_run = None
    if not args.no_aim:
        try:
            from aim import Run
            aim_run = Run(repo=AIM_REPO)
            aim_run["config"] = config
            aim_run["model_type"] = "LightGBM MultiOutputRegressor"
            print(f"  Aim run started: {aim_run.hash}")
        except ImportError:
            print("  WARNING: aim not installed — tracking disabled. pip install aim")

    print("\nTraining model...")
    model = train(train_df, config=config, aim_run=aim_run)

    print("\nEvaluating on training data (in-sample)...")
    from ml.evaluate import evaluate_split, log_metrics_to_aim, log_feature_importances_to_aim
    train_metrics, _ = evaluate_split(model, train_df, split_name="train")

    # Log metrics and feature importances to Aim
    if aim_run is not None:
        log_metrics_to_aim(train_metrics, aim_run, prefix="train")
        log_feature_importances_to_aim(model, aim_run, prefix="")

    # Save model
    model_path = MODELS_DIR / "lgbm_multioutput.pkl"
    joblib.dump(model, model_path)
    print(f"\n  → Model saved to {model_path}")

    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(FEATURE_COLS, indent=2))

    if aim_run is not None:
        aim_run.close()
        print(f"\nRun logged. Start Aim UI with:")
        print(f"  aim up --repo ./aim_logs --port 43800")


if __name__ == "__main__":
    main()
