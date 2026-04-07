"""
Model evaluation: MAE per horizon per subsystem on the OOT test set.
Saves metrics to models/metrics.json.

Usage:
    python -m ml.evaluate
"""

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_DIR = BASE_DIR / "data" / "gold"
MODELS_DIR = BASE_DIR / "models"

ABT_TEST_PATH = str(GOLD_DIR / "abt_test.parquet")

SUBSYSTEMS = ["SE/CO", "S", "NE", "N"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float((np.abs(y_true - y_pred) / denom).mean() * 100.0)


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_split(model, df: pd.DataFrame, split_name: str = "test") -> dict:
    """
    Evaluate model on a given split.
    Returns a dict of metrics keyed by horizon and subsystem.
    """
    from ml.train import FEATURE_COLS, TARGET_COLS

    X = df[FEATURE_COLS]
    Y = df[TARGET_COLS].values
    preds = model.predict(X)

    metrics = {}
    horizons = [1, 2, 3, 4]

    # Overall MAE per horizon
    for i, h in enumerate(horizons):
        metrics[f"mae_t_plus_{h}w"] = _mae(Y[:, i], preds[:, i])
        metrics[f"mape_t_plus_{h}w"] = _mape(Y[:, i], preds[:, i])

    metrics["mae_mean_all"] = float(np.abs(Y - preds).mean())
    metrics["mape_mean_all"] = float(
        sum(metrics[f"mape_t_plus_{h}w"] for h in horizons) / len(horizons)
    )

    # Per-subsystem MAE
    if "subsystem" in df.columns:
        for sub in SUBSYSTEMS:
            mask = df["subsystem"].values == sub
            if mask.sum() == 0:
                continue
            sub_key = sub.lower().replace("/", "")
            for i, h in enumerate(horizons):
                metrics[f"mae_{sub_key}_t_plus_{h}w"] = _mae(Y[mask, i], preds[mask, i])
            metrics[f"mae_{sub_key}_mean"] = float(np.abs(Y[mask] - preds[mask]).mean())

    metrics[f"n_{split_name}_rows"] = int(len(df))
    return metrics


def log_metrics_to_aim(metrics: dict, aim_run, prefix: str = "test"):
    """Log scalar metrics to an Aim run."""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            aim_run.track(value, name=f"{prefix}/{key}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    model_path = MODELS_DIR / "lgbm_multioutput.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run ml/train.py first.")

    print("Loading model...")
    model = joblib.load(model_path)

    print("Evaluating on OOT test set...")
    con = duckdb.connect()
    test_df = con.execute(f"SELECT * FROM read_parquet('{ABT_TEST_PATH}')").fetchdf()
    con.close()

    metrics = evaluate_split(model, test_df, split_name="test")

    print("\n--- OOT Metrics ---")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out_path = MODELS_DIR / "metrics.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
