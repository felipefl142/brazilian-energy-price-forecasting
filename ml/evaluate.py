"""
Model evaluation: MAE per horizon per subsystem on the OOT test set.
Also computes AUC (binary "high price week" framing) and logs ROC curves +
feature importances to Aim when a run is provided.
Saves metrics to models/metrics.json.

Usage:
    python -m ml.evaluate
    python -m ml.evaluate --test-start 2025-01-01 --test-end 2025-06-30
    python -m ml.evaluate --auc-threshold 250   # custom high-price threshold (R$/MWh)
"""

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_DIR = BASE_DIR / "data" / "gold"
MODELS_DIR = BASE_DIR / "models"
AIM_REPO = str(BASE_DIR / "aim_logs")

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
    return metrics, preds


def compute_auc_metrics(
    y_true: np.ndarray, preds: np.ndarray, threshold: float
) -> dict:
    """
    Compute ROC-AUC per horizon treating PLD > threshold as the positive class.
    Uses the raw regression prediction as the classification score.
    """
    from sklearn.metrics import roc_auc_score

    metrics = {}
    auc_values = []
    horizons = [1, 2, 3, 4]

    for i, h in enumerate(horizons):
        y_bin = (y_true[:, i] > threshold).astype(int)
        n_pos = y_bin.sum()
        if n_pos == 0 or n_pos == len(y_bin):
            continue
        auc = roc_auc_score(y_bin, preds[:, i])
        metrics[f"auc_t_plus_{h}w"] = float(auc)
        auc_values.append(auc)

    if auc_values:
        metrics["auc_mean_all"] = float(np.mean(auc_values))
        metrics["auc_threshold"] = float(threshold)

    return metrics


def plot_roc_curves(
    y_true: np.ndarray, preds: np.ndarray, threshold: float, split_name: str
):
    """Return a matplotlib Figure with ROC curves for each of the 4 horizons."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    horizons = [1, 2, 3, 4]

    for i, h in enumerate(horizons):
        ax = axes[i]
        y_bin = (y_true[:, i] > threshold).astype(int)
        n_pos = y_bin.sum()

        if n_pos == 0 or n_pos == len(y_bin):
            ax.set_title(f"t+{h}w — insufficient classes")
            continue

        fpr, tpr, _ = roc_curve(y_bin, preds[:, i])
        auc = roc_auc_score(y_bin, preds[:, i])

        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC t+{h}w  |  {split_name}  (thresh={threshold:.0f})")
        ax.legend(loc="lower right")

    fig.tight_layout()
    return fig


def get_feature_importances(model) -> tuple[np.ndarray, list[str]]:
    """
    Extract mean feature importances across all 4 target estimators.
    Returns (mean_importances array, feature_cols list).
    """
    from ml.train import FEATURE_COLS

    estimators = model.named_steps["model"].estimators_
    all_imp = np.stack([e.feature_importances_ for e in estimators])
    return all_imp.mean(axis=0), FEATURE_COLS


def plot_feature_importances(model, top_n: int = 30):
    """Return a matplotlib Figure with the top-N mean feature importances."""
    import matplotlib.pyplot as plt

    mean_imp, feature_cols = get_feature_importances(model)
    order = np.argsort(mean_imp)[::-1][:top_n]
    names = [feature_cols[j] for j in order]
    values = mean_imp[order]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.28)))
    ax.barh(names[::-1], values[::-1])
    ax.set_xlabel("Mean importance (gain, avg across 4 horizons)")
    ax.set_title(f"Top-{top_n} feature importances")
    fig.tight_layout()
    return fig


def log_metrics_to_aim(metrics: dict, aim_run, prefix: str = "test"):
    """Log scalar metrics to an Aim run."""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            aim_run.track(value, name=f"{prefix}/{key}")


def log_auc_to_aim(auc_metrics: dict, aim_run, prefix: str = "test"):
    """Log AUC scalars and ROC Figure to an Aim run."""
    try:
        from aim import Figure as AimFigure
        has_aim_figure = True
    except ImportError:
        has_aim_figure = False

    for key, value in auc_metrics.items():
        if isinstance(value, (int, float)):
            aim_run.track(value, name=f"{prefix}/{key}")


def log_roc_to_aim(fig, aim_run, prefix: str = "test"):
    """Log a matplotlib ROC Figure to Aim."""
    try:
        from aim import Figure as AimFigure
        aim_run.track(AimFigure(fig), name=f"{prefix}/roc_curves")
    except ImportError:
        pass


def log_feature_importances_to_aim(model, aim_run, prefix: str = ""):
    """Log feature importance figure and dict to Aim."""
    try:
        from aim import Figure as AimFigure
        fig = plot_feature_importances(model, top_n=30)
        aim_run.track(AimFigure(fig), name=f"{prefix}feature_importances_plot")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:
        pass

    mean_imp, feature_cols = get_feature_importances(model)
    order = np.argsort(mean_imp)[::-1]
    aim_run[f"{prefix}top20_feature_importances"] = {
        feature_cols[j]: float(mean_imp[j]) for j in order[:20]
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate LightGBM PLD forecast model")
    parser.add_argument(
        "--test-start", type=str, default=None,
        help="Only evaluate rows with week_start >= this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--test-end", type=str, default=None,
        help="Only evaluate rows with week_start <= this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--auc-threshold", type=float, default=None,
        help="High-price threshold for AUC (R$/MWh). Default: P75 from models/pld_stats.json"
    )
    parser.add_argument("--no-aim", action="store_true", help="Disable Aim tracking")
    args = parser.parse_args()

    model_path = MODELS_DIR / "lgbm_multioutput.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run ml/train.py first.")

    print("Loading model...")
    model = joblib.load(model_path)

    # Resolve AUC threshold
    threshold = args.auc_threshold
    if threshold is None:
        stats_path = MODELS_DIR / "pld_stats.json"
        if stats_path.exists():
            pld_stats = json.loads(stats_path.read_text())
            threshold = pld_stats["p75"]
            print(f"  AUC threshold: P75 = {threshold:.1f} R$/MWh (from pld_stats.json)")
        else:
            threshold = 200.0
            print(f"  AUC threshold: {threshold:.1f} R$/MWh (default fallback — run train.py to save pld_stats.json)")

    print("Loading OOT test set...")
    con = duckdb.connect()
    test_df = con.execute(f"SELECT * FROM read_parquet('{ABT_TEST_PATH}')").fetchdf()
    con.close()

    # Apply date filters
    if args.test_start:
        before = len(test_df)
        test_df = test_df[test_df["week_start"] >= args.test_start]
        print(f"  --test-start {args.test_start}: {before} → {len(test_df)} rows")
    if args.test_end:
        before = len(test_df)
        test_df = test_df[test_df["week_start"] <= args.test_end]
        print(f"  --test-end {args.test_end}: {before} → {len(test_df)} rows")

    if len(test_df) == 0:
        raise ValueError("No rows left after applying date filters.")

    from ml.train import TARGET_COLS
    Y = test_df[TARGET_COLS].values

    print(f"\nEvaluating on {len(test_df):,} OOT rows...")
    metrics, preds = evaluate_split(model, test_df, split_name="test")
    auc_metrics = compute_auc_metrics(Y, preds, threshold=threshold)

    print("\n--- OOT Metrics ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n--- AUC Metrics ---")
    for k, v in sorted(auc_metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    all_metrics = {**metrics, **auc_metrics}
    out_path = MODELS_DIR / "metrics.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nMetrics saved to {out_path}")

    # Aim logging
    if not args.no_aim:
        try:
            from aim import Run
            aim_run = Run(repo=AIM_REPO)
            aim_run["eval_test_start"] = args.test_start
            aim_run["eval_test_end"] = args.test_end
            aim_run["auc_threshold"] = threshold
            aim_run["run_type"] = "evaluate"

            log_metrics_to_aim(metrics, aim_run, prefix="test")
            log_auc_to_aim(auc_metrics, aim_run, prefix="test")

            roc_fig = plot_roc_curves(Y, preds, threshold=threshold, split_name="OOT")
            log_roc_to_aim(roc_fig, aim_run, prefix="test")
            import matplotlib.pyplot as plt
            plt.close(roc_fig)

            log_feature_importances_to_aim(model, aim_run, prefix="")

            aim_run.close()
            print(f"\nRun logged to Aim. Start UI with:")
            print(f"  aim up --repo ./aim_logs --port 43800")
        except ImportError:
            print("  WARNING: aim not installed — tracking disabled. pip install aim")


if __name__ == "__main__":
    main()
