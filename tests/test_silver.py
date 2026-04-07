"""
Unit tests for etl/silver.py — point-in-time (PIT) correctness.

These tests verify that NO feature computed for week T uses data from week T or later.
A PIT violation would cause overly optimistic training metrics and silent degradation
in live inference.

Run:
    pytest tests/test_silver.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etl.silver import _add_ena_anomaly, _add_subsystem_onehot, SUB_COL

SUBSYSTEMS = ["SE/CO", "S", "NE", "N"]


# ---------------------------------------------------------------------------
# Test data factory
# ---------------------------------------------------------------------------

def make_synthetic_long_df(n_weeks: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Minimal long-format DataFrame for testing silver transformations.
    Simulates seasonal ENA values for all 4 subsystems.
    """
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2010-01-04", periods=n_weeks, freq="W-MON")
    rows = []
    for sub in SUBSYSTEMS:
        t = np.arange(n_weeks)
        ena = 5000 + 3000 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 300, n_weeks)
        for i, week in enumerate(weeks):
            rows.append({
                "week_start": week,
                "subsystem": sub,
                "ena_roll_4w": max(0.0, float(ena[i])),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ENA Anomaly PIT Tests
# ---------------------------------------------------------------------------

class TestEnaAnomalyPIT:
    """
    Point-in-time correctness tests for _add_ena_anomaly.

    The ENA anomaly is the most important feature in the model (values < 1.0 signal
    drought → high PLD). A lookahead bug here would be the worst possible data leak.
    """

    def test_first_row_per_subsystem_is_nan(self):
        """Row 0 per subsystem must be NaN — no prior week-of-year history exists."""
        df = make_synthetic_long_df(100)
        result = _add_ena_anomaly(df.copy())

        for sub in SUBSYSTEMS:
            sub_df = result[result["subsystem"] == sub].sort_values("week_start")
            assert pd.isna(sub_df["ena_anomaly"].iloc[0]), (
                f"Row 0 for {sub} should be NaN (no history yet)"
            )

    def test_values_appear_after_enough_history(self):
        """After seeing the same week-of-year at least twice, anomaly should be non-null."""
        df = make_synthetic_long_df(110)
        result = _add_ena_anomaly(df.copy())

        for sub in SUBSYSTEMS:
            sub_df = result[result["subsystem"] == sub].sort_values("week_start")
            later = sub_df.iloc[55:]  # well past 52-week mark
            assert later["ena_anomaly"].notna().sum() > 0, (
                f"No valid anomaly after 55 weeks for {sub}"
            )

    def test_no_lookahead_when_future_data_corrupted(self):
        """
        Critical PIT test: corrupting future ena_roll_4w must not change past anomalies.

        This is the gold standard check — if any future value affects a past result,
        the feature has a lookahead bias and the model will overfit at training time.
        """
        df = make_synthetic_long_df(80)
        result_clean = _add_ena_anomaly(df.copy())

        # Corrupt all rows at or after week index 40
        cutoff = df["week_start"].sort_values().unique()[40]
        df_corrupt = df.copy()
        df_corrupt.loc[df_corrupt["week_start"] >= cutoff, "ena_roll_4w"] = 999_999_999.0
        result_corrupt = _add_ena_anomaly(df_corrupt)

        for sub in SUBSYSTEMS:
            before_clean = (
                result_clean[
                    (result_clean["subsystem"] == sub) &
                    (result_clean["week_start"] < cutoff)
                ]
                .sort_values("week_start")["ena_anomaly"]
                .reset_index(drop=True)
            )
            before_corrupt = (
                result_corrupt[
                    (result_corrupt["subsystem"] == sub) &
                    (result_corrupt["week_start"] < cutoff)
                ]
                .sort_values("week_start")["ena_anomaly"]
                .reset_index(drop=True)
            )

            assert before_clean.isna().equals(before_corrupt.isna()), (
                f"NaN mask changed for {sub} before cutoff — lookahead detected"
            )
            valid_clean = before_clean.dropna().values
            valid_corrupt = before_corrupt.dropna().values
            np.testing.assert_allclose(
                valid_clean, valid_corrupt, rtol=1e-9,
                err_msg=(
                    f"ENA anomaly for {sub} changed before the cutoff when "
                    f"future data was corrupted — this is a lookahead violation"
                ),
            )

    def test_drought_produces_anomaly_below_one(self):
        """Persistently low ENA (drought) should yield anomaly < 1.0."""
        df = make_synthetic_long_df(160)
        late_weeks = df["week_start"].sort_values().unique()[-30:]
        df.loc[
            (df["subsystem"] == "SE/CO") & (df["week_start"].isin(late_weeks)),
            "ena_roll_4w",
        ] = 50.0  # far below 5000 baseline

        result = _add_ena_anomaly(df)
        drought_rows = result[
            (result["subsystem"] == "SE/CO") &
            (result["week_start"].isin(late_weeks[5:]))  # skip transition weeks
        ]["ena_anomaly"].dropna()

        assert len(drought_rows) > 0, "No valid anomaly rows in drought period"
        assert (drought_rows < 1.0).all(), (
            f"Drought anomaly should be < 1.0, got max={drought_rows.max():.3f}"
        )

    def test_above_average_inflow_produces_anomaly_above_one(self):
        """Persistently high ENA (flood year) should yield anomaly > 1.0."""
        df = make_synthetic_long_df(160)
        late_weeks = df["week_start"].sort_values().unique()[-30:]
        df.loc[
            (df["subsystem"] == "SE/CO") & (df["week_start"].isin(late_weeks)),
            "ena_roll_4w",
        ] = 50_000.0  # far above 5000 baseline

        result = _add_ena_anomaly(df)
        flood_rows = result[
            (result["subsystem"] == "SE/CO") &
            (result["week_start"].isin(late_weeks[5:]))
        ]["ena_anomaly"].dropna()

        assert len(flood_rows) > 0, "No valid anomaly rows in flood period"
        assert (flood_rows > 1.0).all(), (
            f"Flood anomaly should be > 1.0, got min={flood_rows.min():.3f}"
        )

    def test_long_run_mean_is_near_one(self):
        """
        By construction (ratio to historical average), the long-run mean should be ≈ 1.0.
        A systematic deviation would indicate a normalization error.
        """
        df = make_synthetic_long_df(260)  # 5 years
        result = _add_ena_anomaly(df)
        valid = result["ena_anomaly"].dropna()
        assert abs(valid.mean() - 1.0) < 0.15, (
            f"Long-run anomaly mean {valid.mean():.3f} deviates from 1.0"
        )

    def test_subsystems_are_computed_independently(self):
        """Corrupting one subsystem's ENA must not affect another subsystem's anomaly."""
        df = make_synthetic_long_df(80)
        result_clean = _add_ena_anomaly(df.copy())

        df_corrupt = df.copy()
        df_corrupt.loc[df_corrupt["subsystem"] == "NE", "ena_roll_4w"] = 999_999_999.0
        result_corrupt = _add_ena_anomaly(df_corrupt)

        # SE/CO, S, N should be unaffected
        for sub in ["SE/CO", "S", "N"]:
            clean = (
                result_clean[result_clean["subsystem"] == sub]
                .sort_values("week_start")["ena_anomaly"]
                .reset_index(drop=True)
            )
            corrupt = (
                result_corrupt[result_corrupt["subsystem"] == sub]
                .sort_values("week_start")["ena_anomaly"]
                .reset_index(drop=True)
            )
            valid_mask = clean.notna()
            np.testing.assert_allclose(
                clean[valid_mask].values, corrupt[valid_mask].values, rtol=1e-9,
                err_msg=f"Corrupting NE contaminated {sub} anomaly — subsystems not independent"
            )


# ---------------------------------------------------------------------------
# Subsystem One-Hot Tests
# ---------------------------------------------------------------------------

class TestSubsystemOnehot:
    """Tests for _add_subsystem_onehot."""

    def test_exactly_one_active_column_per_row(self):
        """Each row must have exactly one '1' across the four is_* columns."""
        df = make_synthetic_long_df(10)
        result = _add_subsystem_onehot(df.copy())
        onehot_cols = [f"is_{v}" for v in SUB_COL.values()]
        row_sums = result[onehot_cols].sum(axis=1)
        assert (row_sums == 1).all(), "Each row must have exactly one active subsystem flag"

    def test_correct_column_is_flagged(self):
        """The is_<x> column must be 1 only for rows belonging to subsystem x."""
        df = make_synthetic_long_df(5)
        result = _add_subsystem_onehot(df.copy())
        for sub, col_suffix in SUB_COL.items():
            col = f"is_{col_suffix}"
            assert (result[result["subsystem"] == sub][col] == 1).all(), (
                f"is_{col_suffix} should be 1 for {sub} rows"
            )
            assert (result[result["subsystem"] != sub][col] == 0).all(), (
                f"is_{col_suffix} should be 0 for non-{sub} rows"
            )

    def test_all_subsystems_present(self):
        """All four subsystem flags must exist in the output."""
        df = make_synthetic_long_df(5)
        result = _add_subsystem_onehot(df.copy())
        for col_suffix in SUB_COL.values():
            assert f"is_{col_suffix}" in result.columns, (
                f"Column is_{col_suffix} missing from output"
            )


# ---------------------------------------------------------------------------
# PLD Lag Conceptual PIT Tests
#
# The actual lag SQL is in etl/sql/silver_features.sql. These tests verify
# that the shift() + rolling() idiom used throughout silver.py is correct —
# they document the expected behaviour and guard against future refactors
# introducing accidental lookahead.
# ---------------------------------------------------------------------------

class TestPLDLagPIT:

    @pytest.fixture
    def pld(self) -> pd.Series:
        rng = np.random.default_rng(0)
        return pd.Series(100.0 + rng.normal(0, 20, 80), name="pld_brl_mwh")

    def test_shift1_equals_previous_value(self, pld):
        """shift(1) at index i must equal the raw value at index i-1."""
        lag = pld.shift(1)
        for i in range(1, len(pld)):
            assert lag.iloc[i] == pld.iloc[i - 1]

    def test_first_n_values_are_nan(self, pld):
        """First n values of shift(n) must all be NaN."""
        for n in [1, 2, 4, 13, 52]:
            lagged = pld.shift(n)
            assert lagged.iloc[:n].isna().all(), (
                f"First {n} values of lag_{n}w must be NaN"
            )

    def test_rolling_window_excludes_current_week(self, pld):
        """
        roll_4w = shift(1).rolling(4) — current week T must NOT be in the window.
        Window for T=i should be [i-4, i-3, i-2, i-1].
        """
        roll = pld.shift(1).rolling(4, min_periods=1).mean()
        for i in range(5, len(pld)):
            expected = pld.iloc[i - 4:i].mean()
            assert abs(roll.iloc[i] - expected) < 1e-9, (
                f"roll_4w at {i}: expected {expected:.6f}, got {roll.iloc[i]:.6f}"
            )

    def test_rolling_unaffected_by_current_week_corruption(self, pld):
        """
        Modifying value at T must not change roll_4w at T.
        But it MUST change roll_4w at T+1 (value at T enters the next window).
        """
        roll_orig = pld.shift(1).rolling(4, min_periods=1).mean()

        pld_mod = pld.copy()
        pld_mod.iloc[10] = 1_000_000.0
        roll_mod = pld_mod.shift(1).rolling(4, min_periods=1).mean()

        # roll at T=10 unchanged (T=10 not in its own window)
        assert abs(roll_orig.iloc[10] - roll_mod.iloc[10]) < 1e-9, (
            "roll_4w at T=10 changed when pld[10] was corrupted — current week is in window!"
        )
        # roll at T=11 must change (T=10 is now in window [7,8,9,10])
        assert abs(roll_orig.iloc[11] - roll_mod.iloc[11]) > 1.0, (
            "roll_4w at T=11 should change when pld[10] was corrupted"
        )

    def test_rolling_13w_uses_13_preceding_weeks(self, pld):
        """roll_13w window for T=i must cover [i-13, …, i-1] (13 values)."""
        roll = pld.shift(1).rolling(13, min_periods=1).mean()
        i = 20
        expected = pld.iloc[i - 13:i].mean()
        assert abs(roll.iloc[i] - expected) < 1e-9, (
            f"roll_13w at {i}: expected {expected:.6f}, got {roll.iloc[i]:.6f}"
        )
