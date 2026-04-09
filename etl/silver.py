"""
Silver layer: point-in-time correct feature computation.
Reads from bronze layer, builds the long-format weekly feature table:
  - One row per week per subsystem (4 rows per week)
  - Cross-subsystem features (reservoir, ENA, PLD of all 4 subsystems) in each row
  - Per-subsystem features (load, weather) aligned to the correct subsystem
  - ENA anomaly vs. historical average for the same week-of-year
  - Subsystem one-hot encoding

Output: data/silver/features.parquet

Usage:
    python -m etl.silver
"""

import math
from pathlib import Path

import duckdb
import holidays
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
BRONZE_DIR = BASE_DIR / "data" / "bronze"
SILVER_DIR = BASE_DIR / "data" / "silver"
SQL_DIR = Path(__file__).resolve().parent / "sql"

SUBSYSTEMS = ["SE/CO", "S", "NE", "N"]

# Column mapping: subsystem → column suffix used in the wide-format SQL output
SUB_COL = {"SE/CO": "seco", "S": "s", "NE": "ne", "N": "n"}


def _build_wide_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Run the silver SQL template to build the wide-format weekly feature table."""
    sql_template = (SQL_DIR / "silver_features.sql").read_text()
    sql = (
        sql_template
        .replace("{pld_path}",           str(BRONZE_DIR / "pld.parquet"))
        .replace("{reservoir_path}",     str(BRONZE_DIR / "reservoir.parquet"))
        .replace("{generation_path}",    str(BRONZE_DIR / "generation.parquet"))
        .replace("{load_path}",          str(BRONZE_DIR / "load.parquet"))
        .replace("{interconnection_path}", str(BRONZE_DIR / "interconnection.parquet"))
    )
    print("  [Silver] Running wide-feature SQL template...")
    return con.execute(sql).fetchdf()


def _unpivot_to_long(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the wide-format weekly table to long format:
      4 rows per week (one per subsystem).
    Per-subsystem lag features are renamed from e.g. 'pld_seco_lag_1w' → 'pld_lag_1w'
    for the SE/CO row, etc.
    Cross-subsystem features (all 4 subsystem columns) remain in every row.
    """
    frames = []
    for sub in SUBSYSTEMS:
        col_suffix = SUB_COL[sub]
        df = wide_df.copy()
        df["subsystem"] = sub

        # Rename "this subsystem" lag features to generic names
        df["pld_lag_1w"]  = df[f"pld_{col_suffix}_lag_1w"]
        df["pld_lag_2w"]  = df[f"pld_{col_suffix}_lag_2w"]
        df["pld_lag_4w"]  = df[f"pld_{col_suffix}_lag_4w"]
        df["pld_lag_13w"] = df[f"pld_{col_suffix}_lag_13w"]
        df["pld_lag_52w"] = df[f"pld_{col_suffix}_lag_52w"]
        df["pld_roll_4w"] = df[f"pld_{col_suffix}_roll_4w"]
        df["pld_roll_13w"] = df[f"pld_{col_suffix}_roll_13w"]

        df["reservoir_lag_1w"] = df[f"reservoir_{col_suffix}_lag_1w"]
        df["reservoir_lag_2w"] = df[f"reservoir_{col_suffix}_lag_2w"]
        df["reservoir_lag_4w"] = df[f"reservoir_{col_suffix}_lag_4w"]
        df["reservoir_roll_4w"] = df[f"reservoir_{col_suffix}_roll_4w"]

        df["ena_lag_1w"]   = df[f"ena_{col_suffix}_lag_1w"]
        df["ena_lag_4w"]   = df[f"ena_{col_suffix}_lag_4w"]
        df["ena_roll_4w"]  = df[f"ena_{col_suffix}_roll_4w"]

        frames.append(df)

    long_df = pd.concat(frames, ignore_index=True)
    return long_df.sort_values(["week_start", "subsystem"]).reset_index(drop=True)


def _add_load_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Join per-subsystem load features onto the long-format table."""
    load_path = BRONZE_DIR / "load.parquet"
    if not load_path.exists():
        print("  [Silver] WARNING: load.parquet not found, skipping load features")
        long_df["load_lag_1w"] = float("nan")
        long_df["load_roll_4w"] = float("nan")
        return long_df

    con = duckdb.connect()
    load_df = con.execute(f"""
        SELECT week_start, subsystem,
               LAG(load_mwh, 1) OVER (PARTITION BY subsystem ORDER BY week_start) AS load_lag_1w,
               AVG(load_mwh) OVER (
                   PARTITION BY subsystem ORDER BY week_start
                   ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
               ) AS load_roll_4w
        FROM read_parquet('{str(load_path)}')
        ORDER BY week_start, subsystem
    """).fetchdf()
    con.close()

    load_df["week_start"] = pd.to_datetime(load_df["week_start"]).dt.date
    long_df["week_start"] = pd.to_datetime(long_df["week_start"]).dt.date

    long_df = long_df.merge(load_df[["week_start", "subsystem", "load_lag_1w", "load_roll_4w"]],
                            on=["week_start", "subsystem"], how="left")
    return long_df


def _add_weather_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Join per-subsystem weather features (weekly aggregated) onto the long-format table."""
    weather_path = BRONZE_DIR / "weather.parquet"
    if not weather_path.exists():
        print("  [Silver] WARNING: weather.parquet not found, skipping weather features")
        for col in ["precip_lag_1w", "precip_lag_2w", "precip_roll_4w", "temp_avg_lag_1w"]:
            long_df[col] = float("nan")
        return long_df

    con = duckdb.connect()
    weather_df = con.execute(f"""
        SELECT
            week_start, subsystem,
            LAG(precip_mm_sum, 1) OVER (PARTITION BY subsystem ORDER BY week_start) AS precip_lag_1w,
            LAG(precip_mm_sum, 2) OVER (PARTITION BY subsystem ORDER BY week_start) AS precip_lag_2w,
            AVG(precip_mm_sum) OVER (
                PARTITION BY subsystem ORDER BY week_start
                ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
            ) AS precip_roll_4w,
            LAG(temp_c_avg, 1) OVER (PARTITION BY subsystem ORDER BY week_start) AS temp_avg_lag_1w
        FROM read_parquet('{str(weather_path)}')
        ORDER BY week_start, subsystem
    """).fetchdf()
    con.close()

    weather_df["week_start"] = pd.to_datetime(weather_df["week_start"]).dt.date
    long_df["week_start"] = pd.to_datetime(long_df["week_start"]).dt.date

    long_df = long_df.merge(
        weather_df[["week_start", "subsystem", "precip_lag_1w", "precip_lag_2w",
                    "precip_roll_4w", "temp_avg_lag_1w"]],
        on=["week_start", "subsystem"], how="left"
    )
    return long_df


def _add_thermal_dispatch_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join national thermal dispatch stress signals onto the long-format table.
    Features are national (same value for all 4 subsystems in a given week).
    Data available from 2013+; earlier rows get NaN (handled by the imputer).

    thermal_emergency_mwmed: weekly sum of GFOM (out-of-merit emergency dispatch nationally).
    thermal_nonmerit_share:  (GFOM + inflexibility) / total verified generation — stress ratio.
    """
    path = BRONZE_DIR / "thermal_dispatch.parquet"
    feature_cols = [
        "thermal_emergency_mwmed_lag_1w",
        "thermal_nonmerit_share_lag_1w",
        "thermal_emergency_mwmed_roll_4w",
        "thermal_nonmerit_share_roll_4w",
    ]
    if not path.exists():
        print("  [Silver] WARNING: thermal_dispatch.parquet not found, skipping thermal dispatch features")
        for col in feature_cols:
            long_df[col] = float("nan")
        return long_df

    con = duckdb.connect()
    td_df = con.execute(f"""
        SELECT
            week_start,
            LAG(thermal_emergency_mwmed, 1) OVER (ORDER BY week_start)
                AS thermal_emergency_mwmed_lag_1w,
            LAG(thermal_nonmerit_share, 1) OVER (ORDER BY week_start)
                AS thermal_nonmerit_share_lag_1w,
            AVG(thermal_emergency_mwmed) OVER (
                ORDER BY week_start ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
            ) AS thermal_emergency_mwmed_roll_4w,
            AVG(thermal_nonmerit_share) OVER (
                ORDER BY week_start ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
            ) AS thermal_nonmerit_share_roll_4w
        FROM read_parquet('{str(path)}')
        ORDER BY week_start
    """).fetchdf()
    con.close()

    td_df["week_start"] = pd.to_datetime(td_df["week_start"]).dt.date
    long_df["week_start"] = pd.to_datetime(long_df["week_start"]).dt.date

    long_df = long_df.merge(td_df, on="week_start", how="left")
    return long_df


def _add_ena_anomaly(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ENA anomaly = ena_roll_4w / historical_avg(ena_roll_4w for same week-of-year).
    Values < 1.0 indicate a drought year; values > 1.0 indicate above-average inflows.
    This is the single most domain-relevant feature for PLD forecasting in Brazil.
    """
    long_df["week_of_year"] = pd.to_datetime(long_df["week_start"]).dt.isocalendar().week.astype(int)

    def _compute_anomaly(group: pd.DataFrame) -> pd.Series:
        # Historical avg for each week-of-year using only past observations (no look-ahead)
        anomaly = group["ena_roll_4w"] / group.groupby("week_of_year")["ena_roll_4w"].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        return anomaly

    long_df = long_df.sort_values(["subsystem", "week_start"])
    long_df["ena_anomaly"] = (
        long_df.groupby("subsystem", group_keys=False)
        .apply(_compute_anomaly, include_groups=False)
        .values
    )
    return long_df


def _add_subsystem_onehot(long_df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot encoding columns for subsystem identity."""
    for sub in SUBSYSTEMS:
        col = f"is_{SUB_COL[sub]}"
        long_df[col] = (long_df["subsystem"] == sub).astype(int)
    return long_df


def _add_holiday_flag(long_df: pd.DataFrame) -> pd.DataFrame:
    """Add Brazilian public holiday flag (True if the week contains a major holiday)."""
    years = pd.to_datetime(long_df["week_start"]).dt.year.unique().tolist()
    br_holidays = set()
    for y in years:
        br_holidays.update(holidays.Brazil(years=y).keys())

    def _week_has_holiday(week_start) -> int:
        week_dates = pd.date_range(week_start, periods=7, freq="D").date
        return int(any(d in br_holidays for d in week_dates))

    long_df["has_holiday_week"] = pd.to_datetime(long_df["week_start"]).map(
        lambda ts: _week_has_holiday(ts.date())
    )
    return long_df


def build_silver():
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # Step 1: wide-format SQL
    wide_df = _build_wide_features(con)
    con.close()
    print(f"  [Silver] Wide table: {len(wide_df):,} rows (weekly)")

    # Step 2: unpivot → long format
    long_df = _unpivot_to_long(wide_df)
    print(f"  [Silver] Long table: {len(long_df):,} rows ({len(long_df)//4} weeks × 4 subsystems)")

    # Step 3: per-subsystem features
    long_df = _add_load_features(long_df)
    long_df = _add_weather_features(long_df)
    long_df = _add_thermal_dispatch_features(long_df)

    # Step 4: ENA anomaly
    long_df = _add_ena_anomaly(long_df)

    # Step 5: one-hot subsystem + holiday
    long_df = _add_subsystem_onehot(long_df)
    long_df = _add_holiday_flag(long_df)

    # Step 6: align week_start type
    long_df["week_start"] = pd.to_datetime(long_df["week_start"])

    out_path = str(SILVER_DIR / "features.parquet")
    long_df.to_parquet(out_path, index=False)
    print(f"  [Silver] Done: {len(long_df):,} rows, {len(long_df.columns)} columns → {out_path}")


if __name__ == "__main__":
    print("Building silver layer...")
    build_silver()
