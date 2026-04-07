"""
Feast feature store definitions for the Brazilian energy price forecast project.

Entity:   pld_point — (subsystem, prediction_week) pair
          subsystem: "SE/CO" | "S" | "NE" | "N"
          prediction_week: ISO week string, e.g. "2024-W01"

Sources:  data/feast/features.parquet (prepared by etl/gold.py)

Feature views (ttl=14 days — weekly data, recheck fortnightly):
  pld_lag_features    — PLD lags + rolling means for all 4 subsystems
  hydro_features      — reservoir levels + ENA + anomaly for all 4 subsystems
  generation_features — generation mix shares (hydro, wind, solar, thermal)
  load_weather_features — load + weather per subsystem
  calendar_features   — cyclical time encodings + dry season + holiday flag

Usage:
    cd feature_store
    feast apply
    feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float64, Int64, String

FEATURES_PARQUET = "../data/feast/features.parquet"

# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------
pld_point = Entity(
    name="pld_point",
    join_keys=["subsystem", "prediction_week"],
    description="Weekly PLD observation point (subsystem + ISO week string)",
)

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
features_source = FileSource(
    name="energy_features_source",
    path=FEATURES_PARQUET,
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# ---------------------------------------------------------------------------
# Feature views
# ---------------------------------------------------------------------------

pld_lag_features = FeatureView(
    name="pld_lag_features",
    entities=[pld_point],
    ttl=timedelta(days=14),
    schema=[
        # This subsystem
        Field(name="pld_lag_1w",    dtype=Float64),
        Field(name="pld_lag_2w",    dtype=Float64),
        Field(name="pld_lag_4w",    dtype=Float64),
        Field(name="pld_lag_13w",   dtype=Float64),
        Field(name="pld_lag_52w",   dtype=Float64),
        Field(name="pld_roll_4w",   dtype=Float64),
        Field(name="pld_roll_13w",  dtype=Float64),
        # Cross-subsystem (1-week lag all subsystems)
        Field(name="pld_seco_lag_1w",  dtype=Float64),
        Field(name="pld_s_lag_1w",     dtype=Float64),
        Field(name="pld_ne_lag_1w",    dtype=Float64),
        Field(name="pld_n_lag_1w",     dtype=Float64),
        Field(name="pld_seco_lag_4w",  dtype=Float64),
        Field(name="pld_s_lag_4w",     dtype=Float64),
        Field(name="pld_ne_lag_4w",    dtype=Float64),
        Field(name="pld_n_lag_4w",     dtype=Float64),
    ],
    source=features_source,
    online=True,
)

hydro_features = FeatureView(
    name="hydro_features",
    entities=[pld_point],
    ttl=timedelta(days=14),
    schema=[
        # Reservoir % — all subsystems, 3 lags
        Field(name="reservoir_seco_lag_1w",  dtype=Float64),
        Field(name="reservoir_seco_lag_2w",  dtype=Float64),
        Field(name="reservoir_seco_lag_4w",  dtype=Float64),
        Field(name="reservoir_s_lag_1w",     dtype=Float64),
        Field(name="reservoir_s_lag_2w",     dtype=Float64),
        Field(name="reservoir_s_lag_4w",     dtype=Float64),
        Field(name="reservoir_ne_lag_1w",    dtype=Float64),
        Field(name="reservoir_ne_lag_2w",    dtype=Float64),
        Field(name="reservoir_ne_lag_4w",    dtype=Float64),
        Field(name="reservoir_n_lag_1w",     dtype=Float64),
        Field(name="reservoir_n_lag_2w",     dtype=Float64),
        Field(name="reservoir_n_lag_4w",     dtype=Float64),
        Field(name="reservoir_seco_roll_4w", dtype=Float64),
        Field(name="reservoir_s_roll_4w",    dtype=Float64),
        Field(name="reservoir_ne_roll_4w",   dtype=Float64),
        Field(name="reservoir_n_roll_4w",    dtype=Float64),
        # ENA lags + anomaly
        Field(name="ena_lag_1w",     dtype=Float64),
        Field(name="ena_lag_4w",     dtype=Float64),
        Field(name="ena_roll_4w",    dtype=Float64),
        Field(name="ena_anomaly",    dtype=Float64),
    ],
    source=features_source,
    online=True,
)

generation_features = FeatureView(
    name="generation_features",
    entities=[pld_point],
    ttl=timedelta(days=14),
    schema=[
        Field(name="hydro_share_lag_1w",   dtype=Float64),
        Field(name="thermal_share_lag_1w", dtype=Float64),
        Field(name="wind_share_lag_1w",    dtype=Float64),
        Field(name="solar_share_lag_1w",   dtype=Float64),
    ],
    source=features_source,
    online=True,
)

load_weather_features = FeatureView(
    name="load_weather_features",
    entities=[pld_point],
    ttl=timedelta(days=14),
    schema=[
        Field(name="load_lag_1w",       dtype=Float64),
        Field(name="load_roll_4w",      dtype=Float64),
        Field(name="precip_lag_1w",     dtype=Float64),
        Field(name="precip_lag_2w",     dtype=Float64),
        Field(name="precip_roll_4w",    dtype=Float64),
        Field(name="temp_avg_lag_1w",   dtype=Float64),
    ],
    source=features_source,
    online=True,
)

calendar_features = FeatureView(
    name="calendar_features",
    entities=[pld_point],
    ttl=timedelta(days=14),
    schema=[
        Field(name="week_sin",          dtype=Float64),
        Field(name="week_cos",          dtype=Float64),
        Field(name="month_sin",         dtype=Float64),
        Field(name="month_cos",         dtype=Float64),
        Field(name="is_dry_season",     dtype=Int64),
        Field(name="has_holiday_week",  dtype=Int64),
        Field(name="year",              dtype=Int64),
        Field(name="week_of_year",      dtype=Int64),
        # Subsystem one-hot
        Field(name="is_seco",           dtype=Int64),
        Field(name="is_s",              dtype=Int64),
        Field(name="is_ne",             dtype=Int64),
        Field(name="is_n",              dtype=Int64),
    ],
    source=features_source,
    online=True,
)
