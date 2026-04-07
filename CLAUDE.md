# CLAUDE.md

This file tells Claude Code how to work in this repo.

## Project goal

Forecast the Brazilian weekly spot electricity price (PLD — Preço de Liquidação das Diferenças) for all 4 subsystems (SE/CO, S, NE, N), 1–4 weeks ahead. Power grid demand forecasting is a planned future module.

## Environment

```bash
source .venv/bin/activate        # always activate before running anything
python -m pytest tests/ -v       # run all tests
```

Python 3.12. Aim experiment tracking requires Python ≤ 3.12.

## Running the pipeline

```bash
# Full run (first time)
python -m etl.run_pipeline --start 2005-01-01 --end 2024-12-31

# Incremental update (skip raw collection)
python -m etl.run_pipeline --start 2024-01-01 --end 2024-12-31 --skip-collect

# Individual stages
python -m etl.collect    # raw → data/raw/
python -m etl.bronze     # raw → data/bronze/
python -m etl.silver     # bronze → data/silver/features.parquet
python -m etl.gold       # silver → data/gold/abt*.parquet

# ML
python -m ml.train       # train + save models/lgbm_multioutput.pkl
python -m ml.evaluate    # compute OOT metrics → models/metrics.json

# Feature store (run after gold, before inference)
cd feature_store && feast apply && feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)

# Serving
uvicorn serving.api:app --reload --port 8000

# Dashboard
streamlit run app/main.py

# Experiment tracking UI
aim up --repo ./aim_logs --port 43800
```

## Architecture

```
data/raw/        Hive-partitioned Parquet (year=YYYY), one source per subdirectory
data/bronze/     Weekly tables, schema-normalised (pld, reservoir, generation, load, weather)
data/silver/     features.parquet — long format (4 rows/week × 76 columns), PIT-correct lags
data/gold/       abt.parquet + abt_train.parquet + abt_test.parquet (supervised ABT)
data/feast/      Feast offline store (features.parquet with event_timestamp)
models/          lgbm_multioutput.pkl, feature_columns.json, metrics.json
```

## Key conventions

### Single source of truth for features
`ml/train.py::FEATURE_COLS` is the canonical feature list. It is imported by `serving/api.py` and `ml/evaluate.py`. Never hardcode feature names elsewhere.

### Point-in-time correctness
Every lag/rolling feature must use only data available at prediction time.
- SQL: `ROWS BETWEEN N PRECEDING AND 1 PRECEDING` (never `CURRENT ROW`)
- Python: `series.shift(N)` before any `.rolling()` call
- Tests: `tests/test_silver.py` has 15 PIT correctness tests — run them after any change to `etl/silver.py` or `etl/sql/silver_features.sql`

### Data pipeline stages are independent
Each stage reads from the previous layer and writes to the next. They can be rerun independently:
- Rerun silver+gold without recollecting: `--skip-collect`
- Rerun only one stage: `python -m etl.silver`

### Notebooks are read-only EDA
Notebooks in `notebooks/` are for exploration and visualisation only. They never run as part of the pipeline. Do not add pipeline logic to notebooks.

Current notebooks:
- `api_exploration.ipynb` — ONS/Open-Meteo API discovery, column mapping checklist
- `eda_01_pld.ipynb` — PLD time series, distributions, ACF/PACF, seasonality
- `eda_02_hydrology.ipynb` — Reservoir, ENA anomaly, drought crisis periods
- `eda_03_features.ipynb` — DEFERRED until ONS generation column mapping is resolved
- `eda_04_model_diagnostics.ipynb` — Error distributions, residuals, horizon degradation

### Long-format ABT
The model uses a single multi-output regressor on a long-format table (one row per week per subsystem). This means 4 rows per week and subsystem identity is encoded via one-hot columns (`is_seco`, `is_s`, `is_ne`, `is_n`). Do not pivot back to wide format for modelling.

## Tests

```bash
python -m pytest tests/ -v
```

`tests/test_silver.py` — PIT correctness only. Tests are pure Python (no real data files required). The critical test is `test_no_lookahead_when_future_data_corrupted`.

## Known issues / open TODOs

- ONS resource IDs for reservoir, ENA, load, generation, interconnection are not confirmed. The `etl/collect.py` `ONS_RESOURCES` dict may need updating after running `api_exploration.ipynb`. This also blocks `eda_03_features.ipynb`.
- Generation column names in `etl/bronze.py` are placeholders (`_build_generation`). Update after inspecting live API responses.
- `etl/silver.py` FutureWarning on `include_groups` — fixed; if it reappears after a pandas upgrade, it is in `_add_ena_anomaly`.
- Train/test cut is hardcoded to `2023-12-31` in `ml/train.py::TRAIN_END`. Update annually.

## What NOT to do

- Do not shuffle time series data before splitting — always use the temporal split (`week_start ≤ TRAIN_END`).
- Do not add features that use `week_start` values from the future (no look-ahead).
- Do not run `feast apply` or `feast materialize` from inside a notebook — use the CLI.
- Do not commit model artifacts (`models/*.pkl`) or raw data (`data/`) to git.
