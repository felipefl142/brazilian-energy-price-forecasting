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
# Full run (first time, ~30 min mostly downloads)
python -m etl.run_pipeline --start 2005-01-01 --end 2024-12-31

# Incremental update (skip raw collection, rebuild layers only)
python -m etl.run_pipeline --start 2024-01-01 --end 2024-12-31 --skip-collect

# Individual stages
python -m etl.collect --start 2024-01-01 --end 2024-12-31   # raw → data/raw/
python -m etl.bronze                                          # raw → data/bronze/
python -m etl.silver                                          # bronze → data/silver/features.parquet
python -m etl.gold                                            # silver → data/gold/abt*.parquet

# Collect only one source (pld | ons | weather)
python -m etl.collect --start 2024-01-01 --end 2024-12-31 --only ons

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
data/silver/     features.parquet — long format (4 rows/week × 113 columns), PIT-correct lags
data/gold/       abt.parquet + abt_train.parquet + abt_test.parquet (supervised ABT)
data/feast/      Feast offline store (features.parquet with event_timestamp)
models/          lgbm_multioutput.pkl, feature_columns.json, metrics.json
```

## Data sources

All free, no API keys required.

| Source | Dataset | URL pattern |
|--------|---------|-------------|
| PLD/CMO | ONS S3 CSV | `cmo_se/CMO_SEMANAL_{year}.csv` (2005+) |
| EAR (reservoir %) | ONS S3 Parquet | `ear_subsistema_di/EAR_DIARIO_SUBSISTEMA_{year}.parquet` (2000+) |
| ENA (inflows) | ONS S3 | `ena_subsistema_di/ENA_DIARIO_SUBSISTEMA_{year}.{parquet\|csv}` (parquet 2021+, CSV before) |
| Load | ONS S3 Parquet | `carga_energia_di/CARGA_ENERGIA_{year}.parquet` (2000+) |
| Generation | ONS S3 Parquet | `geracao_usina_2_ho/GERACAO_USINA-2_{year}.parquet` annual (2000–2021), `.._{year}_{mm:02d}.parquet` monthly (2022+) |
| Interconnection | ONS S3 | `intercambio_nacional_ho/INTERCAMBIO_NACIONAL_{year}.{parquet\|csv}` (parquet 2023+, CSV before) |
| Weather | Open-Meteo archive | 4 cities (one per subsystem), daily → weekly |

All hosted at `https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/`.

**Note:** `carga-energia-verificada` requires API auth — use `carga-energia` (free) instead.

## Key conventions

### Single source of truth for features
`ml/train.py::FEATURE_COLS` is the canonical feature list. It is imported by `serving/api.py` and `ml/evaluate.py`. Never hardcode feature names elsewhere.

### Point-in-time correctness
Every lag/rolling feature must use only data available at prediction time.
- SQL: `ROWS BETWEEN N PRECEDING AND 1 PRECEDING` (never `CURRENT ROW`)
- Python: `series.shift(N)` before any `.rolling()` call
- Tests: `tests/test_silver.py` has 15 PIT correctness tests — run them after any change to `etl/silver.py` or `etl/sql/silver_features.sql`

### Week alignment
PLD uses **Friday** as `week_start` (confirmed from raw data). All daily ONS data is aggregated to Friday-starting weeks in bronze using:
```sql
date - ((EXTRACT(DOW FROM date)::INT + 2) % 7) * INTERVAL '1 day'
```
This ensures JOIN keys in `silver_features.sql` align with the PLD spine.

### Subsystem ID mapping
ONS uses single-letter codes (`SE`, `S`, `NE`, `N`). Bronze maps `SE → SE/CO`. All bronze outputs use the project-standard names (`SE/CO`, `S`, `NE`, `N`).

### Data pipeline stages are independent
Each stage reads from the previous layer and writes to the next. They can be rerun independently:
- Rerun silver+gold without recollecting: `--skip-collect`
- Rerun only one stage: `python -m etl.silver`

### Notebooks are read-only EDA
Notebooks in `notebooks/` are for exploration and visualisation only. They never run as part of the pipeline. Do not add pipeline logic to notebooks.

Current notebooks:
- `api_exploration.ipynb` — ONS/Open-Meteo API discovery, resource ID checklist
- `eda_01_pld.ipynb` — PLD time series, distributions, ACF/PACF, seasonality
- `eda_02_hydrology.ipynb` — Reservoir, ENA anomaly, drought crisis periods
- `eda_03_features.ipynb` — Feature→target correlations, generation mix (now unblocked)
- `eda_04_model_diagnostics.ipynb` — Error distributions, residuals, horizon degradation

### Long-format ABT
The model uses a single multi-output regressor on a long-format table (one row per week per subsystem). This means 4 rows per week and subsystem identity is encoded via one-hot columns (`is_seco`, `is_s`, `is_ne`, `is_n`). Do not pivot back to wide format for modelling.

## Tests

```bash
python -m pytest tests/ -v
```

`tests/test_silver.py` — 15 PIT correctness tests (ENA anomaly, subsystem one-hot, PLD lags). Tests are pure Python (no real data files required). The critical test is `test_no_lookahead_when_future_data_corrupted`.

## Known issues / open TODOs

- `eda_03_features.ipynb` — generation column mapping is now resolved; notebook can be built
- Train/test cut is hardcoded to `2024-12-31` in `etl/gold.py::TRAIN_END`. Update annually.
- OOT MAPE is inflated (~586%) because 2024 includes weeks with PLD near the floor (~R$30/MWh); MAE is more meaningful (~R$51 at t+1w, ~R$108 at t+4w)

## What NOT to do

- Do not shuffle time series data before splitting — always use the temporal split (`week_start ≤ TRAIN_END`).
- Do not add features that use `week_start` values from the future (no look-ahead).
- Do not run `feast apply` or `feast materialize` from inside a notebook — use the CLI.
- Do not commit model artifacts (`models/*.pkl`) or raw data (`data/`) to git.
