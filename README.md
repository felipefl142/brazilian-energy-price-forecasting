# Brazilian Energy Price Forecasting

Weekly PLD (spot electricity price) forecasts for all 4 Brazilian subsystems — 1 to 4 weeks ahead.

## What is PLD?

PLD (Preço de Liquidação das Diferenças) is the weekly marginal cost of operation for Brazil's National Interconnected System (SIN). It drives settlement prices for all electricity traders and is the primary short-term price signal in the ~40 GW hydro-dominated grid.

Brazil has 4 subsystems: **SE/CO** (Southeast/Centre-West), **S** (South), **NE** (Northeast), **N** (North).

## Architecture

```
data sources          ETL pipeline            ML / Serving
─────────────────     ───────────────         ──────────────────
ONS CMO (S3 CSV)  ─►  collect.py             ml/train.py
                  ─►  bronze.py   (weekly)    │  LightGBM
ONS CKAN API      ─►  silver.py   (features)  │  MultiOutputRegressor
(reservoir, ENA,  ─►  gold.py     (ABT + split)  4 horizons × 4 subsystems
 load, generation,                            │
 interconnection)     feature_store/          ▼
                       Feast (SQLite)     serving/api.py   (FastAPI)
Open-Meteo                               app/main.py      (Streamlit)
(4 cities)
```

**Data layers:**

| Layer | Path | Content |
|---|---|---|
| Raw | `data/raw/` | Hive-partitioned Parquet per source/year |
| Bronze | `data/bronze/` | Weekly tables, normalised schema |
| Silver | `data/silver/features.parquet` | 76 PIT-correct features, long format (4 rows/week) |
| Gold | `data/gold/abt*.parquet` | Supervised ABT + temporal train/test split |

## Data Sources

All free, no API keys required.

| Source | Data | Coverage |
|---|---|---|
| [ONS CMO S3](https://dados.ons.org.br/dataset/cmo-semanal) | Weekly PLD/CMO by subsystem | 2005–present |
| [ONS CKAN API](https://dados.ons.org.br) | Reservoir %, ENA, load, generation, interconnection | varies |
| [Open-Meteo](https://open-meteo.com) | Daily weather (precipitation, temperature, wind) | 1940–present |

## Setup

```bash
git clone https://github.com/your-org/brazilian-energy-price-forecasting
cd brazilian-energy-price-forecasting

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Python 3.12 required (Aim experiment tracking constraint).

## Running the pipeline

```bash
# Full pipeline — first run (~30 min, mostly API calls)
python -m etl.run_pipeline --start 2005-01-01 --end 2024-12-31

# Incremental weekly update
python -m etl.run_pipeline --start 2024-01-01 --end 2024-12-31 --skip-collect

# Train model
python -m ml.train

# Evaluate OOT metrics
python -m ml.evaluate
```

## Serving

```bash
# FastAPI (port 8000)
uvicorn serving.api:app --reload --port 8000

# Example prediction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"subsystem": "SE/CO", "prediction_week": "2024-W05"}'
```

Response:
```json
{
  "subsystem": "SE/CO",
  "prediction_week": "2024-W05",
  "forecasts": [
    {"horizon": "t_plus_1w", "pld_brl_mwh": 85.42},
    {"horizon": "t_plus_2w", "pld_brl_mwh": 87.15},
    {"horizon": "t_plus_3w", "pld_brl_mwh": 88.70},
    {"horizon": "t_plus_4w", "pld_brl_mwh": 90.33}
  ]
}
```

## Dashboard

```bash
streamlit run app/main.py
```

Five tabs: Forecast · Model Comparison · EDA · Subsystem Analysis · DuckDB Console.

## Feature store

```bash
cd feature_store
feast apply
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
```

## Experiment tracking

```bash
aim up --repo ./aim_logs --port 43800
# Open http://localhost:43800
```

## Key features

The most predictive features, in order of domain importance:

1. **ENA anomaly** — `ena_roll_4w / historical_avg(same week-of-year)`. Values < 1.0 signal drought → high PLD.
2. **PLD lags** — 1w, 2w, 4w, 13w, 52w for all 4 subsystems (cross-subsystem correlation is high, r > 0.9).
3. **Reservoir %** — storage level for all 4 subsystems, 1–4 week lags.
4. **Dry season flag** — weeks 18–44 (May–October), when ENA drops and reservoirs are drawn down.
5. **Generation mix** — hydro/thermal/wind/solar shares lagged 1 week.
6. **Calendar cyclical** — `week_sin/cos`, `month_sin/cos`.

Point-in-time correctness is enforced at every stage: all features use only data available at the time of prediction (`ROWS BETWEEN N PRECEDING AND 1 PRECEDING` in SQL, `shift(N)` in Python).

## Tests

```bash
python -m pytest tests/ -v
```

`tests/test_silver.py` — 15 tests covering PIT correctness for ENA anomaly, PLD lags, and rolling windows.

## Project structure

```
etl/                 Data pipeline (collect → bronze → silver → gold)
  collect.py         Raw ingestion from 3 sources
  bronze.py          Schema normalisation, weekly aggregation
  silver.py          PIT-correct feature engineering (76 features)
  gold.py            ABT construction + temporal split
  sql/               DuckDB SQL templates
ml/                  Machine learning
  train.py           LightGBM MultiOutputRegressor + Aim tracking
  evaluate.py        OOT MAE/MAPE metrics
  inference.py       Load model + Feast features → predict
  FEATURE_COLS       Single source of truth (76 features)
serving/             FastAPI REST API
  api.py             /health  /model-info  /predict
feature_store/       Feast feature store
  definitions.py     5 feature views (PLD, hydro, generation, load/weather, calendar)
  feature_store.yaml SQLite online + Parquet offline
app/                 Streamlit dashboard
  main.py            Entry point
  tab_*.py           One file per tab
notebooks/           EDA only — never run in pipeline
  api_exploration.ipynb
  eda_01_pld.ipynb
  eda_02_hydrology.ipynb
  eda_04_model_diagnostics.ipynb
tests/               Unit tests
  test_silver.py     PIT correctness tests
```

## Roadmap

- [ ] Resolve ONS resource IDs and generation column mapping (see `notebooks/api_exploration.ipynb`)
- [ ] Complete `eda_03_features.ipynb` (feature→target correlation, generation mix EDA)
- [ ] Power grid demand forecasting module
- [ ] Confidence intervals / quantile regression
- [ ] Automated weekly retraining pipeline
