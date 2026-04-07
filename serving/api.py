"""
FastAPI serving endpoint for 4-week ahead Brazilian PLD forecast.

At startup: loads the trained LightGBM model + Feast feature store.
POST /predict: fetches online features from Feast → runs inference → returns 4-week PLD forecast.

Usage:
    uvicorn serving.api:app --reload --port 8000
    # Swagger UI: http://localhost:8000/docs

Example request:
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"subsystem": "SE/CO", "prediction_week": "2024-W05"}'
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FEATURE_STORE_REPO = BASE_DIR / "feature_store"

TARGET_COLS = ["pld_t_plus_1w", "pld_t_plus_2w", "pld_t_plus_3w", "pld_t_plus_4w"]
SUBSYSTEM_LITERAL = Literal["SE/CO", "S", "NE", "N"]

_state: dict = {}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    subsystem: SUBSYSTEM_LITERAL = Field(
        ..., description="Brazilian grid subsystem: SE/CO, S, NE, or N"
    )
    prediction_week: str = Field(
        ...,
        description="ISO week string to forecast FROM, e.g. '2024-W05'",
        examples=["2024-W05"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {"subsystem": "SE/CO", "prediction_week": "2024-W05"}
        }
    }


class WeeklyForecast(BaseModel):
    horizon: str         # "t_plus_1w", "t_plus_2w", etc.
    pld_brl_mwh: float


class ForecastResponse(BaseModel):
    subsystem: str
    prediction_week: str
    forecasts: list[WeeklyForecast]


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = MODELS_DIR / "lgbm_multioutput.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run python -m ml.train first."
        )
    _state["model"] = joblib.load(str(model_path))

    from feast import FeatureStore
    _state["store"] = FeatureStore(repo_path=str(FEATURE_STORE_REPO))

    fc_path = MODELS_DIR / "feature_columns.json"
    if fc_path.exists():
        _state["feature_columns"] = json.loads(fc_path.read_text())

    print(f"Model loaded with {len(_state.get('feature_columns', []))} features.")
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Brazilian Energy Price Forecast API",
    description="4-week ahead PLD forecast for all 4 Brazilian grid subsystems.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    return {
        "model_type": "LightGBM MultiOutputRegressor",
        "horizons": 4,
        "subsystems": ["SE/CO", "S", "NE", "N"],
        "feature_columns": _state.get("feature_columns", []),
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    try:
        from ml.train import FEATURE_COLS
        from ml.inference import get_features_from_feast

        # Pass the pre-loaded store to avoid re-initializing on every request
        features = get_features_from_feast(
            request.subsystem, request.prediction_week,
            store=_state["store"],
        )
        preds = _state["model"].predict(features)[0]

        forecasts = [
            WeeklyForecast(
                horizon=col.replace("pld_", ""),
                pld_brl_mwh=round(float(preds[i]), 2),
            )
            for i, col in enumerate(TARGET_COLS)
        ]

        return ForecastResponse(
            subsystem=request.subsystem,
            prediction_week=request.prediction_week,
            forecasts=forecasts,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
