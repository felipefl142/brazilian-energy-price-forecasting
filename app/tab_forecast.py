"""Forecast tab: 4-week ahead PLD forecast per subsystem."""

import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.helpers import SILVER_FEATURES, MODELS_DIR, SUBSYSTEMS, SUBSYSTEM_COLORS

BASE_DIR = Path(__file__).resolve().parent.parent
_MODELS_DIR = Path(MODELS_DIR)


@st.cache_resource
def _load_model():
    p = _MODELS_DIR / "lgbm_multioutput.pkl"
    return joblib.load(p) if p.exists() else None


def render_forecast():
    st.header("4-Week Ahead PLD Forecast")

    model = _load_model()
    if model is None:
        st.warning("No trained model found. Run `python -m ml.train` first.")
        return

    # --- Data selector ---
    con = duckdb.connect()
    try:
        ts_range = con.execute(
            f"SELECT MIN(week_start), MAX(week_start) FROM read_parquet('{SILVER_FEATURES}')"
        ).fetchone()
        con.close()
    except Exception:
        con.close()
        st.warning("Silver layer features not found. Run `python -m etl.run_pipeline` first.")
        return

    min_ts = pd.to_datetime(ts_range[0]).date()
    max_ts = pd.to_datetime(ts_range[1]).date()

    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input(
            "Prediction week start",
            value=max_ts, min_value=min_ts, max_value=max_ts,
        )
    with col2:
        selected_sub = st.selectbox("Subsystem", SUBSYSTEMS)

    # --- Load feature vector ---
    con = duckdb.connect()
    try:
        row = con.execute(
            f"SELECT * FROM read_parquet('{SILVER_FEATURES}') "
            f"WHERE CAST(week_start AS DATE) = DATE '{selected_date}' "
            f"AND subsystem = '{selected_sub}' LIMIT 1"
        ).fetchdf()
        con.close()
    except Exception as e:
        con.close()
        st.error(f"Error loading features: {e}")
        return

    if row.empty:
        st.warning(f"No features found for {selected_date} / {selected_sub}.")
        return

    from ml.train import FEATURE_COLS, TARGET_COLS
    X = row[FEATURE_COLS]

    preds = model.predict(X)[0]

    forecast_weeks = pd.date_range(
        start=pd.Timestamp(selected_date) + pd.Timedelta(weeks=1),
        periods=4, freq="W-MON",
    )

    # --- Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_weeks,
        y=preds,
        mode="lines+markers",
        name=f"{selected_sub} — Forecast",
        line=dict(color=SUBSYSTEM_COLORS[selected_sub], width=2),
        marker=dict(size=10),
    ))

    fig.update_layout(
        title=f"4-Week PLD Forecast — {selected_sub} from week of {selected_date}",
        xaxis_title="Week",
        yaxis_title="PLD (R$/MWh)",
        hovermode="x unified",
        template="plotly_dark",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Forecast table ---
    st.subheader("Weekly Forecast Table")
    horizons = [f"t+{h}w" for h in range(1, 5)]
    df_out = pd.DataFrame({
        "Horizon":          horizons,
        "Week Start":       forecast_weeks.strftime("%Y-%m-%d"),
        "Forecast (R$/MWh)": [round(float(v), 2) for v in preds],
    })
    st.dataframe(df_out, use_container_width=True, hide_index=True)

    # --- Historical context ---
    st.subheader("Historical PLD Context")
    try:
        con = duckdb.connect()
        hist_df = con.execute(f"""
            SELECT week_start, subsystem, pld_brl_mwh
            FROM read_parquet('{BASE_DIR}/data/bronze/pld.parquet')
            WHERE subsystem = '{selected_sub}'
              AND week_start >= DATE '{selected_date}' - INTERVAL '52 WEEKS'
            ORDER BY week_start
        """).fetchdf()
        con.close()

        if not hist_df.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=hist_df["week_start"], y=hist_df["pld_brl_mwh"],
                mode="lines", name="Historical PLD",
                line=dict(color="lightgray", width=1),
            ))
            fig2.add_trace(go.Scatter(
                x=forecast_weeks, y=preds,
                mode="lines+markers", name="Forecast",
                line=dict(color=SUBSYSTEM_COLORS[selected_sub], width=2, dash="dash"),
                marker=dict(size=8),
            ))
            fig2.update_layout(
                xaxis_title="Week", yaxis_title="PLD (R$/MWh)",
                template="plotly_dark", height=300,
                title="Last 52 weeks + 4-week forecast",
            )
            st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        pass
