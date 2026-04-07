import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="Brazilian Energy Price Forecast",
    page_icon="🇧🇷",
    layout="wide",
)

st.title("Brazilian Energy Price Forecast — PLD by Subsystem")
st.caption("4-week ahead PLD forecasting | DuckDB · Feast · Aim · LightGBM")

tab_forecast, tab_models, tab_eda, tab_subsystem, tab_sql = st.tabs([
    "Forecast",
    "Model Comparison",
    "EDA",
    "Subsystem Analysis",
    "DuckDB Console",
])

with tab_forecast:
    from app.tab_forecast import render_forecast
    render_forecast()

with tab_models:
    from app.tab_model_comparison import render_model_comparison
    render_model_comparison()

with tab_eda:
    from app.tab_eda import render_eda
    render_eda()

with tab_subsystem:
    from app.tab_subsystem import render_subsystem
    render_subsystem()

with tab_sql:
    from app.tab_duckdb import render_duckdb
    render_duckdb()
