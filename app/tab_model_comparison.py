"""Model comparison tab: MAE per horizon per subsystem + Aim UI link."""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from app.helpers import SUBSYSTEMS, SUBSYSTEM_COLORS

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
AIM_PORT = 43800


def render_model_comparison():
    st.header("Model Evaluation")

    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        st.warning("No metrics.json found. Run `python -m ml.evaluate` first.")
        return

    metrics = json.loads(metrics_path.read_text())

    # --- Overall MAE per horizon ---
    st.subheader("MAE per Forecast Horizon — OOT Test Set")
    horizons = [1, 2, 3, 4]
    mae_values = [metrics.get(f"mae_t_plus_{h}w") for h in horizons]

    if all(v is not None for v in mae_values):
        fig = go.Figure(go.Bar(
            x=[f"t+{h}w" for h in horizons],
            y=mae_values,
            text=[f"R${v:.1f}" for v in mae_values],
            textposition="outside",
            marker_color=mae_values,
            marker_colorscale="Blues",
        ))
        fig.update_layout(
            xaxis_title="Horizon", yaxis_title="MAE (R$/MWh)",
            title="Higher MAE at longer horizons is expected",
            template="plotly_dark", height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Per-subsystem MAE ---
    st.subheader("MAE by Subsystem (mean across all horizons)")
    sub_data = {}
    for sub in SUBSYSTEMS:
        sub_key = sub.lower().replace("/", "")
        sub_mae = metrics.get(f"mae_{sub_key}_mean")
        if sub_mae is not None:
            sub_data[sub] = sub_mae

    if sub_data:
        fig2 = go.Figure(go.Bar(
            x=list(sub_data.keys()),
            y=list(sub_data.values()),
            marker_color=[SUBSYSTEM_COLORS[s] for s in sub_data],
            text=[f"R${v:.1f}" for v in sub_data.values()],
            textposition="outside",
        ))
        fig2.update_layout(
            xaxis_title="Subsystem", yaxis_title="Mean MAE (R$/MWh)",
            title="MAE varies by subsystem (NE and N often harder to forecast)",
            template="plotly_dark", height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Summary metrics ---
    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean MAE (all)", f"R${metrics.get('mae_mean_all', 0):.2f}/MWh")
    col2.metric("MAPE (mean)", f"{metrics.get('mape_mean_all', 0):.2f}%")
    n_test = metrics.get("n_test_rows", 0)
    col3.metric("Test rows", f"{n_test:,} ({n_test // 4} weeks)")

    # Per-subsystem detail table
    sub_rows = []
    for sub in SUBSYSTEMS:
        sub_key = sub.lower().replace("/", "")
        row = {"Subsystem": sub}
        for h in horizons:
            row[f"MAE t+{h}w"] = metrics.get(f"mae_{sub_key}_t_plus_{h}w")
        row["Mean MAE"] = metrics.get(f"mae_{sub_key}_mean")
        sub_rows.append(row)

    if any(r.get("Mean MAE") is not None for r in sub_rows):
        import pandas as pd
        df = pd.DataFrame(sub_rows)
        float_cols = [c for c in df.columns if c != "Subsystem"]
        st.dataframe(
            df.style.format({c: "R${:.2f}" for c in float_cols if c in df.columns}),
            use_container_width=True, hide_index=True,
        )

    # --- Aim UI link ---
    st.divider()
    st.subheader("Experiment Tracking")
    st.info(
        f"Run `aim up --repo ./aim_logs --port {AIM_PORT}` to open the Aim experiment "
        f"comparison UI for hyperparameter sweep visualization."
    )
    if st.button("Open Aim UI", type="primary"):
        st.markdown(f"[Open Aim UI →](http://localhost:{AIM_PORT})")

    # --- Raw JSON ---
    with st.expander("All metrics (raw JSON)"):
        st.json(metrics)
