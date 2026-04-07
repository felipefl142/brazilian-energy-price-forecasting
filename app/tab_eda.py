"""EDA tab: PLD time series, reservoir levels, generation mix, autocorrelations."""

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.helpers import (BRONZE_PLD, BRONZE_RESERVOIR, BRONZE_GENERATION,
                          BRONZE_WEATHER, SUBSYSTEMS, SUBSYSTEM_COLORS,
                          get_duckdb_connection)


def render_eda():
    st.header("Exploratory Data Analysis")

    con = get_duckdb_connection()
    try:
        ts_range = con.execute(
            f"SELECT MIN(week_start), MAX(week_start) FROM read_parquet('{BRONZE_PLD}')"
        ).fetchone()
    except Exception:
        st.warning("Bronze layer data not found. Run the ETL pipeline first.")
        con.close()
        return

    min_ts = pd.to_datetime(ts_range[0]).date()
    max_ts = pd.to_datetime(ts_range[1]).date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=min_ts, min_value=min_ts, max_value=max_ts)
    with col2:
        end_date = st.date_input("To", value=max_ts, min_value=min_ts, max_value=max_ts)

    # --- PLD time series (all 4 subsystems) ---
    st.subheader("PLD — Weekly Spot Price by Subsystem")
    df_pld = con.execute(f"""
        SELECT week_start, subsystem, pld_brl_mwh
        FROM read_parquet('{BRONZE_PLD}')
        WHERE week_start BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        ORDER BY week_start, subsystem
    """).fetchdf()

    if not df_pld.empty:
        fig = go.Figure()
        for sub in SUBSYSTEMS:
            sub_df = df_pld[df_pld["subsystem"] == sub]
            if not sub_df.empty:
                fig.add_trace(go.Scatter(
                    x=sub_df["week_start"], y=sub_df["pld_brl_mwh"],
                    mode="lines", name=sub,
                    line=dict(color=SUBSYSTEM_COLORS[sub], width=1.5),
                ))
        fig.update_layout(
            xaxis_title="Week", yaxis_title="PLD (R$/MWh)",
            template="plotly_dark", height=400, hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        stats = df_pld.groupby("subsystem")["pld_brl_mwh"].agg(["min", "mean", "max"])
        st.caption(" | ".join(
            f"{sub}: avg R${row['mean']:.0f}" for sub, row in stats.iterrows()
        ))

    # --- Reservoir levels ---
    st.subheader("Reservoir Storage Level (% of Useful Volume)")
    try:
        df_res = con.execute(f"""
            SELECT week_start, subsystem, reservoir_pct
            FROM read_parquet('{BRONZE_RESERVOIR}')
            WHERE week_start BETWEEN DATE '{start_date}' AND DATE '{end_date}'
            ORDER BY week_start, subsystem
        """).fetchdf()

        if not df_res.empty:
            fig_res = go.Figure()
            for sub in SUBSYSTEMS:
                sub_df = df_res[df_res["subsystem"] == sub]
                if not sub_df.empty:
                    fig_res.add_trace(go.Scatter(
                        x=sub_df["week_start"], y=sub_df["reservoir_pct"],
                        mode="lines", name=sub,
                        line=dict(color=SUBSYSTEM_COLORS[sub], width=1.5),
                    ))
            # Critical threshold lines
            fig_res.add_hline(y=30, line_dash="dash", line_color="red",
                              annotation_text="Critical (30%)", annotation_position="right")
            fig_res.add_hline(y=50, line_dash="dash", line_color="orange",
                              annotation_text="Low (50%)", annotation_position="right")
            fig_res.update_layout(
                xaxis_title="Week", yaxis_title="Reservoir Level (%)",
                template="plotly_dark", height=350, hovermode="x unified",
            )
            st.plotly_chart(fig_res, use_container_width=True)
            st.caption("Below 30%: high risk of thermal dispatch → PLD spike")
    except Exception as e:
        st.warning(f"Reservoir data not available: {e}")

    # --- Reservoir vs PLD scatter ---
    st.subheader("Reservoir Level vs PLD (SE/CO)")
    try:
        df_scatter = con.execute(f"""
            SELECT r.reservoir_pct, p.pld_brl_mwh,
                   EXTRACT(YEAR FROM r.week_start)::INTEGER AS year
            FROM read_parquet('{BRONZE_RESERVOIR}') r
            JOIN read_parquet('{BRONZE_PLD}') p
                ON r.week_start = p.week_start AND r.subsystem = p.subsystem
            WHERE r.subsystem = 'SE/CO'
              AND r.week_start BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        """).fetchdf()

        if not df_scatter.empty:
            fig_sc = px.scatter(
                df_scatter, x="reservoir_pct", y="pld_brl_mwh", color="year",
                title="SE/CO: higher reservoir → lower PLD (non-linear threshold effect near 30%)",
                labels={"reservoir_pct": "Reservoir Level (%)", "pld_brl_mwh": "PLD (R$/MWh)"},
                template="plotly_dark", opacity=0.6, height=350,
            )
            st.plotly_chart(fig_sc, use_container_width=True)
    except Exception as e:
        st.warning(f"Scatter not available: {e}")

    # --- Generation mix ---
    st.subheader("National Generation Mix")
    try:
        df_gen = con.execute(f"""
            SELECT week_start, source,
                   SUM(generation_avg_mw) AS generation_avg_mw
            FROM read_parquet('{BRONZE_GENERATION}')
            WHERE week_start BETWEEN DATE '{start_date}' AND DATE '{end_date}'
            GROUP BY 1, 2
            ORDER BY 1, 2
        """).fetchdf()

        if not df_gen.empty:
            fig_gen = px.area(
                df_gen, x="week_start", y="generation_avg_mw", color="source",
                title="Generation Mix by Source (stacked area)",
                labels={"generation_avg_mw": "MW (avg)", "week_start": "Week"},
                template="plotly_dark", height=350,
            )
            st.plotly_chart(fig_gen, use_container_width=True)
    except Exception as e:
        st.warning(f"Generation data not available: {e}")

    # --- PLD autocorrelation ---
    st.subheader("PLD Autocorrelation by Lag (SE/CO)")
    try:
        df_acf = con.execute(f"""
            SELECT lag_n, CORR(pld_brl_mwh, lagged) AS autocorr
            FROM (
                SELECT
                    pld_brl_mwh,
                    unnest([1, 2, 3, 4, 8, 13, 26, 52]) AS lag_n,
                    unnest([
                        LAG(pld_brl_mwh, 1)  OVER w,
                        LAG(pld_brl_mwh, 2)  OVER w,
                        LAG(pld_brl_mwh, 3)  OVER w,
                        LAG(pld_brl_mwh, 4)  OVER w,
                        LAG(pld_brl_mwh, 8)  OVER w,
                        LAG(pld_brl_mwh, 13) OVER w,
                        LAG(pld_brl_mwh, 26) OVER w,
                        LAG(pld_brl_mwh, 52) OVER w
                    ]) AS lagged
                FROM read_parquet('{BRONZE_PLD}')
                WHERE subsystem = 'SE/CO'
                WINDOW w AS (ORDER BY week_start)
            )
            GROUP BY lag_n
            ORDER BY lag_n
        """).fetchdf()

        if not df_acf.empty:
            fig_acf = px.bar(
                df_acf, x="lag_n", y="autocorr",
                title="PLD Autocorrelation (annual 52-week lag expected to be strong)",
                labels={"lag_n": "Lag (weeks)", "autocorr": "Pearson r"},
                template="plotly_dark", height=320,
            )
            fig_acf.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_acf, use_container_width=True)
    except Exception as e:
        st.warning(f"Autocorrelation not available: {e}")

    con.close()
