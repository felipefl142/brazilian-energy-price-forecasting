"""
Subsystem Analysis tab: reservoir gauges, interconnection flows,
PLD spread across subsystems, dry season overlay.
"""

import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from app.helpers import (BRONZE_PLD, BRONZE_RESERVOIR, BRONZE_INTERCONNECTION,
                          SUBSYSTEMS, SUBSYSTEM_COLORS, get_duckdb_connection)


def _latest_reservoir(con: duckdb.DuckDBPyConnection) -> dict:
    """Return latest reservoir % for each subsystem."""
    try:
        rows = con.execute(f"""
            SELECT subsystem, reservoir_pct
            FROM read_parquet('{BRONZE_RESERVOIR}')
            WHERE week_start = (SELECT MAX(week_start) FROM read_parquet('{BRONZE_RESERVOIR}'))
        """).fetchdf()
        return dict(zip(rows["subsystem"], rows["reservoir_pct"])) if not rows.empty else {}
    except Exception:
        return {}


def render_subsystem():
    st.header("Subsystem Analysis")

    con = get_duckdb_connection()

    # --- Reservoir level gauges ---
    st.subheader("Current Reservoir Levels")
    levels = _latest_reservoir(con)

    if levels:
        cols = st.columns(4)
        for i, sub in enumerate(SUBSYSTEMS):
            pct = levels.get(sub)
            if pct is not None:
                color = "red" if pct < 30 else ("orange" if pct < 50 else "green")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(pct, 1),
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": sub},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 30], "color": "rgba(255,0,0,0.15)"},
                            {"range": [30, 50], "color": "rgba(255,165,0,0.15)"},
                            {"range": [50, 100], "color": "rgba(0,128,0,0.1)"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 3}, "value": 50},
                    },
                    number={"suffix": "%"},
                ))
                fig.update_layout(height=220, template="plotly_dark", margin=dict(t=40, b=10))
                cols[i].plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Reservoir data not available yet.")

    # --- PLD spread over time ---
    st.subheader("PLD Spread (Max - Min across Subsystems)")
    try:
        df_spread = con.execute(f"""
            SELECT
                week_start,
                MAX(pld_brl_mwh) - MIN(pld_brl_mwh) AS pld_spread,
                MAX(pld_brl_mwh) AS pld_max,
                MIN(pld_brl_mwh) AS pld_min,
                AVG(pld_brl_mwh) AS pld_avg
            FROM read_parquet('{BRONZE_PLD}')
            GROUP BY week_start
            ORDER BY week_start
        """).fetchdf()

        if not df_spread.empty:
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(
                x=df_spread["week_start"], y=df_spread["pld_spread"],
                mode="lines", name="Spread (max-min)",
                line=dict(color="#FFA500", width=1.5),
            ))
            fig_spread.update_layout(
                xaxis_title="Week", yaxis_title="Spread (R$/MWh)",
                template="plotly_dark", height=300, hovermode="x unified",
                title="Large spread = subsystems decoupled (usually NE/N isolated from SE/CO)",
            )
            st.plotly_chart(fig_spread, use_container_width=True)
    except Exception as e:
        st.warning(f"Spread chart not available: {e}")

    # --- PLD by subsystem + dry season overlay ---
    st.subheader("Annual PLD Pattern: Wet vs Dry Season")
    try:
        df_seasonal = con.execute(f"""
            SELECT
                EXTRACT(WEEK FROM week_start) AS week_of_year,
                subsystem,
                AVG(pld_brl_mwh) AS avg_pld
            FROM read_parquet('{BRONZE_PLD}')
            GROUP BY 1, 2
            ORDER BY 1, 2
        """).fetchdf()

        if not df_seasonal.empty:
            fig_seas = go.Figure()
            for sub in SUBSYSTEMS:
                sub_df = df_seasonal[df_seasonal["subsystem"] == sub]
                if not sub_df.empty:
                    fig_seas.add_trace(go.Scatter(
                        x=sub_df["week_of_year"], y=sub_df["avg_pld"],
                        mode="lines", name=sub,
                        line=dict(color=SUBSYSTEM_COLORS[sub], width=2),
                    ))
            # Dry season band (weeks ~18–44 ≈ May–October)
            fig_seas.add_vrect(
                x0=18, x1=44,
                fillcolor="rgba(255,165,0,0.1)", line_width=0,
                annotation_text="Dry Season", annotation_position="top left",
            )
            fig_seas.update_layout(
                xaxis_title="Week of Year", yaxis_title="Avg PLD (R$/MWh)",
                template="plotly_dark", height=350, hovermode="x unified",
                title="Historical average PLD by week of year (all history)",
            )
            st.plotly_chart(fig_seas, use_container_width=True)
    except Exception as e:
        st.warning(f"Seasonal chart not available: {e}")

    # --- Interconnection flows ---
    st.subheader("Subsystem Interconnection Flows")
    try:
        df_flow = con.execute(f"""
            SELECT
                from_subsystem, to_subsystem,
                AVG(ABS(flow_mwh)) AS avg_abs_flow_mwh
            FROM read_parquet('{BRONZE_INTERCONNECTION}')
            GROUP BY from_subsystem, to_subsystem
        """).fetchdf()

        if not df_flow.empty:
            # Sankey diagram for average flows
            nodes = list(set(df_flow["from_subsystem"]) | set(df_flow["to_subsystem"]))
            node_idx = {n: i for i, n in enumerate(nodes)}
            node_colors = [SUBSYSTEM_COLORS.get(n, "#aaa") for n in nodes]

            fig_sankey = go.Figure(go.Sankey(
                node=dict(
                    pad=15, thickness=20,
                    label=nodes,
                    color=node_colors,
                ),
                link=dict(
                    source=[node_idx[r["from_subsystem"]] for _, r in df_flow.iterrows()],
                    target=[node_idx[r["to_subsystem"]] for _, r in df_flow.iterrows()],
                    value=df_flow["avg_abs_flow_mwh"].tolist(),
                ),
            ))
            fig_sankey.update_layout(
                title="Average Interchange Flows between Subsystems (MWh/week)",
                template="plotly_dark", height=350,
            )
            st.plotly_chart(fig_sankey, use_container_width=True)
        else:
            st.info("Interconnection data not yet available.")
    except Exception as e:
        st.info(f"Interconnection chart not available: {e}")

    con.close()
