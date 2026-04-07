"""DuckDB Console tab: run arbitrary SQL against any project layer."""

import duckdb
import streamlit as st

from app.helpers import AVAILABLE_TABLES, EXAMPLE_QUERIES, get_duckdb_connection


def render_duckdb():
    st.header("DuckDB Console")

    # --- Quick reference ---
    with st.expander("Available tables (copy into your query)"):
        for name, expr in AVAILABLE_TABLES.items():
            st.code(f"-- {name}\n{expr}", language="sql")

    # --- Example query selector ---
    example_name = st.selectbox(
        "Load example query",
        options=["— custom —"] + list(EXAMPLE_QUERIES.keys()),
    )

    default_sql = (
        EXAMPLE_QUERIES[example_name]
        if example_name != "— custom —"
        else "SELECT * FROM read_parquet('data/bronze/pld.parquet') LIMIT 10"
    )

    # --- SQL editor ---
    with st.form("sql_form"):
        sql = st.text_area("SQL query", value=default_sql, height=180)
        col1, col2 = st.columns([1, 5])
        with col1:
            limit = st.checkbox("Limit 1000 rows", value=True)
        submitted = st.form_submit_button("Run", type="primary")

    if submitted and sql.strip():
        try:
            con = get_duckdb_connection()
            query = f"SELECT * FROM ({sql}) _q LIMIT 1000" if limit else sql
            df = con.execute(query).fetchdf()
            con.close()

            st.success(f"{len(df):,} rows × {len(df.columns)} columns")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV", data=csv,
                file_name="query_result.csv", mime="text/csv",
            )
        except Exception as e:
            st.error(f"Query error: {e}")
