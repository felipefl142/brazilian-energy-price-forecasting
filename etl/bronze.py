"""
Bronze layer: schema normalization, type casting, and weekly aggregation via DuckDB.
Reads hive-partitioned raw Parquet and writes clean weekly files:

  data/bronze/pld.parquet          — weekly PLD per subsystem (R$/MWh)
  data/bronze/reservoir.parquet    — weekly reservoir level (%) + ENA (GWh) per subsystem
  data/bronze/generation.parquet   — weekly generation by source, national (MW avg)
  data/bronze/load.parquet         — weekly load per subsystem (MWh/week)
  data/bronze/interconnection.parquet  — weekly net interchange flows (MWh/week)
  data/bronze/weather.parquet      — weekly weather per subsystem city (precip, temp)

IMPORTANT: Column name mappings in each function must match the actual ONS API response.
Run notebook 01_data_ingestion.ipynb first to inspect raw schemas and update mappings here.

Usage:
    python -m etl.bronze
"""

from pathlib import Path

import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
BRONZE_DIR = BASE_DIR / "data" / "bronze"


def _row_count(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    return con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]


# ---------------------------------------------------------------------------
# PLD
# ---------------------------------------------------------------------------

def _build_pld(con: duckdb.DuckDBPyConnection):
    raw = str(RAW_DIR / "pld" / "**" / "*.parquet")
    out = str(BRONZE_DIR / "pld.parquet")
    print("  [Bronze] Building pld.parquet...")
    con.execute(f"""
        COPY (
            SELECT
                TRY_CAST(week_start AS DATE)      AS week_start,
                TRIM(CAST(subsystem AS VARCHAR))  AS subsystem,
                TRY_CAST(pld_brl_mwh AS DOUBLE)  AS pld_brl_mwh
            FROM read_parquet('{raw}', hive_partitioning=true)
            WHERE week_start IS NOT NULL
              AND pld_brl_mwh IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY CAST(week_start AS DATE), TRIM(CAST(subsystem AS VARCHAR))
                ORDER BY CAST(week_start AS DATE)
            ) = 1
            ORDER BY week_start, subsystem
        ) TO '{out}' (FORMAT PARQUET)
    """)
    print(f"    → {_row_count(con, Path(out)):,} rows")


# ---------------------------------------------------------------------------
# Reservoir + ENA
# ---------------------------------------------------------------------------
# Expected ONS columns (verify in notebook 01):
#   reservoir: date col (e.g. dat_referencia), subsystem col, reservoir_pct col
#   ena:       date col, subsystem col, ena_gwh col
# We combine both into one bronze table keyed by (week_start, subsystem).

def _build_reservoir(con: duckdb.DuckDBPyConnection):
    raw_reservoir = str(RAW_DIR / "reservoir" / "**" / "*.parquet")
    raw_ena = str(RAW_DIR / "ena" / "**" / "*.parquet")
    out = str(BRONZE_DIR / "reservoir.parquet")
    print("  [Bronze] Building reservoir.parquet (reservoir + ENA)...")

    # Load reservoir raw and inspect columns interactively (run notebook 01 first)
    # Placeholder column names — UPDATE these after running notebook 01:
    #   dat_semana_inicio → week start date
    #   nom_submercado    → subsystem name (SE/CO, S, NE, N)
    #   val_pct_volume_util → reservoir % of useful volume
    #   val_ena_bruta_gwh   → ENA (GWh weekly)
    try:
        df_res = con.execute(f"""
            SELECT
                CAST(dat_semana_inicio AS DATE)            AS week_start,
                TRIM(CAST(nom_submercado AS VARCHAR))      AS subsystem,
                TRY_CAST(val_pct_volume_util AS DOUBLE)   AS reservoir_pct
            FROM read_parquet('{raw_reservoir}', hive_partitioning=true)
            WHERE dat_semana_inicio IS NOT NULL
        """).fetchdf()
    except Exception as e:
        print(f"    WARNING: reservoir fetch failed ({e}) — using empty frame")
        df_res = pd.DataFrame(columns=["week_start", "subsystem", "reservoir_pct"])

    try:
        df_ena = con.execute(f"""
            SELECT
                CAST(dat_semana_inicio AS DATE)           AS week_start,
                TRIM(CAST(nom_submercado AS VARCHAR))     AS subsystem,
                TRY_CAST(val_ena_bruta_gwh AS DOUBLE)    AS ena_gwh
            FROM read_parquet('{raw_ena}', hive_partitioning=true)
            WHERE dat_semana_inicio IS NOT NULL
        """).fetchdf()
    except Exception as e:
        print(f"    WARNING: ENA fetch failed ({e}) — using empty frame")
        df_ena = pd.DataFrame(columns=["week_start", "subsystem", "ena_gwh"])

    if df_res.empty and df_ena.empty:
        print("    → No reservoir/ENA data available yet.")
        return

    df = pd.merge(df_res, df_ena, on=["week_start", "subsystem"], how="outer")
    df = df.sort_values(["week_start", "subsystem"])
    df.to_parquet(out, index=False)
    print(f"    → {len(df):,} rows")


# ---------------------------------------------------------------------------
# Generation by source (national weekly)
# ---------------------------------------------------------------------------

def _build_generation(con: duckdb.DuckDBPyConnection):
    raw = str(RAW_DIR / "generation" / "**" / "*.parquet")
    out = str(BRONZE_DIR / "generation.parquet")
    print("  [Bronze] Building generation.parquet...")
    # Expected ONS columns (UPDATE after running notebook 01):
    #   dat_semana_inicio  → week start
    #   nom_tipo_geracao   → source type (HIDRO, EOLICA, SOLAR, TERMELETRICA, NUCLEAR)
    #   val_geracao_mwmed  → average generation MW for the week
    try:
        con.execute(f"""
            COPY (
                SELECT
                    CAST(dat_semana_inicio AS DATE)           AS week_start,
                    TRIM(UPPER(CAST(nom_tipo_geracao AS VARCHAR))) AS source,
                    TRY_CAST(val_geracao_mwmed AS DOUBLE)    AS generation_avg_mw
                FROM read_parquet('{raw}', hive_partitioning=true)
                WHERE dat_semana_inicio IS NOT NULL
                  AND val_geracao_mwmed IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY CAST(dat_semana_inicio AS DATE),
                                 TRIM(UPPER(CAST(nom_tipo_geracao AS VARCHAR)))
                    ORDER BY CAST(dat_semana_inicio AS DATE)
                ) = 1
                ORDER BY week_start, source
            ) TO '{out}' (FORMAT PARQUET)
        """)
        print(f"    → {_row_count(con, Path(out)):,} rows")
    except Exception as e:
        print(f"    WARNING: generation build failed ({e})")


# ---------------------------------------------------------------------------
# Load by subsystem (weekly)
# ---------------------------------------------------------------------------

def _build_load(con: duckdb.DuckDBPyConnection):
    raw = str(RAW_DIR / "load" / "**" / "*.parquet")
    out = str(BRONZE_DIR / "load.parquet")
    print("  [Bronze] Building load.parquet...")
    # Expected ONS columns (UPDATE after notebook 01):
    #   dat_semana_inicio  → week start
    #   nom_submercado     → subsystem
    #   val_carga_energia  → energy consumed (MWh/week)
    try:
        con.execute(f"""
            COPY (
                SELECT
                    CAST(dat_semana_inicio AS DATE)           AS week_start,
                    TRIM(CAST(nom_submercado AS VARCHAR))     AS subsystem,
                    TRY_CAST(val_carga_energia AS DOUBLE)    AS load_mwh
                FROM read_parquet('{raw}', hive_partitioning=true)
                WHERE dat_semana_inicio IS NOT NULL
                  AND val_carga_energia IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY CAST(dat_semana_inicio AS DATE),
                                 TRIM(CAST(nom_submercado AS VARCHAR))
                    ORDER BY CAST(dat_semana_inicio AS DATE)
                ) = 1
                ORDER BY week_start, subsystem
            ) TO '{out}' (FORMAT PARQUET)
        """)
        print(f"    → {_row_count(con, Path(out)):,} rows")
    except Exception as e:
        print(f"    WARNING: load build failed ({e})")


# ---------------------------------------------------------------------------
# Interconnection flows (weekly, net per subsystem pair)
# ---------------------------------------------------------------------------

def _build_interconnection(con: duckdb.DuckDBPyConnection):
    raw = str(RAW_DIR / "interconnection" / "**" / "*.parquet")
    out = str(BRONZE_DIR / "interconnection.parquet")
    print("  [Bronze] Building interconnection.parquet...")
    # Expected ONS columns (UPDATE after notebook 01):
    #   dat_semana_inicio    → week start
    #   nom_submercado_orig  → origin subsystem
    #   nom_submercado_dest  → destination subsystem
    #   val_intercambio_mwh  → interchange energy (MWh/week)
    try:
        con.execute(f"""
            COPY (
                SELECT
                    CAST(dat_semana_inicio AS DATE)               AS week_start,
                    TRIM(CAST(nom_submercado_orig AS VARCHAR))    AS from_subsystem,
                    TRIM(CAST(nom_submercado_dest AS VARCHAR))    AS to_subsystem,
                    TRY_CAST(val_intercambio_mwh AS DOUBLE)      AS flow_mwh
                FROM read_parquet('{raw}', hive_partitioning=true)
                WHERE dat_semana_inicio IS NOT NULL
                  AND val_intercambio_mwh IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY CAST(dat_semana_inicio AS DATE),
                                 TRIM(CAST(nom_submercado_orig AS VARCHAR)),
                                 TRIM(CAST(nom_submercado_dest AS VARCHAR))
                    ORDER BY CAST(dat_semana_inicio AS DATE)
                ) = 1
                ORDER BY week_start, from_subsystem, to_subsystem
            ) TO '{out}' (FORMAT PARQUET)
        """)
        print(f"    → {_row_count(con, Path(out)):,} rows")
    except Exception as e:
        print(f"    WARNING: interconnection build failed ({e})")


# ---------------------------------------------------------------------------
# Weather (daily → weekly aggregation)
# ---------------------------------------------------------------------------

def _build_weather(con: duckdb.DuckDBPyConnection):
    raw = str(RAW_DIR / "weather" / "**" / "*.parquet")
    out = str(BRONZE_DIR / "weather.parquet")
    print("  [Bronze] Building weather.parquet (weekly aggregation)...")
    try:
        con.execute(f"""
            COPY (
                SELECT
                    DATE_TRUNC('week', CAST(date AS DATE)) AS week_start,
                    TRIM(CAST(subsystem AS VARCHAR))       AS subsystem,
                    SUM(TRY_CAST(precip_mm AS DOUBLE))        AS precip_mm_sum,
                    AVG(TRY_CAST(temp_c AS DOUBLE))           AS temp_c_avg,
                    AVG(TRY_CAST(wind_speed_kmh AS DOUBLE))   AS wind_speed_avg,
                    AVG(TRY_CAST(solar_radiation AS DOUBLE))  AS solar_radiation_avg
                FROM read_parquet('{raw}', hive_partitioning=true)
                WHERE date IS NOT NULL
                GROUP BY 1, 2
                ORDER BY week_start, subsystem
            ) TO '{out}' (FORMAT PARQUET)
        """)
        print(f"    → {_row_count(con, Path(out)):,} rows")
    except Exception as e:
        print(f"    WARNING: weather build failed ({e})")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_bronze():
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    _build_pld(con)
    _build_reservoir(con)
    _build_generation(con)
    _build_load(con)
    _build_interconnection(con)
    _build_weather(con)

    con.close()
    print("  [Bronze] Done.")


if __name__ == "__main__":
    print("Building bronze layer...")
    build_bronze()
