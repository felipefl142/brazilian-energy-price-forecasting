"""
Bronze layer: schema normalization, type casting, subsystem mapping, and weekly aggregation.
Reads hive-partitioned raw Parquet (as downloaded by collect.py) and writes clean weekly files.

Output files (data/bronze/):
  pld.parquet              — weekly PLD per subsystem (R$/MWh)
  reservoir.parquet        — weekly reservoir storage (% + MWmonth absolute) + ENA (GWh) per subsystem
  generation.parquet       — weekly generation by fuel type, national (avg MW)
  load.parquet             — weekly load per subsystem (MWh)
  interconnection.parquet  — daily interchange flows per subsystem pair (avg MWmed)
  weather.parquet          — weekly weather per subsystem city (precip mm, temp °C)
  thermal_dispatch.parquet — weekly national thermal dispatch stress signals (emergency MWmed, non-merit share)

Raw column names (verified April 2026):
  EAR:           id_subsistema, ear_data, ear_verif_subsistema_percentual, ear_verif_subsistema_mwmes
  ENA:           id_subsistema, ena_data, ena_armazenavel_regiao_mwmed
  Load:          id_subsistema, din_instante, val_cargaenergiamwmed
  Generation:    din_instante, nom_tipocombustivel, val_geracao (string MWmed)
  Interconnect:  din_instante, id_subsistema_origem, id_subsistema_destino, val_intercambiomwmed

Subsystem ID mapping (ONS → project standard):
  SE → SE/CO,  S → S,  NE → NE,  N → N

Usage:
    python -m etl.bronze
"""

from pathlib import Path

import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
BRONZE_DIR = BASE_DIR / "data" / "bronze"

# DuckDB CASE expression to map ONS id_subsistema → project standard name
# Applied to any column that holds the ONS single-letter codes (SE, S, NE, N)
_SUBSYS_MAP = """
    CASE {col}
        WHEN 'SE' THEN 'SE/CO'
        WHEN 'S'  THEN 'S'
        WHEN 'NE' THEN 'NE'
        WHEN 'N'  THEN 'N'
        ELSE TRIM(CAST({col} AS VARCHAR))
    END
"""

# Standard generation fuel type mapping: ONS Portuguese → project codes
_FUEL_MAP = """
    CASE TRIM(nom_tipocombustivel)
        WHEN 'Hidráulica'      THEN 'HIDRO'
        WHEN 'Eólica'          THEN 'EOLICA'
        WHEN 'Solar'           THEN 'SOLAR'
        WHEN 'Nuclear'         THEN 'NUCLEAR'
        WHEN 'Gás'             THEN 'TERMELETRICA'
        WHEN 'Óleo Diesel'     THEN 'TERMELETRICA'
        WHEN 'Óleo Combustível' THEN 'TERMELETRICA'
        WHEN 'Carvão'          THEN 'TERMELETRICA'
        WHEN 'Biomassa'        THEN 'TERMELETRICA'
        ELSE NULL
    END
"""

# PLD uses Friday as week_start (confirmed from raw data: all week_start dates are Fridays).
# Aggregate daily observations into the same Friday-starting weeks so JOIN keys align.
#
# Formula: date - ((DOW + 2) % 7) days, where DOW is DuckDB's 0=Sun … 6=Sat convention.
#   Friday (5): (5+2)%7 = 0  → 0 days back = Friday ✓
#   Saturday(6): (6+2)%7 = 1  → 1 day  back = Friday ✓
#   Sunday (0): (0+2)%7 = 2  → 2 days back = Friday ✓
#   Monday (1): (1+2)%7 = 3  → 3 days back = Friday ✓
#   Tuesday(2): (2+2)%7 = 4  → 4 days back = Friday ✓
#   Wednesday(3): (3+2)%7 = 5  → 5 days back = Friday ✓
#   Thursday(4): (4+2)%7 = 6  → 6 days back = Friday ✓  (assigns Thu to prior week's Friday)
def _friday_week(date_col: str) -> str:
    """Return the Friday that starts the ISO-like week containing date_col."""
    return (
        f"(TRY_CAST({date_col} AS DATE)"
        f" - CAST(((EXTRACT(DOW FROM TRY_CAST({date_col} AS DATE))::INT + 2) % 7) AS INT)"
        f" * INTERVAL '1 day')"
    )


def _row_count(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    return con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]


def _raw_glob(source: str) -> str:
    return str(RAW_DIR / source / "**" / "*.parquet")


# ---------------------------------------------------------------------------
# PLD
# ---------------------------------------------------------------------------

def _build_pld(con: duckdb.DuckDBPyConnection):
    raw = _raw_glob("pld")
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
                PARTITION BY TRY_CAST(week_start AS DATE), TRIM(CAST(subsystem AS VARCHAR))
                ORDER BY TRY_CAST(week_start AS DATE)
            ) = 1
            ORDER BY week_start, subsystem
        ) TO '{out}' (FORMAT PARQUET)
    """)
    print(f"    → {_row_count(con, Path(out)):,} rows")


# ---------------------------------------------------------------------------
# Reservoir (EAR) + ENA → reservoir.parquet
# ---------------------------------------------------------------------------

def _build_reservoir(con: duckdb.DuckDBPyConnection):
    raw_ear = _raw_glob("reservoir")
    raw_ena = _raw_glob("ena")
    out = str(BRONZE_DIR / "reservoir.parquet")
    print("  [Bronze] Building reservoir.parquet (EAR + ENA)...")

    # EAR: daily → weekly average of storage %
    # Columns: id_subsistema, ear_data (date string), ear_verif_subsistema_percentual
    ear_ok = (RAW_DIR / "reservoir").exists()
    ena_ok = (RAW_DIR / "ena").exists()

    if not ear_ok and not ena_ok:
        print("    → No reservoir/ENA data found, skipping.")
        return

    subsys_map = _SUBSYS_MAP.replace("{col}", "id_subsistema")

    if ear_ok:
        try:
            df_ear = con.execute(f"""
                SELECT
                    {_friday_week('ear_data')} AS week_start,
                    {subsys_map} AS subsystem,
                    AVG(TRY_CAST(ear_verif_subsistema_percentual AS DOUBLE)) AS reservoir_pct,
                    AVG(TRY_CAST(ear_verif_subsistema_mwmes AS DOUBLE))      AS reservoir_mwmonth
                FROM read_parquet('{raw_ear}', hive_partitioning=true)
                WHERE ear_data IS NOT NULL
                  AND ear_verif_subsistema_percentual IS NOT NULL
                GROUP BY 1, 2
                ORDER BY 1, 2
            """).fetchdf()
        except Exception as e:
            print(f"    WARNING: EAR build failed ({e})")
            df_ear = pd.DataFrame(columns=["week_start", "subsystem", "reservoir_pct"])
    else:
        df_ear = pd.DataFrame(columns=["week_start", "subsystem", "reservoir_pct"])

    # ENA: daily → weekly sum of inflows
    # ena_armazenavel_regiao_mwmed (daily avg MW) × 24h × 7 days / 1000 = GWh/week
    # Stored as GWh (weekly total); the anomaly ratio in silver normalises the unit.
    if ena_ok:
        try:
            df_ena = con.execute(f"""
                SELECT
                    {_friday_week('ena_data')} AS week_start,
                    {subsys_map} AS subsystem,
                    SUM(TRY_CAST(ena_armazenavel_regiao_mwmed AS DOUBLE)) * 24.0 / 1000.0 AS ena_gwh
                FROM read_parquet('{raw_ena}', hive_partitioning=true)
                WHERE ena_data IS NOT NULL
                  AND ena_armazenavel_regiao_mwmed IS NOT NULL
                GROUP BY 1, 2
                ORDER BY 1, 2
            """).fetchdf()
        except Exception as e:
            print(f"    WARNING: ENA build failed ({e})")
            df_ena = pd.DataFrame(columns=["week_start", "subsystem", "ena_gwh"])
    else:
        df_ena = pd.DataFrame(columns=["week_start", "subsystem", "ena_gwh"])

    if df_ear.empty and df_ena.empty:
        print("    → Both EAR and ENA are empty, skipping.")
        return

    df = pd.merge(df_ear, df_ena, on=["week_start", "subsystem"], how="outer")
    df = df.sort_values(["week_start", "subsystem"]).reset_index(drop=True)
    df.to_parquet(out, index=False)
    print(f"    → {len(df):,} rows")


# ---------------------------------------------------------------------------
# Generation by fuel type (national weekly avg MW)
# ---------------------------------------------------------------------------

def _build_generation(con: duckdb.DuckDBPyConnection):
    raw = _raw_glob("generation")
    out = str(BRONZE_DIR / "generation.parquet")
    print("  [Bronze] Building generation.parquet...")

    if not (RAW_DIR / "generation").exists():
        print("    → No generation data found, skipping.")
        return

    # val_geracao is stored as string (e.g. "2234.70000000") — use TRY_CAST
    # Aggregate: SUM of plant-level daily MWmed → weekly avg MW per fuel type nationally
    # (sum all plants in all subsystems, then average over days in the week)
    try:
        con.execute(f"""
            COPY (
                SELECT
                    {_friday_week('din_instante')} AS week_start,
                    {_FUEL_MAP} AS source,
                    AVG(TRY_CAST(val_geracao AS DOUBLE))               AS generation_avg_mw
                FROM read_parquet('{raw}', hive_partitioning=true)
                WHERE din_instante IS NOT NULL
                  AND ({_FUEL_MAP}) IS NOT NULL
                GROUP BY 1, 2
                ORDER BY 1, 2
            ) TO '{out}' (FORMAT PARQUET)
        """)
        print(f"    → {_row_count(con, Path(out)):,} rows")
    except Exception as e:
        print(f"    WARNING: generation build failed ({e})")


# ---------------------------------------------------------------------------
# Load by subsystem (weekly MWh)
# ---------------------------------------------------------------------------

def _build_load(con: duckdb.DuckDBPyConnection):
    raw = _raw_glob("load")
    out = str(BRONZE_DIR / "load.parquet")
    print("  [Bronze] Building load.parquet...")

    if not (RAW_DIR / "load").exists():
        print("    → No load data found, skipping.")
        return

    subsys_map = _SUBSYS_MAP.replace("{col}", "id_subsistema")

    # val_cargaenergiamwmed = daily avg MW → weekly MWh = sum(daily MWmed) × 24h
    try:
        con.execute(f"""
            COPY (
                SELECT
                    {_friday_week('din_instante')} AS week_start,
                    {subsys_map} AS subsystem,
                    SUM(TRY_CAST(val_cargaenergiamwmed AS DOUBLE)) * 24.0 AS load_mwh
                FROM read_parquet('{raw}', hive_partitioning=true)
                WHERE din_instante IS NOT NULL
                  AND val_cargaenergiamwmed IS NOT NULL
                GROUP BY 1, 2
                ORDER BY 1, 2
            ) TO '{out}' (FORMAT PARQUET)
        """)
        print(f"    → {_row_count(con, Path(out)):,} rows")
    except Exception as e:
        print(f"    WARNING: load build failed ({e})")


# ---------------------------------------------------------------------------
# Interconnection flows (weekly avg MWmed per subsystem pair)
# ---------------------------------------------------------------------------

def _build_interconnection(con: duckdb.DuckDBPyConnection):
    raw = _raw_glob("interconnection")
    out = str(BRONZE_DIR / "interconnection.parquet")
    print("  [Bronze] Building interconnection.parquet...")

    if not (RAW_DIR / "interconnection").exists():
        print("    → No interconnection data found, skipping.")
        return

    from_map = _SUBSYS_MAP.replace("{col}", "id_subsistema_origem")
    to_map   = _SUBSYS_MAP.replace("{col}", "id_subsistema_destino")

    # val_intercambiomwmed = daily avg MW (positive = flow from origem to destino)
    # Store weekly avg MWmed (preserves sign convention)
    try:
        con.execute(f"""
            COPY (
                SELECT
                    {_friday_week('din_instante')} AS week_start,
                    {from_map} AS from_subsystem,
                    {to_map}   AS to_subsystem,
                    AVG(TRY_CAST(val_intercambiomwmed AS DOUBLE))      AS flow_mwmed
                FROM read_parquet('{raw}', hive_partitioning=true)
                WHERE din_instante IS NOT NULL
                  AND val_intercambiomwmed IS NOT NULL
                GROUP BY 1, 2, 3
                ORDER BY 1, 2, 3
            ) TO '{out}' (FORMAT PARQUET)
        """)
        print(f"    → {_row_count(con, Path(out)):,} rows")
    except Exception as e:
        print(f"    WARNING: interconnection build failed ({e})")


# ---------------------------------------------------------------------------
# Weather (daily → weekly aggregation)
# ---------------------------------------------------------------------------

def _build_weather(con: duckdb.DuckDBPyConnection):
    raw = _raw_glob("weather")
    out = str(BRONZE_DIR / "weather.parquet")
    print("  [Bronze] Building weather.parquet (weekly aggregation)...")

    if not (RAW_DIR / "weather").exists():
        print("    → No weather data found, skipping.")
        return

    try:
        con.execute(f"""
            COPY (
                SELECT
                    {_friday_week('date')} AS week_start,
                    TRIM(CAST(subsystem AS VARCHAR))           AS subsystem,
                    SUM(TRY_CAST(precip_mm AS DOUBLE))        AS precip_mm_sum,
                    AVG(TRY_CAST(temp_c AS DOUBLE))           AS temp_c_avg,
                    AVG(TRY_CAST(wind_speed_kmh AS DOUBLE))   AS wind_speed_avg,
                    AVG(TRY_CAST(solar_radiation AS DOUBLE))  AS solar_radiation_avg
                FROM read_parquet('{raw}', hive_partitioning=true)
                WHERE date IS NOT NULL
                GROUP BY 1, 2
                ORDER BY 1, 2
            ) TO '{out}' (FORMAT PARQUET)
        """)
        print(f"    → {_row_count(con, Path(out)):,} rows")
    except Exception as e:
        print(f"    WARNING: weather build failed ({e})")


# ---------------------------------------------------------------------------
# Thermal dispatch stress signals (weekly national aggregates, 2013+)
# ---------------------------------------------------------------------------

def _build_thermal_dispatch(con: duckdb.DuckDBPyConnection):
    raw = _raw_glob("thermal_dispatch")
    out = str(BRONZE_DIR / "thermal_dispatch.parquet")
    print("  [Bronze] Building thermal_dispatch.parquet...")

    if not (RAW_DIR / "thermal_dispatch").exists():
        print("    → No thermal dispatch data found, skipping.")
        return

    # Plant-patamar level → weekly national aggregates.
    # thermal_emergency_mwmed: sum of GFOM (out-of-merit emergency dispatch) nationally.
    # thermal_nonmerit_share:  (GFOM + inflexibility) / total verified generation —
    #   signals how much thermal is running for non-economic reasons (stress indicator).
    try:
        con.execute(f"""
            COPY (
                SELECT
                    {_friday_week('din_instante')} AS week_start,
                    SUM(COALESCE(TRY_CAST(val_verifgfom AS DOUBLE), 0.0)) AS thermal_emergency_mwmed,
                    (
                        SUM(COALESCE(TRY_CAST(val_verifgfom AS DOUBLE), 0.0))
                        + SUM(COALESCE(TRY_CAST(val_verifinflexibilidade AS DOUBLE), 0.0))
                    ) / NULLIF(SUM(COALESCE(TRY_CAST(val_verifgeracao AS DOUBLE), 0.0)), 0)
                        AS thermal_nonmerit_share
                FROM read_parquet('{raw}', hive_partitioning=true)
                WHERE din_instante IS NOT NULL
                GROUP BY 1
                ORDER BY 1
            ) TO '{out}' (FORMAT PARQUET)
        """)
        print(f"    → {_row_count(con, Path(out)):,} rows")
    except Exception as e:
        print(f"    WARNING: thermal dispatch build failed ({e})")


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
    _build_thermal_dispatch(con)

    con.close()
    print("  [Bronze] Done.")


if __name__ == "__main__":
    print("Building bronze layer...")
    build_bronze()
