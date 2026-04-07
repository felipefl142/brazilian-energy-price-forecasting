"""Shared helpers for the Streamlit dashboard."""

from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

BRONZE_PLD = str(DATA_DIR / "bronze" / "pld.parquet")
BRONZE_RESERVOIR = str(DATA_DIR / "bronze" / "reservoir.parquet")
BRONZE_GENERATION = str(DATA_DIR / "bronze" / "generation.parquet")
BRONZE_LOAD = str(DATA_DIR / "bronze" / "load.parquet")
BRONZE_INTERCONNECTION = str(DATA_DIR / "bronze" / "interconnection.parquet")
BRONZE_WEATHER = str(DATA_DIR / "bronze" / "weather.parquet")
SILVER_FEATURES = str(DATA_DIR / "silver" / "features.parquet")
GOLD_ABT = str(DATA_DIR / "gold" / "abt.parquet")
GOLD_TRAIN = str(DATA_DIR / "gold" / "abt_train.parquet")
GOLD_TEST = str(DATA_DIR / "gold" / "abt_test.parquet")
FEAST_FEATURES = str(DATA_DIR / "feast" / "features.parquet")
MODELS_DIR = str(BASE_DIR / "models")

SUBSYSTEMS = ["SE/CO", "S", "NE", "N"]
SUBSYSTEM_COLORS = {
    "SE/CO": "#636EFA",
    "S":     "#EF553B",
    "NE":    "#00CC96",
    "N":     "#AB63FA",
}

AVAILABLE_TABLES = {
    "Bronze — PLD":          f"read_parquet('{BRONZE_PLD}')",
    "Bronze — Reservoir":    f"read_parquet('{BRONZE_RESERVOIR}')",
    "Bronze — Generation":   f"read_parquet('{BRONZE_GENERATION}')",
    "Bronze — Load":         f"read_parquet('{BRONZE_LOAD}')",
    "Bronze — Interconnection": f"read_parquet('{BRONZE_INTERCONNECTION}')",
    "Bronze — Weather":      f"read_parquet('{BRONZE_WEATHER}')",
    "Silver — Features":     f"read_parquet('{SILVER_FEATURES}')",
    "Gold — ABT (full)":     f"read_parquet('{GOLD_ABT}')",
    "Gold — ABT (train)":    f"read_parquet('{GOLD_TRAIN}')",
    "Gold — ABT (test)":     f"read_parquet('{GOLD_TEST}')",
    "Feast — Features":      f"read_parquet('{FEAST_FEATURES}')",
}

EXAMPLE_QUERIES = {
    "PLD by year by subsystem": f"""
SELECT
    EXTRACT(YEAR FROM week_start) AS year,
    subsystem,
    MIN(pld_brl_mwh)  AS min_pld,
    AVG(pld_brl_mwh)  AS avg_pld,
    MAX(pld_brl_mwh)  AS max_pld
FROM read_parquet('{BRONZE_PLD}')
GROUP BY 1, 2
ORDER BY 1, 2
""".strip(),

    "PLD spread (max - min across subsystems)": f"""
SELECT
    week_start,
    MAX(pld_brl_mwh) - MIN(pld_brl_mwh) AS pld_spread_brl_mwh
FROM read_parquet('{BRONZE_PLD}')
GROUP BY week_start
ORDER BY week_start
""".strip(),

    "Reservoir vs PLD correlation by subsystem": f"""
SELECT
    r.subsystem,
    CORR(r.reservoir_pct, p.pld_brl_mwh) AS reservoir_pld_corr
FROM read_parquet('{BRONZE_RESERVOIR}') r
JOIN read_parquet('{BRONZE_PLD}') p
    ON r.week_start = p.week_start AND r.subsystem = p.subsystem
GROUP BY r.subsystem
ORDER BY r.subsystem
""".strip(),

    "Dry season price premium (SE/CO)": f"""
SELECT
    CASE WHEN EXTRACT(MONTH FROM week_start) BETWEEN 5 AND 10 THEN 'Dry' ELSE 'Wet' END AS season,
    AVG(pld_brl_mwh) AS avg_pld,
    STDDEV(pld_brl_mwh) AS std_pld,
    COUNT(*) AS n_weeks
FROM read_parquet('{BRONZE_PLD}')
WHERE subsystem = 'SE/CO'
GROUP BY 1
""".strip(),

    "Generation mix latest week": f"""
SELECT source, generation_avg_mw,
    ROUND(100.0 * generation_avg_mw / SUM(generation_avg_mw) OVER (), 1) AS share_pct
FROM read_parquet('{BRONZE_GENERATION}')
WHERE week_start = (SELECT MAX(week_start) FROM read_parquet('{BRONZE_GENERATION}'))
ORDER BY generation_avg_mw DESC
""".strip(),

    "ABT row counts by split": f"""
SELECT 'train' AS split, COUNT(*) AS rows, COUNT(*)/4 AS weeks FROM read_parquet('{GOLD_TRAIN}')
UNION ALL
SELECT 'test',            COUNT(*),         COUNT(*)/4           FROM read_parquet('{GOLD_TEST}')
""".strip(),

    "Silver feature schema": f"""
DESCRIBE SELECT * FROM read_parquet('{SILVER_FEATURES}')
""".strip(),
}


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()
