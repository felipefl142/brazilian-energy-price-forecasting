-- Silver layer: wide-format weekly feature table.
-- One row per week (cross-subsystem). This is the intermediate output
-- that silver.py then unpivots into long format (one row per week per subsystem).
--
-- All window functions use ROWS BETWEEN N PRECEDING AND 1 PRECEDING
-- to ensure NO look-ahead (point-in-time correctness).
--
-- Parameters (replaced before execution):
--   {pld_path}           — data/bronze/pld.parquet
--   {reservoir_path}     — data/bronze/reservoir.parquet
--   {generation_path}    — data/bronze/generation.parquet
--   {load_path}          — data/bronze/load.parquet
--   {interconnection_path} — data/bronze/interconnection.parquet

-- ============================================================
-- STEP 1: Pivot PLD to wide format (all 4 subsystems per row)
-- ============================================================
WITH pld_wide AS (
    SELECT
        week_start,
        MAX(CASE WHEN subsystem = 'SE/CO' THEN pld_brl_mwh END) AS pld_seco,
        MAX(CASE WHEN subsystem = 'S'     THEN pld_brl_mwh END) AS pld_s,
        MAX(CASE WHEN subsystem = 'NE'    THEN pld_brl_mwh END) AS pld_ne,
        MAX(CASE WHEN subsystem = 'N'     THEN pld_brl_mwh END) AS pld_n
    FROM read_parquet('{pld_path}')
    GROUP BY week_start
),

-- ============================================================
-- STEP 2: Pivot reservoir + ENA to wide format
-- ============================================================
reservoir_wide AS (
    SELECT
        week_start,
        MAX(CASE WHEN subsystem = 'SE/CO' THEN reservoir_pct END) AS reservoir_pct_seco,
        MAX(CASE WHEN subsystem = 'S'     THEN reservoir_pct END) AS reservoir_pct_s,
        MAX(CASE WHEN subsystem = 'NE'    THEN reservoir_pct END) AS reservoir_pct_ne,
        MAX(CASE WHEN subsystem = 'N'     THEN reservoir_pct END) AS reservoir_pct_n,
        MAX(CASE WHEN subsystem = 'SE/CO' THEN ena_gwh END)       AS ena_seco,
        MAX(CASE WHEN subsystem = 'S'     THEN ena_gwh END)       AS ena_s,
        MAX(CASE WHEN subsystem = 'NE'    THEN ena_gwh END)       AS ena_ne,
        MAX(CASE WHEN subsystem = 'N'     THEN ena_gwh END)       AS ena_n
    FROM read_parquet('{reservoir_path}')
    GROUP BY week_start
),

-- ============================================================
-- STEP 3: Generation mix (national share per source)
-- ============================================================
generation_mix AS (
    SELECT
        week_start,
        SUM(CASE WHEN source = 'HIDRO'       THEN generation_avg_mw ELSE 0 END)
            / NULLIF(SUM(generation_avg_mw), 0) AS hydro_share,
        SUM(CASE WHEN source = 'EOLICA'      THEN generation_avg_mw ELSE 0 END)
            / NULLIF(SUM(generation_avg_mw), 0) AS wind_share,
        SUM(CASE WHEN source = 'SOLAR'       THEN generation_avg_mw ELSE 0 END)
            / NULLIF(SUM(generation_avg_mw), 0) AS solar_share,
        SUM(CASE WHEN source = 'TERMELETRICA' THEN generation_avg_mw ELSE 0 END)
            / NULLIF(SUM(generation_avg_mw), 0) AS thermal_share,
        SUM(CASE WHEN source = 'NUCLEAR'     THEN generation_avg_mw ELSE 0 END)
            / NULLIF(SUM(generation_avg_mw), 0) AS nuclear_share,
        SUM(generation_avg_mw) AS total_mw
    FROM read_parquet('{generation_path}')
    GROUP BY week_start
),

-- ============================================================
-- STEP 4: Weekly wide join (spine = PLD, left-join everything)
-- ============================================================
weekly_raw AS (
    SELECT
        p.week_start,
        p.pld_seco, p.pld_s, p.pld_ne, p.pld_n,
        r.reservoir_pct_seco, r.reservoir_pct_s, r.reservoir_pct_ne, r.reservoir_pct_n,
        r.ena_seco, r.ena_s, r.ena_ne, r.ena_n,
        g.hydro_share, g.wind_share, g.solar_share, g.thermal_share, g.nuclear_share,
        g.total_mw
    FROM pld_wide p
    LEFT JOIN reservoir_wide r USING (week_start)
    LEFT JOIN generation_mix  g USING (week_start)
),

-- ============================================================
-- STEP 5: Lag and rolling window features (point-in-time)
-- ============================================================
weekly_features AS (
    SELECT
        week_start,

        -- Raw current values (NOT used as features — targets are derived from these)
        pld_seco, pld_s, pld_ne, pld_n,
        reservoir_pct_seco, reservoir_pct_s, reservoir_pct_ne, reservoir_pct_n,
        ena_seco, ena_s, ena_ne, ena_n,
        hydro_share, wind_share, solar_share, thermal_share, nuclear_share,

        -- ---- PLD LAGS (all subsystems) ----
        LAG(pld_seco, 1)  OVER w AS pld_seco_lag_1w,
        LAG(pld_seco, 2)  OVER w AS pld_seco_lag_2w,
        LAG(pld_seco, 4)  OVER w AS pld_seco_lag_4w,
        LAG(pld_seco, 13) OVER w AS pld_seco_lag_13w,
        LAG(pld_seco, 52) OVER w AS pld_seco_lag_52w,
        LAG(pld_s, 1)     OVER w AS pld_s_lag_1w,
        LAG(pld_s, 2)     OVER w AS pld_s_lag_2w,
        LAG(pld_s, 4)     OVER w AS pld_s_lag_4w,
        LAG(pld_s, 13)    OVER w AS pld_s_lag_13w,
        LAG(pld_s, 52)    OVER w AS pld_s_lag_52w,
        LAG(pld_ne, 1)    OVER w AS pld_ne_lag_1w,
        LAG(pld_ne, 2)    OVER w AS pld_ne_lag_2w,
        LAG(pld_ne, 4)    OVER w AS pld_ne_lag_4w,
        LAG(pld_ne, 13)   OVER w AS pld_ne_lag_13w,
        LAG(pld_ne, 52)   OVER w AS pld_ne_lag_52w,
        LAG(pld_n, 1)     OVER w AS pld_n_lag_1w,
        LAG(pld_n, 2)     OVER w AS pld_n_lag_2w,
        LAG(pld_n, 4)     OVER w AS pld_n_lag_4w,
        LAG(pld_n, 13)    OVER w AS pld_n_lag_13w,
        LAG(pld_n, 52)    OVER w AS pld_n_lag_52w,

        -- PLD rolling means (cross-subsystem spread indicator)
        AVG(pld_seco) OVER (ORDER BY week_start ROWS BETWEEN 4  PRECEDING AND 1 PRECEDING) AS pld_seco_roll_4w,
        AVG(pld_seco) OVER (ORDER BY week_start ROWS BETWEEN 13 PRECEDING AND 1 PRECEDING) AS pld_seco_roll_13w,
        AVG(pld_s)    OVER (ORDER BY week_start ROWS BETWEEN 4  PRECEDING AND 1 PRECEDING) AS pld_s_roll_4w,
        AVG(pld_s)    OVER (ORDER BY week_start ROWS BETWEEN 13 PRECEDING AND 1 PRECEDING) AS pld_s_roll_13w,
        AVG(pld_ne)   OVER (ORDER BY week_start ROWS BETWEEN 4  PRECEDING AND 1 PRECEDING) AS pld_ne_roll_4w,
        AVG(pld_ne)   OVER (ORDER BY week_start ROWS BETWEEN 13 PRECEDING AND 1 PRECEDING) AS pld_ne_roll_13w,
        AVG(pld_n)    OVER (ORDER BY week_start ROWS BETWEEN 4  PRECEDING AND 1 PRECEDING) AS pld_n_roll_4w,
        AVG(pld_n)    OVER (ORDER BY week_start ROWS BETWEEN 13 PRECEDING AND 1 PRECEDING) AS pld_n_roll_13w,

        -- ---- RESERVOIR LAGS (all subsystems) ----
        LAG(reservoir_pct_seco, 1) OVER w AS reservoir_seco_lag_1w,
        LAG(reservoir_pct_seco, 2) OVER w AS reservoir_seco_lag_2w,
        LAG(reservoir_pct_seco, 4) OVER w AS reservoir_seco_lag_4w,
        LAG(reservoir_pct_s, 1)    OVER w AS reservoir_s_lag_1w,
        LAG(reservoir_pct_s, 2)    OVER w AS reservoir_s_lag_2w,
        LAG(reservoir_pct_s, 4)    OVER w AS reservoir_s_lag_4w,
        LAG(reservoir_pct_ne, 1)   OVER w AS reservoir_ne_lag_1w,
        LAG(reservoir_pct_ne, 2)   OVER w AS reservoir_ne_lag_2w,
        LAG(reservoir_pct_ne, 4)   OVER w AS reservoir_ne_lag_4w,
        LAG(reservoir_pct_n, 1)    OVER w AS reservoir_n_lag_1w,
        LAG(reservoir_pct_n, 2)    OVER w AS reservoir_n_lag_2w,
        LAG(reservoir_pct_n, 4)    OVER w AS reservoir_n_lag_4w,

        -- Reservoir rolling (trend indicator)
        AVG(reservoir_pct_seco) OVER (ORDER BY week_start ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS reservoir_seco_roll_4w,
        AVG(reservoir_pct_s)    OVER (ORDER BY week_start ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS reservoir_s_roll_4w,
        AVG(reservoir_pct_ne)   OVER (ORDER BY week_start ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS reservoir_ne_roll_4w,
        AVG(reservoir_pct_n)    OVER (ORDER BY week_start ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS reservoir_n_roll_4w,

        -- ---- ENA LAGS (all subsystems) ----
        LAG(ena_seco, 1) OVER w AS ena_seco_lag_1w,
        LAG(ena_seco, 4) OVER w AS ena_seco_lag_4w,
        AVG(ena_seco) OVER (ORDER BY week_start ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS ena_seco_roll_4w,
        LAG(ena_s, 1)    OVER w AS ena_s_lag_1w,
        LAG(ena_s, 4)    OVER w AS ena_s_lag_4w,
        AVG(ena_s)    OVER (ORDER BY week_start ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS ena_s_roll_4w,
        LAG(ena_ne, 1)   OVER w AS ena_ne_lag_1w,
        LAG(ena_ne, 4)   OVER w AS ena_ne_lag_4w,
        AVG(ena_ne)   OVER (ORDER BY week_start ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS ena_ne_roll_4w,
        LAG(ena_n, 1)    OVER w AS ena_n_lag_1w,
        LAG(ena_n, 4)    OVER w AS ena_n_lag_4w,
        AVG(ena_n)    OVER (ORDER BY week_start ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS ena_n_roll_4w,

        -- ---- GENERATION MIX LAGS ----
        LAG(hydro_share,   1) OVER w AS hydro_share_lag_1w,
        LAG(thermal_share, 1) OVER w AS thermal_share_lag_1w,
        LAG(wind_share,    1) OVER w AS wind_share_lag_1w,
        LAG(solar_share,   1) OVER w AS solar_share_lag_1w,

        -- ---- CALENDAR FEATURES ----
        EXTRACT(WEEK  FROM week_start) AS week_of_year,
        EXTRACT(MONTH FROM week_start) AS month,
        EXTRACT(YEAR  FROM week_start) AS year,
        SIN(2 * PI() * EXTRACT(WEEK  FROM week_start) / 52) AS week_sin,
        COS(2 * PI() * EXTRACT(WEEK  FROM week_start) / 52) AS week_cos,
        SIN(2 * PI() * EXTRACT(MONTH FROM week_start) / 12) AS month_sin,
        COS(2 * PI() * EXTRACT(MONTH FROM week_start) / 12) AS month_cos,
        -- Dry season: May–October (historically higher PLD due to lower reservoir inflows)
        CASE WHEN EXTRACT(MONTH FROM week_start) BETWEEN 5 AND 10 THEN 1 ELSE 0 END AS is_dry_season

    FROM weekly_raw
    WINDOW w AS (ORDER BY week_start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)

SELECT *
FROM weekly_features
-- Require at least 52 weeks of history (1 year) for the longest lag to be populated
WHERE pld_seco_lag_52w IS NOT NULL
ORDER BY week_start
