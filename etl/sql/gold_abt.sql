-- Gold layer: Analytical Base Table (ABT).
-- One row per week per subsystem (long format from silver).
-- Features from the silver layer at time t (already lag-shifted, no look-ahead).
-- 4 target columns: actual PLD at t+1w, t+2w, t+3w, t+4w for the same subsystem.
--
-- Parameters:
--   {silver_path}  — data/silver/features.parquet
--   {pld_path}     — data/bronze/pld.parquet (for future targets)

WITH spine AS (
    SELECT *
    FROM read_parquet('{silver_path}')
),

-- Pre-compute origin week for each future horizon
pld_targets AS (
    SELECT
        week_start AS target_week,
        subsystem,
        pld_brl_mwh                                    AS target_pld,
        week_start - INTERVAL '1 WEEK'                 AS origin_1w,
        week_start - INTERVAL '2 WEEKS'                AS origin_2w,
        week_start - INTERVAL '3 WEEKS'                AS origin_3w,
        week_start - INTERVAL '4 WEEKS'                AS origin_4w
    FROM read_parquet('{pld_path}')
)

SELECT
    s.*,

    -- Target columns: future PLD for THIS subsystem (the supervised learning labels)
    t1.target_pld AS pld_t_plus_1w,
    t2.target_pld AS pld_t_plus_2w,
    t3.target_pld AS pld_t_plus_3w,
    t4.target_pld AS pld_t_plus_4w

FROM spine s
LEFT JOIN pld_targets t1 ON t1.origin_1w = s.week_start AND t1.subsystem = s.subsystem
LEFT JOIN pld_targets t2 ON t2.origin_2w = s.week_start AND t2.subsystem = s.subsystem
LEFT JOIN pld_targets t3 ON t3.origin_3w = s.week_start AND t3.subsystem = s.subsystem
LEFT JOIN pld_targets t4 ON t4.origin_4w = s.week_start AND t4.subsystem = s.subsystem

-- Only keep rows where ALL 4 future targets are known
WHERE t4.target_pld IS NOT NULL

ORDER BY s.week_start, s.subsystem
