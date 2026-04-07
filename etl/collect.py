"""
Data ingestion: ONS CMO (weekly marginal cost), ONS grid data, and Open-Meteo weather.
Outputs hive-partitioned Parquet under data/raw/{source}/year=YYYY/.

No API keys required — all sources are public.

Usage:
    python -m etl.collect --start 2005-01-01 --end 2024-12-31
    python -m etl.collect --start 2025-01-01 --end 2025-12-31 --force

Data sources:
    - PLD/CMO: ONS dados.ons.org.br (weekly CMO per subsystem, available from 2005)
    - ONS: dados.ons.org.br CKAN REST API (load, generation, reservoir, ENA, interconnection)
    - Weather: Open-Meteo archive API (hourly → aggregated weekly per subsystem city)
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests
import requests_cache
from retry_requests import retry
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

SUBSYSTEMS = ["SE/CO", "S", "NE", "N"]

# Representative city per subsystem (for weather)
SUBSYSTEM_CITIES = {
    "SE/CO": {"lat": -23.55, "lon": -46.63, "name": "São Paulo"},
    "S":     {"lat": -25.43, "lon": -49.27, "name": "Curitiba"},
    "NE":    {"lat": -3.72,  "lon": -38.54, "name": "Fortaleza"},
    "N":     {"lat": -1.45,  "lon": -48.50, "name": "Belém"},
}

# ONS CMO (Custo Marginal de Operação) — weekly, per subsystem
# CSV files hosted on S3, discovered via dados.ons.org.br package 'cmo-semanal'
# Available from 2005 onwards. Column val_cmomediasemanal ≈ PLD (R$/MWh).
ONS_CMO_CSV_TEMPLATE = (
    "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/cmo_se/CMO_SEMANAL_{year}.csv"
)
# Subsystem ID mapping: ONS short code → project standard name
ONS_SUBSYSTEM_MAP = {"SE": "SE/CO", "S": "S", "NE": "NE", "N": "N"}

# ONS CKAN API resource IDs
# Verify at: https://dados.ons.org.br
ONS_BASE = "https://dados.ons.org.br/api/3/action/datastore_search"
ONS_RESOURCES = {
    "reservoir": "RESERVATORIO_NIVEL",        # Reservoir storage level (% useful volume)
    "ena":       "ENA_SEMANA_SUBMERCADO",      # ENA weekly by subsystem (GWh)
    "load":      "CARGA_ENERGIA_SUBMERCADO",   # Load by subsystem (weekly or daily)
    "generation":"GERACAO_FONTE_ONS",          # Generation by source (MW, may need weekly agg)
    "interconnection": "INTERCAMBIO_SEMANAL",  # Weekly interchange flows
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _year_range(start: datetime, end: datetime):
    """Yield year integers from start to end inclusive."""
    for year in range(start.year, end.year + 1):
        yield year


def _partition_path(source: str, year: int) -> Path:
    return RAW_DIR / source / f"year={year}" / f"{source}.parquet"


def _already_collected(source: str, year: int, force: bool) -> bool:
    path = _partition_path(source, year)
    if path.exists() and not force:
        print(f"  [{source}] {year} already collected, skipping.")
        return True
    return False


def _ons_fetch_all(resource_id: str, filters: dict | None = None) -> pd.DataFrame:
    """Paginate through the ONS CKAN API and return all records as a DataFrame."""
    offset = 0
    limit = 5000
    records = []
    while True:
        params = {"resource_id": resource_id, "limit": limit, "offset": offset}
        if filters:
            params.update(filters)
        resp = requests.get(ONS_BASE, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()["result"]["records"]
        if not batch:
            break
        records.extend(batch)
        offset += limit
        if len(batch) < limit:
            break
        time.sleep(0.2)
    return pd.DataFrame(records) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# PLD/CMO (ONS — weekly marginal cost per subsystem)
# ---------------------------------------------------------------------------

class CollectPLD:
    """
    Fetches weekly CMO (Custo Marginal de Operação ≈ PLD) for all 4 subsystems
    from ONS Open Data (dados.ons.org.br), S3-hosted CSV files per year.

    Available from 2005. Column used: val_cmomediasemanal (weekly average R$/MWh).
    Source: https://dados.ons.org.br/dataset/cmo-semanal
    """

    def _fetch_year(self, year: int) -> pd.DataFrame:
        url = ONS_CMO_CSV_TEMPLATE.format(year=year)
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            return pd.DataFrame()
        resp.raise_for_status()
        df = pd.read_csv(url, sep=";")
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "din_instante": "week_start",
            "val_cmomediasemanal": "pld_brl_mwh",
            "id_subsistema": "subsystem_code",
        })
        df["week_start"] = pd.to_datetime(df["week_start"]).dt.date
        df["pld_brl_mwh"] = pd.to_numeric(df["pld_brl_mwh"], errors="coerce")
        df["subsystem"] = df["subsystem_code"].map(ONS_SUBSYSTEM_MAP)
        df = df.dropna(subset=["week_start", "pld_brl_mwh", "subsystem"])
        return df[["week_start", "pld_brl_mwh", "subsystem"]]

    def process(self, start: datetime, end: datetime, force: bool = False):
        print("\n[PLD] Starting collection (ONS CMO)...")
        for year in tqdm(list(_year_range(start, end)), desc="PLD"):
            if _already_collected("pld", year, force):
                continue
            try:
                df = self._fetch_year(year)
                if df.empty:
                    print(f"  [PLD] {year}: no data (CSV not found or empty)")
                    continue
                out = _partition_path("pld", year)
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out, index=False)
                print(f"  [PLD] {year}: {len(df)} rows → {out}")
            except Exception as e:
                print(f"  [PLD] {year} ERROR: {e}")


# ---------------------------------------------------------------------------
# ONS — Reservoir levels + ENA
# ---------------------------------------------------------------------------

class CollectONS:
    """
    Fetches grid data from ONS Open Data portal (dados.ons.org.br).
    Each dataset is fetched in full and split by year for partitioned storage.

    IMPORTANT: Resource IDs in ONS_RESOURCES must be verified against the portal.
    Visit https://dados.ons.org.br to find current resource IDs.
    Column names are inferred from the API response — check the notebook
    01_data_ingestion.ipynb to validate schema after first run.
    """

    def _fetch_and_save(self, source_key: str, resource_id: str,
                        start: datetime, end: datetime, force: bool):
        print(f"\n  [ONS/{source_key}] Fetching full dataset from resource {resource_id}...")
        try:
            df = _ons_fetch_all(resource_id)
        except Exception as e:
            print(f"  [ONS/{source_key}] ERROR fetching: {e}")
            return

        if df.empty:
            print(f"  [ONS/{source_key}] WARNING: empty response")
            return

        # Detect the date column (first column containing 'dat' or 'data' case-insensitively)
        date_col = next(
            (c for c in df.columns if "dat" in c.lower() or "semana" in c.lower()),
            df.columns[0],
        )
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df["_year"] = df[date_col].dt.year

        for year in _year_range(start, end):
            if _already_collected(source_key, year, force):
                continue
            year_df = df[df["_year"] == year].drop(columns=["_year"]).copy()
            if year_df.empty:
                print(f"  [ONS/{source_key}] {year}: no data")
                continue
            out = _partition_path(source_key, year)
            out.parent.mkdir(parents=True, exist_ok=True)
            year_df.to_parquet(out, index=False)
            print(f"  [ONS/{source_key}] {year}: {len(year_df)} rows → {out}")

    def process(self, start: datetime, end: datetime, force: bool = False):
        for source_key, resource_id in ONS_RESOURCES.items():
            self._fetch_and_save(source_key, resource_id, start, end, force)


# ---------------------------------------------------------------------------
# Weather (Open-Meteo — 4 cities, one per subsystem)
# ---------------------------------------------------------------------------

WEATHER_VARIABLES = [
    "precipitation_sum",    # daily sum (mm)
    "temperature_2m_mean",  # daily mean (°C)
    "wind_speed_10m_mean",  # daily mean (km/h)
    "shortwave_radiation_sum",  # daily sum (MJ/m²)
]


class CollectWeather:
    """
    Fetches daily weather data for 4 representative Brazilian cities (one per subsystem)
    from Open-Meteo archive API. Data is free, no API key needed.
    """

    def __init__(self):
        cache_session = requests_cache.CachedSession(".weather_cache", expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)

    def _fetch_city(self, subsystem: str, lat: float, lon: float,
                    start_date: str, end_date: str) -> pd.DataFrame:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": WEATHER_VARIABLES,
            "timezone": "America/Sao_Paulo",
        }
        responses = self.client.weather_api(url, params=params)
        daily = responses[0].Daily()

        dates = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
        data = {"date": dates}
        for i, var in enumerate(WEATHER_VARIABLES):
            data[var] = daily.Variables(i).ValuesAsNumpy()

        df = pd.DataFrame(data)
        df["subsystem"] = subsystem
        df = df.rename(columns={
            "precipitation_sum": "precip_mm",
            "temperature_2m_mean": "temp_c",
            "wind_speed_10m_mean": "wind_speed_kmh",
            "shortwave_radiation_sum": "solar_radiation",
        })
        return df

    def process(self, start: datetime, end: datetime, force: bool = False):
        print("\n[Weather] Starting collection...")
        for year in tqdm(list(_year_range(start, end)), desc="Weather"):
            if _already_collected("weather", year, force):
                continue
            try:
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
                frames = []
                for subsystem, city in SUBSYSTEM_CITIES.items():
                    df = self._fetch_city(
                        subsystem, city["lat"], city["lon"], start_date, end_date
                    )
                    frames.append(df)
                    time.sleep(0.2)

                combined = pd.concat(frames, ignore_index=True)
                out = _partition_path("weather", year)
                out.parent.mkdir(parents=True, exist_ok=True)
                combined.to_parquet(out, index=False)
                print(f"  [Weather] {year}: {len(combined)} rows → {out}")
            except Exception as e:
                print(f"  [Weather] {year} ERROR: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect Brazilian energy data: CCEE PLD + ONS grid + Open-Meteo weather"
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--force", "-f", action="store_true", help="Re-download existing partitions")
    parser.add_argument("--only", choices=["pld", "ons", "weather"],
                        help="Collect only one source (default: all)")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print("=" * 60)
    print(f"Brazilian Energy — Data Collection: {args.start} → {args.end}")
    print("=" * 60)

    if not args.only or args.only == "pld":
        CollectPLD().process(start, end, force=args.force)
    if not args.only or args.only == "ons":
        CollectONS().process(start, end, force=args.force)
    if not args.only or args.only == "weather":
        CollectWeather().process(start, end, force=args.force)

    print("\nCollection complete.")


if __name__ == "__main__":
    main()
