"""
Data ingestion: ONS CMO (weekly marginal cost), ONS grid data, and Open-Meteo weather.
Outputs hive-partitioned Parquet under data/raw/{source}/year=YYYY/.

No API keys required — all sources are public.

Usage:
    python -m etl.collect --start 2005-01-01 --end 2024-12-31
    python -m etl.collect --start 2025-01-01 --end 2025-12-31 --force

Data sources:
    - PLD/CMO: ONS dados.ons.org.br (weekly CMO per subsystem, S3-hosted CSV by year)
    - ONS grid: dados.ons.org.br S3 (EAR, ENA, load, generation, interconnection — Parquet/CSV by year)
    - Weather: Open-Meteo archive API (hourly → aggregated daily, 4 cities per subsystem)

ONS S3 URL patterns (verified April 2026 via package_show API):
    EAR:            ear_subsistema_di/EAR_DIARIO_SUBSISTEMA_{year}.parquet        (2000+)
    ENA:            ena_subsistema_di/ENA_DIARIO_SUBSISTEMA_{year}.{ext}           (parquet 2021+, csv 2000–2020)
    Load:           carga_energia_di/CARGA_ENERGIA_{year}.parquet                 (2000+)
    Generation:     geracao_usina_2_ho/GERACAO_USINA-2_{year}.parquet             (annual 2000–2021)
                    geracao_usina_2_ho/GERACAO_USINA-2_{year}_{mm:02d}.parquet    (monthly 2022+)
    Interconnection: intercambio_nacional_ho/INTERCAMBIO_NACIONAL_{year}.{ext}   (parquet 2023+, csv 2000–2022)
"""

import argparse
import io
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

# ONS CMO S3 template
ONS_CMO_CSV_TEMPLATE = (
    "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/cmo_se/CMO_SEMANAL_{year}.csv"
)
ONS_SUBSYSTEM_MAP = {"SE": "SE/CO", "S": "S", "NE": "NE", "N": "N"}

# ONS Open Data S3 base
ONS_S3 = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _year_range(start: datetime, end: datetime):
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


def _get(url: str) -> requests.Response | None:
    """GET with 60s timeout. Returns None on 404, raises on other HTTP errors."""
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp


def _read_parquet_url(url: str) -> pd.DataFrame | None:
    resp = _get(url)
    if resp is None:
        return None
    return pd.read_parquet(io.BytesIO(resp.content))


def _read_csv_url(url: str, sep: str = ";") -> pd.DataFrame | None:
    resp = _get(url)
    if resp is None:
        return None
    return pd.read_csv(io.StringIO(resp.text), sep=sep)


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
# ONS grid data — direct S3 file downloads per year
# ---------------------------------------------------------------------------

class CollectONS:
    """
    Downloads ONS grid datasets from S3-hosted files (one per year or month).
    All files are on ons-aws-prod-opendata.s3.amazonaws.com — no auth required.

    Raw column names are preserved as-is; bronze.py normalises them.

    Datasets collected:
      reservoir   — EAR daily by subsystem (% useful energy storage)
      ena         — ENA daily by subsystem (natural inflows, MWmed)
      load        — Daily energy load by subsystem (MWmed)
      generation  — Daily generation by plant/fuel (MWmed), annual pre-2022 / monthly 2022+
      interconnection — Daily interchange flows between subsystems (MWmed)
    """

    def _fetch_reservoir(self, year: int) -> pd.DataFrame | None:
        url = f"{ONS_S3}/ear_subsistema_di/EAR_DIARIO_SUBSISTEMA_{year}.parquet"
        return _read_parquet_url(url)

    def _fetch_ena(self, year: int) -> pd.DataFrame | None:
        if year >= 2021:
            url = f"{ONS_S3}/ena_subsistema_di/ENA_DIARIO_SUBSISTEMA_{year}.parquet"
            return _read_parquet_url(url)
        url = f"{ONS_S3}/ena_subsistema_di/ENA_DIARIO_SUBSISTEMA_{year}.csv"
        df = _read_csv_url(url)
        return df

    def _fetch_load(self, year: int) -> pd.DataFrame | None:
        url = f"{ONS_S3}/carga_energia_di/CARGA_ENERGIA_{year}.parquet"
        return _read_parquet_url(url)

    def _fetch_generation(self, year: int) -> pd.DataFrame | None:
        if year >= 2022:
            frames = []
            for month in range(1, 13):
                url = f"{ONS_S3}/geracao_usina_2_ho/GERACAO_USINA-2_{year}_{month:02d}.parquet"
                df = _read_parquet_url(url)
                if df is not None and not df.empty:
                    frames.append(df)
            return pd.concat(frames, ignore_index=True) if frames else None
        url = f"{ONS_S3}/geracao_usina_2_ho/GERACAO_USINA-2_{year}.parquet"
        return _read_parquet_url(url)

    def _fetch_interconnection(self, year: int) -> pd.DataFrame | None:
        if year >= 2023:
            url = f"{ONS_S3}/intercambio_nacional_ho/INTERCAMBIO_NACIONAL_{year}.parquet"
            return _read_parquet_url(url)
        url = f"{ONS_S3}/intercambio_nacional_ho/INTERCAMBIO_NACIONAL_{year}.csv"
        return _read_csv_url(url)

    def _save(self, source_key: str, year: int, df: pd.DataFrame):
        out = _partition_path(source_key, year)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"  [ONS/{source_key}] {year}: {len(df):,} rows → {out}")

    def process(self, start: datetime, end: datetime, force: bool = False):
        sources = {
            "reservoir": self._fetch_reservoir,
            "ena":        self._fetch_ena,
            "load":       self._fetch_load,
            "generation": self._fetch_generation,
            "interconnection": self._fetch_interconnection,
        }
        for source_key, fetch_fn in sources.items():
            print(f"\n[ONS/{source_key}] Starting collection...")
            for year in tqdm(list(_year_range(start, end)), desc=source_key):
                if _already_collected(source_key, year, force):
                    continue
                try:
                    df = fetch_fn(year)
                    if df is None or df.empty:
                        print(f"  [ONS/{source_key}] {year}: no data")
                        continue
                    self._save(source_key, year, df)
                except Exception as e:
                    print(f"  [ONS/{source_key}] {year} ERROR: {e}")


# ---------------------------------------------------------------------------
# Weather (Open-Meteo — 4 cities, one per subsystem)
# ---------------------------------------------------------------------------

WEATHER_VARIABLES = [
    "precipitation_sum",       # daily sum (mm)
    "temperature_2m_mean",     # daily mean (°C)
    "wind_speed_10m_mean",     # daily mean (km/h)
    "shortwave_radiation_sum", # daily sum (MJ/m²)
]

# Open-Meteo free tier: ~600 requests/hour. Use 3s between city requests
# (4 cities × 20 years = 80 requests ≈ 4 min total for a full backfill).
_WEATHER_SLEEP_BETWEEN_CITIES = 3.0


class CollectWeather:
    """
    Fetches daily weather data for 4 representative Brazilian cities (one per subsystem)
    from Open-Meteo archive API. Data is free, no API key needed.
    """

    def __init__(self):
        cache_session = requests_cache.CachedSession(".weather_cache", expire_after=-1)
        # backoff_factor=2 → waits 4s / 8s / 16s / 32s / 64s on retries
        retry_session = retry(cache_session, retries=5, backoff_factor=2)
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
                    time.sleep(_WEATHER_SLEEP_BETWEEN_CITIES)

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
