"""
Pull raw data for the wind forecasting project.

Sources
- ENTSO-E Transparency Platform: actual wind generation, ENTSO-E day-ahead
  wind forecast (benchmark), day-ahead prices (used for the back-test later)
- Open-Meteo Historical API: hourly weather at 16 locations across Germany

Usage (from the project root):
    python -m src.ingest --start 2023-01-01 --end 2025-12-31 --zone DE_LU
    python -m src.ingest --weather-only        # re-pull weather only, keep ENTSO-E files

Outputs hourly parquet files in data/raw/.
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

RAW_DIR = Path("data/raw")
TZ = "Europe/Berlin"

# 16 points: offshore North Sea / Baltic, onshore north, centre and south.
# Names are used by src/features.py to build the regional averages.
LOCATIONS = {
    # offshore
    "north_sea": (54.50, 7.50),
    "offshore_borkum": (54.00, 6.30),
    "offshore_helgoland": (54.30, 7.70),
    "offshore_baltic": (54.80, 13.00),
    # onshore north
    "hamburg": (53.55, 9.99),
    "schleswig_holstein": (54.30, 9.00),
    "lower_saxony": (52.90, 8.00),
    "lower_saxony_coast": (53.50, 8.00),
    "emsland": (52.70, 7.30),
    "mecklenburg": (53.80, 12.00),
    "brandenburg": (52.40, 13.00),
    # onshore centre / south
    "saxony_anhalt": (52.00, 11.50),
    "nrw": (51.50, 7.50),
    "thuringia_hesse": (51.00, 10.00),
    "rhineland_palatinate": (50.00, 7.50),
    "bavaria_bw": (48.50, 10.50),
}


def year_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Split the range into <=1-year chunks to stay within API limits."""
    cur = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(years=1), end)
        yield cur, nxt
        cur = nxt


def to_hourly(obj):
    """ENTSO-E data may be 15-min or hourly; resample to hourly means."""
    return obj.resample("1h").mean()


def pull_entsoe(client, zone, start, end):
    gen_parts, fc_parts, price_parts = [], [], []
    for s, e in year_chunks(start, end):
        print(f"ENTSO-E {zone}: {s.date()} -> {e.date()}")

        # Actual generation per type -> keep wind onshore + offshore
        gen = client.query_generation(zone, start=s, end=e, psr_type=None)
        if isinstance(gen.columns, pd.MultiIndex):
            # keep only "Actual Aggregated" columns, drop consumption columns
            gen = gen.xs("Actual Aggregated", axis=1, level=1, drop_level=True)
        wind_cols = [c for c in gen.columns if "Wind" in c]
        gen_parts.append(to_hourly(gen[wind_cols]))

        # ENTSO-E day-ahead wind forecast (benchmark)
        fc = client.query_wind_and_solar_forecast(zone, start=s, end=e, psr_type=None)
        fc_wind = fc[[c for c in fc.columns if "Wind" in c]]
        fc_parts.append(to_hourly(fc_wind))

        # Day-ahead prices
        price_parts.append(to_hourly(client.query_day_ahead_prices(zone, start=s, end=e)))
        time.sleep(1)  # be polite to the API

    gen = pd.concat(gen_parts)
    fc = pd.concat(fc_parts)
    price = pd.concat(price_parts).rename("day_ahead_price_eur_mwh")

    # Drop overlaps at chunk boundaries
    gen = gen[~gen.index.duplicated(keep="first")]
    fc = fc[~fc.index.duplicated(keep="first")]
    price = price[~price.index.duplicated(keep="first")]
    return gen, fc, price


def fetch_location(lat, lon, start, end, retries=6):
    """One Open-Meteo request, waiting and retrying if the free-tier rate limit (HTTP 429) is hit."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "wind_speed_100m,wind_direction_100m,temperature_2m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    for attempt in range(1, retries + 1):
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=90)
        if r.status_code == 429:
            wait = 30 * attempt
            print(f"  rate limited, waiting {wait}s (attempt {attempt}/{retries})")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["hourly"]
    raise RuntimeError("Open-Meteo kept returning 429; wait a few minutes and rerun.")


def pull_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for i, (name, (lat, lon)) in enumerate(LOCATIONS.items(), start=1):
        print(f"Open-Meteo {i}/{len(LOCATIONS)}: {name}")
        df = pd.DataFrame(fetch_location(lat, lon, start, end))
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").add_prefix(f"{name}_")
        frames.append(df)
        time.sleep(10)  # stay under the free-tier rate limit
    weather = pd.concat(frames, axis=1)
    # Requested in UTC to avoid daylight-saving ambiguity; convert to local time
    weather.index = weather.index.tz_convert(TZ)
    return weather


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--zone", default="DE_LU")
    parser.add_argument("--weather-only", action="store_true",
                        help="Only re-pull weather; keep existing ENTSO-E files")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start, tz=TZ)
    end = pd.Timestamp(args.end, tz=TZ)

    if not args.weather_only:
        load_dotenv()
        token = os.getenv("ENTSOE_API_KEY")
        if not token:
            raise SystemExit("Set ENTSOE_API_KEY in your .env file.")
        client = EntsoePandasClient(api_key=token)
        gen, fc, price = pull_entsoe(client, args.zone, start, end)
        gen.to_parquet(RAW_DIR / "entsoe_wind_actual.parquet")
        fc.to_parquet(RAW_DIR / "entsoe_wind_forecast.parquet")
        price.to_frame().to_parquet(RAW_DIR / "entsoe_day_ahead_price.parquet")
        print(f"ENTSO-E rows: actual={len(gen)} forecast={len(fc)} prices={len(price)}")

    weather = pull_weather(start, end)
    weather.to_parquet(RAW_DIR / "weather.parquet")
    print(f"\nDone. Weather saved: {weather.shape[0]} rows x {weather.shape[1]} columns")


if __name__ == "__main__":
    main()
