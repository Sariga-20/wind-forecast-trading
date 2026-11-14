"""
Pull yearly installed wind capacity from ENTSO-E (onshore + offshore) for 2023-2025.

Usage (from the project root):
    python -m src.capacity                 # default zone DE_LU
    python -m src.capacity --zone DE       # try this if DE_LU returns nothing

Writes data/raw/installed_capacity.parquet with a 'wind_capacity_mw' column (one row per year).
The raw frames are printed so you can check the values look sensible.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

TZ = "Europe/Berlin"
YEARS = [2023, 2024, 2025]
OUT = Path("data/raw/installed_capacity.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="DE_LU")
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("ENTSOE_API_KEY")
    if not token:
        raise SystemExit("Set ENTSOE_API_KEY in your .env file.")
    client = EntsoePandasClient(api_key=token)

    rows = {}
    for year in YEARS:
        start = pd.Timestamp(f"{year}-01-01", tz=TZ)
        end = pd.Timestamp(f"{year + 1}-01-01", tz=TZ)
        try:
            cap = client.query_installed_generation_capacity(args.zone, start=start, end=end, psr_type=None)
        except Exception as exc:  # the API may have no data for a year or zone
            print(f"{year}: failed -> {exc!r}")
            continue
        wind_cols = [c for c in cap.columns if "Wind" in str(c)]
        print(f"\n{year}: columns = {list(cap.columns)}")
        print(cap[wind_cols].to_string() if wind_cols else "no wind columns found")
        if wind_cols:
            rows[start] = float(cap[wind_cols].iloc[0].sum())

    if not rows:
        raise SystemExit("No capacity data returned. Try --zone DE, or tell me the printed errors.")

    out = pd.DataFrame({"wind_capacity_mw": pd.Series(rows)})
    out.index.name = "timestamp"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    print("\nSaved", OUT)
    print(out.to_string())


if __name__ == "__main__":
    main()
