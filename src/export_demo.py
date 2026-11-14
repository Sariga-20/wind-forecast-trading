"""
Export the small data files the deployed dashboard needs, so the app can run without the full data/ folder.

Usage (from the project root, after `python -m src.models`):
    python -m src.export_demo

Writes app/demo_data/predictions.parquet and app/demo_data/clean_small.parquet (only two columns of the cleaned
table). Commit app/demo_data/ so Streamlit Community Cloud can read it.
"""
from pathlib import Path

import pandas as pd

OUT = Path("app/demo_data")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet("data/processed/predictions.parquet")
    clean = pd.read_parquet("data/processed/wind_clean.parquet")[["mean_wind_speed_100m", "day_ahead_price_eur_mwh"]]
    pred.to_parquet(OUT / "predictions.parquet")
    clean.to_parquet(OUT / "clean_small.parquet")
    for f in sorted(OUT.glob("*.parquet")):
        print(f"{f}  {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
