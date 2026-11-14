"""
Day 2: merge and clean the raw data into one hourly UTC table.

Usage (from the project root):
    python -m src.clean

Reads  data/raw/*.parquet
Writes data/processed/wind_clean.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")

# Plausibility limits
MAX_WIND_MW = 90_000        # Germany's installed wind capacity is well below this
MAX_WIND_SPEED_MS = 60      # anything above is a data error
PRICE_MIN, PRICE_MAX = -500, 4000   # EPEX day-ahead harmonised price limits (EUR/MWh)
MAX_FILL_HOURS = 3          # only interpolate short gaps


def _to_utc(obj):
    """Convert index to UTC (avoids daylight-saving duplicates/gaps) and drop duplicates."""
    obj = obj.copy()
    obj.index = pd.DatetimeIndex(obj.index).tz_convert("UTC")
    obj.index.name = "timestamp_utc"
    obj = obj.sort_index()
    return obj[~obj.index.duplicated(keep="first")]


def _wind_total(df: pd.DataFrame) -> pd.Series:
    """Sum all wind columns (onshore + offshore) into one series."""
    wind_cols = [c for c in df.columns if "Wind" in str(c)]
    if not wind_cols:
        raise ValueError(f"No wind columns found. Columns are: {list(df.columns)}")
    return df[wind_cols].sum(axis=1, min_count=1)


def load_raw() -> dict:
    actual = _to_utc(pd.read_parquet(RAW / "entsoe_wind_actual.parquet"))
    fc = _to_utc(pd.read_parquet(RAW / "entsoe_wind_forecast.parquet"))
    price = _to_utc(pd.read_parquet(RAW / "entsoe_day_ahead_price.parquet"))
    weather = _to_utc(pd.read_parquet(RAW / "weather.parquet"))
    return {"actual": actual, "fc": fc, "price": price, "weather": weather}


def build_table(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame(index=raw["actual"].index)
    df["wind_actual_mw"] = _wind_total(raw["actual"])
    df["wind_entsoe_forecast_mw"] = _wind_total(raw["fc"]).reindex(df.index)
    df["day_ahead_price_eur_mwh"] = raw["price"].iloc[:, 0].reindex(df.index)

    weather = raw["weather"].reindex(df.index)
    speed_cols = [c for c in weather.columns if c.endswith("wind_speed_100m")]
    df = df.join(weather)
    df["mean_wind_speed_100m"] = weather[speed_cols].mean(axis=1)

    # Make the timeline complete: every hour from first to last timestamp
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC", name="timestamp_utc")
    return df.reindex(full_index)


def clean_table(df: pd.DataFrame):
    df = df.copy()
    report = {"rows": len(df)}

    # 1) Impossible values -> NaN (negative prices are legitimate, so they stay)
    bad_wind = (df["wind_actual_mw"] < 0) | (df["wind_actual_mw"] > MAX_WIND_MW)
    bad_fc = (df["wind_entsoe_forecast_mw"] < 0) | (df["wind_entsoe_forecast_mw"] > MAX_WIND_MW)
    bad_price = (df["day_ahead_price_eur_mwh"] < PRICE_MIN) | (df["day_ahead_price_eur_mwh"] > PRICE_MAX)
    speed_cols = [c for c in df.columns if c.endswith("wind_speed_100m")]
    bad_speed = (df[speed_cols] > MAX_WIND_SPEED_MS) | (df[speed_cols] < 0)

    # 1b) Remove isolated forecast dips on October clock-change weekends
    fc = df["wind_entsoe_forecast_mw"]
    neighbour_mean = (fc.shift(1) + fc.shift(-1)) / 2

    dip = (fc < 0.5 * neighbour_mean) & (neighbour_mean > 2000)

    report["forecast_dips_removed"] = int(dip.sum())
    df.loc[dip, "wind_entsoe_forecast_mw"] = np.nan

    report["outliers_wind_actual"] = int(bad_wind.sum())
    report["outliers_wind_forecast"] = int(bad_fc.sum())
    report["outliers_price"] = int(bad_price.sum())
    report["outliers_wind_speed"] = int(bad_speed.sum().sum())

    df.loc[bad_wind, "wind_actual_mw"] = np.nan
    df.loc[bad_fc, "wind_entsoe_forecast_mw"] = np.nan
    df.loc[bad_price, "day_ahead_price_eur_mwh"] = np.nan
    df[speed_cols] = df[speed_cols].mask(bad_speed)

    # 2) Missing values before filling
    report["missing_before"] = df.isna().sum().loc[lambda s: s > 0].to_dict()

    # 3) Fill only short gaps (<= MAX_FILL_HOURS); flag rows that were filled
    value_cols = list(df.columns)
    was_missing = df[value_cols].isna()
    filled = df[value_cols].interpolate(method="time", limit=MAX_FILL_HOURS, limit_area="inside")
    df[value_cols] = filled
    df["was_imputed"] = (was_missing & df[value_cols].notna()).any(axis=1)

    # 4) What is still missing = long gaps; leave as NaN and report
    report["missing_after"] = df[value_cols].isna().sum().loc[lambda s: s > 0].to_dict()
    report["imputed_rows"] = int(df["was_imputed"].sum())
    return df, report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    for name, obj in raw.items():
        print(f"{name:8s} shape={obj.shape}  {obj.index.min()} -> {obj.index.max()}")

    df = build_table(raw)
    df, report = clean_table(df)

    df = df.dropna(subset=["mean_wind_speed_100m"])
    df.to_parquet(OUT / "wind_clean.parquet")
    print("\nCleaning report")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nSaved {OUT / 'wind_clean.parquet'}  shape={df.shape}")


if __name__ == "__main__":
    main()
