"""
Feature engineering for day-ahead wind forecasting.

Forecast set-up (state this in the README):
  The forecast for day D is made around 12:00 UTC on day D-1, so for the target hours
  (00:00-23:59 of D) the most recent known actual generation is 12-35 hours old.
  To avoid look-ahead leakage, every feature built from past generation is lagged by
  at least 36 hours.

Weather features come from Open-Meteo's historical archive (reanalysis). In real trading
you would only have weather *forecasts*, so results here are optimistic.
"""
import numpy as np
import pandas as pd

TARGET = "wind_actual_mw"
LAGS = [36, 48, 72, 168]
MIN_LAG = 36  # hours between "last known actual" and the target hour

# Regional groups (names must match LOCATIONS in src/ingest.py; missing names are skipped)
GROUPS = {
    "offshore": ["north_sea", "offshore_borkum", "offshore_helgoland", "offshore_baltic"],
    "north": ["hamburg", "schleswig_holstein", "lower_saxony", "lower_saxony_coast",
              "emsland", "mecklenburg", "brandenburg"],
    "south": ["saxony_anhalt", "nrw", "thuringia_hesse", "rhineland_palatinate", "bavaria_bw"],
}


def _locations(df: pd.DataFrame) -> list:
    return sorted(c[: -len("_wind_speed_100m")] for c in df.columns if c.endswith("_wind_speed_100m"))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df = data/processed/wind_clean.parquet (complete hourly UTC index)."""
    out = pd.DataFrame(index=df.index)
    locs = _locations(df)

    # --- calendar features in local time
    local = df.index.tz_convert("Europe/Berlin")
    out["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * local.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * local.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * local.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * local.month / 12)
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)

    # --- weather: speed per location, regional means, overall mean and cube
    for loc in locs:
        out[f"{loc}_speed"] = df[f"{loc}_wind_speed_100m"]
    for group, members in GROUPS.items():
        cols = [f"{m}_wind_speed_100m" for m in members if f"{m}_wind_speed_100m" in df.columns]
        if cols:
            out[f"{group}_speed"] = df[cols].mean(axis=1)
            out[f"{group}_speed_cubed"] = out[f"{group}_speed"] ** 3

    out["mean_speed"] = df["mean_wind_speed_100m"]
    out["mean_speed_cubed"] = out["mean_speed"] ** 3  # power is roughly proportional to speed^3

    # mean wind direction as a vector (an average of raw angles would be wrong at 0/360)
    dir_cols = [c for c in df.columns if c.endswith("_wind_direction_100m")]
    rad = np.deg2rad(df[dir_cols])
    out["mean_dir_sin"] = np.sin(rad).mean(axis=1)
    out["mean_dir_cos"] = np.cos(rad).mean(axis=1)

    temp_cols = [c for c in df.columns if c.endswith("_temperature_2m")]
    out["mean_temp"] = df[temp_cols].mean(axis=1)

    # --- lagged generation (all lags >= MIN_LAG so nothing from the future leaks in)
    y = df[TARGET]
    for lag in LAGS:
        assert lag >= MIN_LAG
        out[f"lag_{lag}h"] = y.shift(lag)
    known = y.shift(MIN_LAG)
    out["roll_mean_24h"] = known.rolling(24).mean()
    out["roll_mean_168h"] = known.rolling(168).mean()
    out["roll_std_24h"] = known.rolling(24).std()

    # --- capacity proxy: recent peak output (installed capacity grows over the years)
    out["cap_proxy_30d"] = known.rolling(24 * 30, min_periods=24 * 7).max()
    out["cap_proxy_90d"] = known.rolling(24 * 90, min_periods=24 * 14).max()

    return out


def make_dataset(df: pd.DataFrame):
    """Return X (features), y (target) and the ENTSO-E forecast, rows with NaN features removed."""
    X = build_features(df)
    data = X.join(df[[TARGET, "wind_entsoe_forecast_mw"]]).dropna(subset=list(X.columns) + [TARGET])
    X = data[list(X.columns)]
    return X, data[TARGET], data["wind_entsoe_forecast_mw"]
