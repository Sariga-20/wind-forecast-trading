"""
Data validation checks for the cleaned dataset.
Run from the project root:  pytest -q
"""
from pathlib import Path

import pandas as pd
import pytest

PATH = Path("data/processed/wind_clean.parquet")


@pytest.fixture(scope="module")
def df():
    if not PATH.exists():
        pytest.skip("Run `python -m src.clean` first.")
    return pd.read_parquet(PATH)


def test_index_is_utc_and_sorted(df):
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing


def test_no_duplicate_timestamps(df):
    assert not df.index.duplicated().any()


def test_no_missing_hours(df):
    expected = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC")
    assert len(expected) == len(df)


def test_wind_generation_is_physical(df):
    assert (df["wind_actual_mw"].dropna() >= 0).all()
    assert (df["wind_actual_mw"].dropna() < 90_000).all()


def test_wind_speed_is_physical(df):
    speeds = [c for c in df.columns if c.endswith("wind_speed_100m")]
    assert (df[speeds].min() >= 0).all()
    assert (df[speeds].max() < 60).all()


def test_target_missing_share_is_small(df):
    assert df["wind_actual_mw"].isna().mean() < 0.01


def test_prices_within_exchange_limits(df):
    p = df["day_ahead_price_eur_mwh"].dropna()
    assert p.between(-500, 4000).all()
