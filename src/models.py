"""
Walk-forward comparison of models vs baselines and ENTSO-E.

Usage (from the project root):
    python -m src.models

Each month of 2025 is predicted by models trained only on data that ends 2 days
before that month starts (expanding window). No random splits, no shuffling.
All models see exactly the same features and the same test hours.

If data/raw/installed_capacity.parquet exists (made by `python -m src.capacity`), two extra
"capacity factor" variants are trained: they predict generation / installed capacity and
multiply back, to test whether growth in installed capacity explains the under-forecasting.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.features import make_dataset

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
GAP = pd.Timedelta(days=2)
OUT_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports")
CF_BASES = ["lightgbm", "xgboost"]


def score(y, pred) -> dict:
    err = pd.Series(pred, index=y.index) - y
    return {
        "MAE_MW": err.abs().mean(),
        "RMSE_MW": float(np.sqrt((err ** 2).mean())),
        "wMAPE_%": err.abs().sum() / y.sum() * 100,
        "bias_MW": err.mean(),  # forecast minus actual; negative = under-forecasting
    }


def model_factories() -> dict:
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=42),
        "lightgbm": lambda: LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1),
        "xgboost": lambda: XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
    }


def load_capacity(index):
    """Installed wind capacity (MW) per hour, carrying the latest known yearly value forward."""
    path = RAW_DIR / "installed_capacity.parquet"
    if not path.exists():
        return None
    cap = pd.read_parquet(path)["wind_capacity_mw"]
    cap.index = pd.DatetimeIndex(cap.index).tz_convert("UTC")
    hourly = cap.reindex(cap.index.union(index)).sort_index().ffill().bfill()
    return hourly.reindex(index)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(OUT_DIR / "wind_clean.parquet")
    X, y, entsoe = make_dataset(df)
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features, {X.index.min()} -> {X.index.max()}")

    cap = load_capacity(X.index)
    if cap is None:
        print("No installed_capacity.parquet found: skipping capacity-factor variants.")
    else:
        print(f"Installed capacity (MW) used: min={cap.min():,.0f}, max={cap.max():,.0f}")

    factories = model_factories()
    months = pd.date_range(TEST_START, X.index.max(), freq="MS", tz="UTC")
    preds, last_xgb, fit_seconds = [], None, {name: 0.0 for name in factories}

    for start in months:
        end = start + pd.offsets.MonthBegin(1)
        train = X.index < (start - GAP)
        test = (X.index >= start) & (X.index < end)
        if test.sum() == 0:
            continue
        fold = pd.DataFrame({"actual": y[test]}, index=X.index[test])
        for name, make in factories.items():
            t0 = time.time()
            model = make().fit(X[train], y[train])
            fold[name] = model.predict(X[test])
            fit_seconds[name] += time.time() - t0
            if name == "xgboost":
                last_xgb = model
        if cap is not None:
            for base in CF_BASES:
                m = factories[base]().fit(X[train], y[train] / cap[train])
                fold[f"{base}_cf"] = m.predict(X[test]) * cap[test]
        preds.append(fold)
        print(f"  fold {start:%Y-%m}: trained on {train.sum()} rows, predicted {test.sum()}")

    res = pd.concat(preds)
    res["entsoe"] = entsoe.reindex(res.index)
    res["naive_48h"] = X["lag_48h"].reindex(res.index)
    res["seasonal_naive_168h"] = X["lag_168h"].reindex(res.index)
    res.to_parquet(OUT_DIR / "predictions.parquet")

    # Compare every method on exactly the same hours
    res = res.dropna()
    model_cols = [c for c in res.columns if c not in ("actual", "entsoe", "naive_48h", "seasonal_naive_168h")]
    order = ["seasonal_naive_168h", "naive_48h"] + model_cols + ["entsoe"]
    table = pd.DataFrame({name: score(res["actual"], res[name]) for name in order}).T.round(1)
    print(f"\nWalk-forward results, {len(res)} test hours ({res.index.min():%Y-%m-%d} to {res.index.max():%Y-%m-%d})")
    print(table.to_string())
    table.to_csv(REPORT_DIR / "metrics.csv")

    print("\nTotal training time (seconds)")
    print(pd.Series(fit_seconds).round(1).to_string())

    monthly_cols = [c for c in ["ridge", "random_forest", "lightgbm", "xgboost", "lightgbm_cf", "xgboost_cf"] if c in res.columns]
    monthly = pd.DataFrame({
        name: (res[name] - res["actual"]).abs().groupby(res.index.strftime("%Y-%m")).mean()
        for name in monthly_cols + ["entsoe"]
    }).round(0)
    print("\nMonthly MAE (MW)")
    print(monthly.to_string())
    monthly.to_csv(REPORT_DIR / "monthly_mae.csv")

    imp = pd.Series(last_xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 XGBoost features (last fold)")
    print(imp.head(10).round(3).to_string())


if __name__ == "__main__":
    main()
