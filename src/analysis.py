"""
Error analysis and SHAP explainability for the final model (LightGBM, capacity-factor target).

Usage (from the project root, after `python -m src.capacity` and `python -m src.models`):
    python -m src.analysis

Writes figures to reports/figures/ and prints tables you can paste into the README.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.features import make_dataset
from src.models import load_capacity, model_factories

MODEL = "lightgbm_cf"  # final model; falls back to plain lightgbm if capacity data is missing

sns.set_theme(style="whitegrid")
FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/wind_clean.parquet")
X, y, entsoe = make_dataset(df)
res = pd.read_parquet("data/processed/predictions.parquet").dropna()
if MODEL not in res.columns:
    print(f"{MODEL} not found in predictions.parquet, using 'lightgbm' instead.")
    MODEL = "lightgbm"
print(f"Final model analysed: {MODEL}")
res["speed"] = X["mean_speed"].reindex(res.index)
res["m_err"] = (res[MODEL] - res["actual"]).abs()
res["ens_err"] = (res["entsoe"] - res["actual"]).abs()

# ---- 1) Model comparison bar chart
metrics = pd.read_csv("reports/metrics.csv", index_col=0)
ax = metrics["MAE_MW"].sort_values().plot.barh(figsize=(8, 5))
ax.set(title="Walk-forward MAE, 2025 test year", xlabel="MAE (MW)")
plt.tight_layout()
plt.savefig(FIG / "05_model_comparison.png", dpi=150)
plt.close()

# ---- 2) Error by wind regime
bins = [0, 4, 7, 10, 13, 40]
labels = ["<4 m/s", "4-7", "7-10", "10-13", ">13"]
res["regime"] = pd.cut(res["speed"], bins=bins, labels=labels)
regime = res.groupby("regime", observed=True).agg(
    hours=("actual", "size"),
    model_MAE=("m_err", "mean"),
    entsoe_MAE=("ens_err", "mean"),
).round(0)
print("\nMAE by wind-speed regime (mean of 16 locations)")
print(regime.to_string())
regime[["model_MAE", "entsoe_MAE"]].plot.bar(figsize=(8, 4), rot=0)
plt.title(f"Forecast error by wind regime ({MODEL} vs ENTSO-E)")
plt.ylabel("MAE (MW)")
plt.tight_layout()
plt.savefig(FIG / "06_error_by_regime.png", dpi=150)
plt.close()

# ---- 3) Error by local hour of day
local_hour = res.index.tz_convert("Europe/Berlin").hour
by_hour = res.groupby(local_hour)[["m_err", "ens_err"]].mean().round(0)
by_hour.columns = ["model_MAE", "entsoe_MAE"]
by_hour.plot(figsize=(9, 4), marker="o")
plt.title("Forecast error by hour of day (local time)")
plt.xlabel("Hour")
plt.ylabel("MAE (MW)")
plt.tight_layout()
plt.savefig(FIG / "07_error_by_hour.png", dpi=150)
plt.close()

# ---- 4) Ramps: top 5% of 3-hour changes in actual generation
change3h = df["wind_actual_mw"].diff(3).abs().reindex(res.index)
ramp = change3h >= change3h.quantile(0.95)
print("\nError during large ramps (top 5% of 3-hour changes) vs normal hours")
print(pd.DataFrame({
    "model_MAE": [res.loc[ramp, "m_err"].mean(), res.loc[~ramp, "m_err"].mean()],
    "entsoe_MAE": [res.loc[ramp, "ens_err"].mean(), res.loc[~ramp, "ens_err"].mean()],
}, index=["ramp hours", "other hours"]).round(0).to_string())

# ---- 5) Bias in negative-price hours (curtailment pattern); forecast minus actual
neg = df["day_ahead_price_eur_mwh"].reindex(res.index) < 0
bias = pd.DataFrame({
    "hours": [int((~neg).sum()), int(neg.sum())],
    "model_bias_MW": [(res.loc[~neg, MODEL] - res.loc[~neg, "actual"]).mean(),
                      (res.loc[neg, MODEL] - res.loc[neg, "actual"]).mean()],
    "entsoe_bias_MW": [(res.loc[~neg, "entsoe"] - res.loc[~neg, "actual"]).mean(),
                       (res.loc[neg, "entsoe"] - res.loc[neg, "actual"]).mean()],
}, index=["price >= 0", "price < 0"]).round(0)
print("\nSigned error (forecast - actual) by price sign")
print(bias.to_string())

# ---- 6) Actual vs predicted scatter
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, col, title in [(axes[0], MODEL, MODEL), (axes[1], "entsoe", "ENTSO-E")]:
    ax.hexbin(res["actual"], res[col], gridsize=45, cmap="viridis", mincnt=1)
    lim = [0, res["actual"].max()]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set(title=f"{title}: predicted vs actual", xlabel="Actual (MW)", ylabel="Predicted (MW)")
plt.tight_layout()
plt.savefig(FIG / "08_pred_vs_actual.png", dpi=150)
plt.close()

# ---- 7) Largest misses
worst = res.nlargest(8, "m_err")[["actual", MODEL, "entsoe", "m_err"]].round(0)
print(f"\n{MODEL}'s 8 largest misses")
print(worst.to_string())

# ---- 8) SHAP: train on data before July 2025, explain July-December 2025
try:
    import shap

    cut = pd.Timestamp("2025-07-01", tz="UTC")
    cap = load_capacity(X.index) if MODEL.endswith("_cf") else None
    target = y / cap if cap is not None else y
    train = X.index < cut
    model = model_factories()["lightgbm"]().fit(X[train], target[train])
    sample = X[X.index >= cut].sample(2000, random_state=0)
    shap_values = shap.TreeExplainer(model).shap_values(sample)

    shap.summary_plot(shap_values, sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG / "09_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    scale = float(cap.loc[sample.index].mean()) if cap is not None else 1.0
    unit = "MW, capacity-factor effect x mean capacity" if cap is not None else "MW"
    top = pd.Series(np.abs(shap_values).mean(axis=0) * scale, index=sample.columns).sort_values(ascending=False)
    print(f"\nTop 10 features by mean |SHAP| ({unit})")
    print(top.head(10).round(0).to_string())
except ImportError:
    print("\nSkipping SHAP: run `pip install shap` and rerun.")

print(f"\nFigures saved in {FIG}")
