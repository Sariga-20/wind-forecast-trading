# %% [markdown]
# # Day 3: Exploratory data analysis
# Wind generation in Germany (DE_LU), 2023-2025, hourly.
# Run cell by cell in VS Code ("Run Cell" above each `# %%`), or run the whole file.
# Figures are saved to reports/figures/ for the README.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/wind_clean.parquet")
local = df.copy()
local.index = local.index.tz_convert("Europe/Berlin")  # for hour-of-day / month patterns
print(df.shape)
print(df[["wind_actual_mw", "wind_entsoe_forecast_mw", "day_ahead_price_eur_mwh", "mean_wind_speed_100m"]].describe().round(1))

# %% [markdown]
# ## 1. Two weeks of actual vs ENTSO-E day-ahead forecast

# %%
sample = df.loc["2024-01-08":"2024-01-21", ["wind_actual_mw", "wind_entsoe_forecast_mw"]]
ax = sample.plot(figsize=(13, 4), lw=1.2)
ax.set_title("Wind generation: actual vs ENTSO-E day-ahead forecast (Jan 2024)")
ax.set_ylabel("MW")
plt.tight_layout()
plt.savefig(FIG / "01_actual_vs_forecast_sample.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. Daily and seasonal patterns

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
local.groupby(local.index.hour)["wind_actual_mw"].mean().plot(ax=axes[0], marker="o")
axes[0].set(title="Average wind by hour of day (local time)", xlabel="Hour", ylabel="MW")
local.groupby(local.index.month)["wind_actual_mw"].mean().plot.bar(ax=axes[1])
axes[1].set(title="Average wind by month", xlabel="Month", ylabel="MW")
plt.tight_layout()
plt.savefig(FIG / "02_daily_seasonal_patterns.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Wind speed vs generation (the power curve)
# Generation rises roughly with the cube of wind speed, then flattens at rated power.

# %%
plt.figure(figsize=(7, 5))
plt.hexbin(df["mean_wind_speed_100m"], df["wind_actual_mw"], gridsize=50, cmap="viridis", mincnt=1)
plt.colorbar(label="hours")
plt.xlabel("Mean wind speed at 100 m (m/s), 5 locations")
plt.ylabel("Wind generation (MW)")
plt.title("Wind speed vs generation")
plt.tight_layout()
plt.savefig(FIG / "03_power_curve.png", dpi=150)
plt.show()

corr = df[["wind_actual_mw", "mean_wind_speed_100m"]].corr().iloc[0, 1]
print(f"Correlation actual vs mean wind speed: {corr:.2f}")

# %% [markdown]
# ## 4. Distribution and autocorrelation

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(df["wind_actual_mw"], bins=60, ax=axes[0])
axes[0].set(title="Distribution of hourly wind generation", xlabel="MW")

lags = range(1, 73)
acf = [df["wind_actual_mw"].autocorr(lag=l) for l in lags]
axes[1].bar(list(lags), acf)
axes[1].set(title="Autocorrelation up to 72 hours", xlabel="Lag (hours)", ylabel="Correlation")
plt.tight_layout()
plt.savefig(FIG / "04_distribution_autocorr.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Benchmark: how good is ENTSO-E's own day-ahead forecast?
# This is the number your model has to beat or honestly compare against.
# MAPE explodes when generation is near zero, so wMAPE (total absolute error / total actual) is used too.

# %%
b = df[["wind_actual_mw", "wind_entsoe_forecast_mw"]].dropna()
err = b["wind_entsoe_forecast_mw"] - b["wind_actual_mw"]
mae = err.abs().mean()
rmse = np.sqrt((err ** 2).mean())
wmape = err.abs().sum() / b["wind_actual_mw"].sum() * 100
print(f"ENTSO-E day-ahead forecast  MAE={mae:,.0f} MW  RMSE={rmse:,.0f} MW  wMAPE={wmape:.1f}%")

# Naive baselines for comparison
persist_24 = b["wind_actual_mw"].shift(24)
persist_168 = b["wind_actual_mw"].shift(168)
for name, pred in [("naive (same hour yesterday)", persist_24), ("seasonal naive (same hour last week)", persist_168)]:
    m = pd.concat([b["wind_actual_mw"], pred], axis=1).dropna()
    e = m.iloc[:, 1] - m.iloc[:, 0]
    print(f"{name:38s} MAE={e.abs().mean():,.0f} MW  RMSE={np.sqrt((e**2).mean()):,.0f} MW  "
          f"wMAPE={e.abs().sum() / m.iloc[:, 0].sum() * 100:.1f}%")

# %%
# Where does ENTSO-E's forecast make its biggest errors?
worst = err.abs().sort_values(ascending=False).head(10)
print(pd.DataFrame({"actual": b.loc[worst.index, "wind_actual_mw"],
                    "forecast": b.loc[worst.index, "wind_entsoe_forecast_mw"],
                    "abs_error": worst}).round(0))
