"""
Streamlit dashboard for the German wind forecasting project.

Run from the project root:
    streamlit run app/streamlit_app.py

Needs the outputs of: src.clean, src.models (and optionally src.analysis, src.llm_report).
"""
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FINAL_MODEL = "lightgbm_cf"
TZ = "Europe/Berlin"

st.set_page_config(page_title="German wind forecast", layout="wide")


@st.cache_data
def load_data():
    pred = pd.read_parquet(ROOT / "data/processed/predictions.parquet").dropna()
    clean = pd.read_parquet(ROOT / "data/processed/wind_clean.parquet")
    metrics = pd.read_csv(ROOT / "reports/metrics.csv", index_col=0)
    monthly = pd.read_csv(ROOT / "reports/monthly_mae.csv", index_col=0)
    return pred, clean, metrics, monthly


try:
    pred, clean, metrics, monthly = load_data()
except FileNotFoundError as exc:
    st.error(f"Missing file: {exc.filename}")
    st.info("Run `python -m src.clean` and `python -m src.models` first, then reload this page.")
    st.stop()


def score(actual: pd.Series, forecast: pd.Series) -> dict:
    err = forecast - actual
    return {
        "MAE (MW)": err.abs().mean(),
        "RMSE (MW)": float(np.sqrt((err ** 2).mean())),
        "wMAPE (%)": err.abs().sum() / actual.sum() * 100,
        "Bias (MW)": err.mean(),  # forecast minus actual; negative = under-forecast
    }


# ------------------------------------------------------------------ sidebar
local = pred.copy()
local.index = local.index.tz_convert(TZ).tz_localize(None)  # local clock time for display
all_models = [c for c in local.columns if c != "actual"]

st.sidebar.header("Filters")
d0, d1 = local.index.min().date(), local.index.max().date()
picked = st.sidebar.date_input("Date range (local time)", value=(d0, d1), min_value=d0, max_value=d1)
if not isinstance(picked, tuple) or len(picked) != 2:
    st.info("Pick an end date in the sidebar.")
    st.stop()
start, end = picked
defaults = [m for m in [FINAL_MODEL, "entsoe"] if m in all_models]
chosen = st.sidebar.multiselect("Forecasts to show", all_models, default=defaults)

mask = (local.index >= pd.Timestamp(start)) & (local.index < pd.Timestamp(end) + timedelta(days=1))
view_local, view_utc = local[mask], pred[mask]
if view_local.empty:
    st.warning("No data in that range.")
    st.stop()

# ------------------------------------------------------------------ header
st.title("German wind generation: day-ahead forecast")
st.caption(
    "Hourly forecasts for Germany (DE_LU), walk-forward test on 2025. Each month is predicted by models trained only "
    "on earlier data. Weather inputs are reanalysis, so real-world accuracy would be lower. "
    f"Showing {len(view_local):,} hours from {start} to {end}."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Forecast vs actual", "Accuracy", "Where it fails", "Report", "About"])

# ------------------------------------------------------------------ tab 1
with tab1:
    cols = st.columns(2)
    for col, name in zip(cols, [m for m in [FINAL_MODEL, "entsoe"] if m in all_models]):
        s = score(view_utc["actual"], view_utc[name])
        bias = s["Bias (MW)"]

        if abs(bias) < 0.1 * s["MAE (MW)"]:
            bias_label = "Roughly balanced"
        elif bias < 0:
            bias_label = "Under-forecasting"
        else:
            bias_label = "Over-forecasting"

        col.metric(f"{name}: MAE in range", f"{s['MAE (MW)']:,.0f} MW")
        col.caption(f"Bias: {bias:,.0f} MW | {bias_label}")

    plot = view_local[["actual"] + chosen]
    if (end - start).days > 31:
        plot = plot.resample("D").mean()
        st.caption("Range longer than 31 days: showing daily averages.")
    else:
        plot = plot.groupby(level=0).mean()  # clock-change hour appears twice in local time
    st.line_chart(plot, height=420)
    st.caption("MW. Pick a shorter range in the sidebar to see hourly detail.")
# ------------------------------------------------------------------ tab 2
with tab2:
    rows = {m: score(view_utc["actual"], view_utc[m]) for m in all_models}
    table = pd.DataFrame(rows).T.sort_values("MAE (MW)").round(1)
    st.subheader("Accuracy in the selected range")
    st.dataframe(table, use_container_width=True)
    st.bar_chart(table["MAE (MW)"])
    st.caption("Bias = forecast minus actual. Negative means the forecast is too low; a bias further from zero is worse.")

    st.subheader("Monthly MAE over the whole test year")
    st.line_chart(monthly)
    st.caption("This chart ignores the sidebar date filter.")

# ------------------------------------------------------------------ tab 3
with tab3:
    model = st.selectbox("Model to analyse", [m for m in all_models if m != "entsoe"],
                         index=[m for m in all_models if m != "entsoe"].index(FINAL_MODEL) if FINAL_MODEL in all_models else 0)
    err_m = (view_utc[model] - view_utc["actual"]).abs()
    err_e = (view_utc["entsoe"] - view_utc["actual"]).abs()

    st.subheader("Error by wind regime")
    speed = clean["mean_wind_speed_100m"].reindex(view_utc.index)
    regime = pd.cut(speed, [0, 4, 7, 10, 13, 40], labels=["<4 m/s", "4-7", "7-10", "10-13", ">13"])
    reg = pd.DataFrame({"m": err_m, "e": err_e, "r": regime}).groupby("r", observed=True).agg(
        hours=("m", "size"), model_MAE=("m", "mean"), entsoe_MAE=("e", "mean"))
    reg["model vs ENTSO-E (%)"] = (reg["model_MAE"] / reg["entsoe_MAE"] - 1) * 100
    st.dataframe(reg.round(1), use_container_width=True)
    st.caption("Positive percentage = the model is worse than ENTSO-E in that wind regime. "
               "Regimes with few hours (for example above 13 m/s) are too small to draw conclusions from.")

    st.subheader("Signed error by price sign")
    neg = clean["day_ahead_price_eur_mwh"].reindex(view_utc.index) < 0
    sign_tbl = pd.DataFrame({
        "hours": [int((~neg).sum()), int(neg.sum())],
        f"{model} bias (MW)": [(view_utc.loc[~neg, model] - view_utc.loc[~neg, "actual"]).mean(),
                               (view_utc.loc[neg, model] - view_utc.loc[neg, "actual"]).mean()],
        "ENTSO-E bias (MW)": [(view_utc.loc[~neg, "entsoe"] - view_utc.loc[~neg, "actual"]).mean(),
                              (view_utc.loc[neg, "entsoe"] - view_utc.loc[neg, "actual"]).mean()],
    }, index=["price >= 0", "price < 0"]).round(0)
    st.dataframe(sign_tbl, use_container_width=True)
    st.caption("Positive = forecast above actual (over-forecast). Over-forecasting in negative-price hours is "
               "consistent with curtailment, but that cause is not proven.")

    st.subheader("Largest misses in the range")
    worst = view_local.assign(abs_error=(view_local[model] - view_local["actual"]).abs()).nlargest(10, "abs_error")
    st.dataframe(worst[["actual", model, "entsoe", "abs_error"]].round(0), use_container_width=True)

    figs = ROOT / "reports/figures"
    shap_png = figs / "09_shap_summary.png"
    if shap_png.exists():
        st.subheader("What drives the forecast (SHAP)")
        st.image(str(shap_png))

# ------------------------------------------------------------------ tab 4
with tab4:
    report = ROOT / "reports/model_report.md"
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
    else:
        st.info("No report yet. Run `python -m src.llm_report`.")

# ------------------------------------------------------------------ tab 5
with tab5:
    st.markdown(
        """
**Forecast set-up.** The forecast for day D is made around midday on day D-1. Every feature built from past
generation is lagged by at least 36 hours, so nothing from the future leaks in.

**Final model.** LightGBM predicting capacity factor (generation divided by installed wind capacity), then scaled back
to MW. Compared with Ridge, random forest, XGBoost, two naive baselines and ENTSO-E's own day-ahead forecast.

**Limitations**
- Weather inputs are reanalysis (what actually happened), not archived forecasts, so real day-ahead accuracy would be worse.
- Installed capacity was assumed known at the start of each year, and the capacity-factor idea was found by studying the 2025 test year.
- One test year, a single run per model, no significance tests.
- ENTSO-E's forecast is more accurate overall; this model does not consistently beat it.
- Three single-hour dips in ENTSO-E's forecast at clock-change hours were treated as data artifacts and removed.
"""
    )
