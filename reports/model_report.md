# Wind forecast model report

_Generated 2026-11-14 using groq: openai/gpt-oss-120b. All figures are computed by code from the walk-forward results._

**Summary**  
- Task: day‑ahead hourly wind generation forecast for Germany (DE LU) using a walk‑forward test on 2025 (1 Jan 2025 – 30 Dec 2025, 8 735 hours).  
- Model: LightGBM trained to predict capacity factor (CF) and then scaled back to MW.  
- Compared with the best naive baseline (48‑hour naive), the model reduces MAE by **83.7 %**.  
- Compared with ENTSO‑E, the model’s MAE is **11.8 % higher**.  
- The model out‑performs ENTSO‑E in the months **2025‑01** and **2025‑02** only.  
- In the 10‑13 m/s wind regime the model’s MAE is **‑2.5 %** different from ENTSO‑E (treated as roughly equal).  
- The >13 m/s regime contains only **133 hours**; the verdict notes a small sample and the result is inconclusive.  

---

**Comparison with ENTSO‑E**  

| Metric | Model (LightGBM CF) | ENTSO‑E |
|--------|--------------------|---------|
| MAE (MW) | 1 533.2 | 1 371.5 |
| RMSE (MW) | 2 094.1 | 1 884.4 |
| Weighted MAPE (%) | 10.1 % | 9.0 % |
| Bias (MW) | –283.7 (under‑forecast) | –180.1 (under‑forecast) |

- **Overall accuracy:** ENTSO‑E is more accurate overall (lower MAE, RMSE, wMAPE).  
- **Ramps:** For ramp hours the model’s MAE is 2 232 MW versus ENTSO‑E’s 2 266 MW (slightly better). For non‑ramp hours the model’s MAE is 1 496 MW versus ENTSO‑E’s 1 324 MW (worse).  
- **Negative‑price periods:** In the 576 hours with negative prices the model’s bias is +1 639 MW (over‑forecast) while ENTSO‑E’s bias is +1 962 MW; the model is closer to zero here. In all other hours the model’s bias is –419 MW versus ENTSO‑E’s –331 MW (the model is farther from zero).  

---

**Where the model is weak**  

| Wind regime | Hours | Model vs ENTSO‑E (MAE %) | Verdict |
|-------------|------|--------------------------|---------|
| < 4 m/s | 870 | 25.6 % worse | model worse |
| 4‑7 m/s | 4 202 | 19.6 % worse | model worse |
| 7‑10 m/s | 2 850 | 8.4 % worse | model worse |
| 10‑13 m/s | 680 | –2.5 % (roughly equal) | roughly equal |
| > 13 m/s | 133 | 4.3 % worse | model worse (small sample, inconclusive) |

- The model on average under-forecasts overall (bias –283.7 MW).  
- It is less accurate than ENTSO‑E in most wind regimes, especially at low and moderate wind speeds.  
- In the strongest wind regime (>13 m/s) the sample size is small, so the result is not conclusive.  

---

**Data quality**  

- The dataset contains **26 279 rows** covering **1 Jan 2023 – 30 Dec 2025**.  
- Only **5 rows** were imputed to fill short gaps; there are **no missing target hours**.  
- Three single‑hour ENTSO‑E forecast dips at clock‑change hours were treated as artifacts and removed.  
- Bias is defined as forecast minus actual; a negative value means under‑forecasting.  

---

**Limitations**  

- Weather inputs are reanalysis (what actually happened), not archived forecasts, so real day‑ahead performance would be worse.  
- Installed‑capacity timing was assumed known at the start of each year; the capacity‑factor idea was found by studying the 2025 test year.  
- One test year, single run per model, no significance tests.  
- ENTSO‑E remains more accurate overall; this model does not beat it.  

---
Number check: every figure in this text matches the source facts.
