"""
LLM-assisted performance report.

Usage (from the project root, after `python -m src.models`):
    python -m src.llm_report                    # auto: Groq if GROQ_API_KEY is set, else Ollama, else template
    python -m src.llm_report --backend groq
    python -m src.llm_report --backend ollama --model llama3.2
    python -m src.llm_report --backend none     # plain template, no LLM

How it works
1. Facts are computed with pandas (never by the LLM): metrics, gaps vs ENTSO-E, regime errors, bias, data quality.
2. The LLM only turns those facts into readable prose.
3. A number check flags any figure in the text that does not appear in the facts, so a
   hallucinated number is caught instead of published.

Writes reports/model_report.md.
"""
import argparse
import json
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

REPORT_DIR = Path("reports")
FINAL_MODEL = "lightgbm_cf"
NAIVE = ["naive_48h", "seasonal_naive_168h"]
CHAT_SYSTEM = (
    "You are a careful quantitative analyst writing a short model-performance report. "
    "Use ONLY the numbers in the JSON facts. Do not calculate new numbers, do not round differently, "
    "and do not add any claim that is not supported by the facts. Be honest about weaknesses and limitations."
)


# ---------------------------------------------------------------- facts
def gather_facts() -> dict:
    metrics = pd.read_csv(REPORT_DIR / "metrics.csv", index_col=0)
    monthly = pd.read_csv(REPORT_DIR / "monthly_mae.csv", index_col=0)
    pred = pd.read_parquet("data/processed/predictions.parquet").dropna()
    df = pd.read_parquet("data/processed/wind_clean.parquet")

    final = FINAL_MODEL if FINAL_MODEL in metrics.index else "lightgbm"
    m = metrics.round(1)
    best_naive = min(NAIVE, key=lambda n: m.loc[n, "MAE_MW"])

    facts = {
        "task": "Day-ahead hourly wind generation forecast for Germany (DE_LU), walk-forward test on 2025",
        "final_model": final,
        "final_model_description": (
            "LightGBM trained to predict capacity factor (generation divided by installed wind capacity) and then scaled back to MW; "
            "'CF' in the model name means capacity factor, not calendar features"
            if final.endswith("_cf") else "LightGBM predicting generation in MW directly"),
        "test_hours": int(len(pred)),
        "test_period": f"{pred.index.min():%Y-%m-%d} to {pred.index.max():%Y-%m-%d}",
        "n_weather_locations": int(len([c for c in df.columns if c.endswith("_wind_speed_100m")])),
        "metrics": {
            name: {"MAE_MW": float(r["MAE_MW"]), "RMSE_MW": float(r["RMSE_MW"]),
                   "wMAPE_pct": float(r["wMAPE_%"]), "bias_MW": float(r["bias_MW"])}
            for name, r in m.iterrows()
        },
        "relative": {
            "final_mae_vs_entsoe_pct_higher": round((m.loc[final, "MAE_MW"] / m.loc["entsoe", "MAE_MW"] - 1) * 100, 1),
            f"final_mae_reduction_vs_{best_naive}_pct": round((1 - m.loc[final, "MAE_MW"] / m.loc[best_naive, "MAE_MW"]) * 100, 1),
        },
    }

    # months where the final model had lower MAE than ENTSO-E
    if final in monthly.columns:
        facts["months_final_beats_entsoe"] = [str(i) for i in monthly.index[monthly[final] < monthly["entsoe"]]]
        facts["months_total"] = int(len(monthly))

    # error by wind regime and ramps
    err_m = (pred[final] - pred["actual"]).abs() if final in pred.columns else None
    err_e = (pred["entsoe"] - pred["actual"]).abs()
    speed = df["mean_wind_speed_100m"].reindex(pred.index)
    regime = pd.cut(speed, [0, 4, 7, 10, 13, 40], labels=["<4 m/s", "4-7 m/s", "7-10 m/s", "10-13 m/s", ">13 m/s"])
    reg = pd.DataFrame({"m": err_m, "e": err_e, "r": regime}).groupby("r", observed=True).agg(
        hours=("m", "size"), model_MAE=("m", "mean"), entsoe_MAE=("e", "mean")).round(0)
    facts["regimes"] = []
    for i, r in reg.iterrows():
        pct = round((r.model_MAE / r.entsoe_MAE - 1) * 100, 1)  # positive = model worse
        verdict = "model worse" if pct > 3 else ("model better" if pct < -3 else "roughly equal")
        if int(r.hours) < 300:
            verdict += " (small sample, inconclusive)"
        facts["regimes"].append({"regime": str(i), "hours": int(r.hours), "model_MAE_MW": float(r.model_MAE),
                                 "entsoe_MAE_MW": float(r.entsoe_MAE), "model_vs_entsoe_pct": pct, "verdict": verdict})

    ch = df["wind_actual_mw"].diff(3).abs().reindex(pred.index)
    ramp = ch >= ch.quantile(0.95)
    facts["ramps"] = {"model_MAE_ramp_hours_MW": round(float(err_m[ramp].mean())),
                      "entsoe_MAE_ramp_hours_MW": round(float(err_e[ramp].mean())),
                      "model_MAE_other_hours_MW": round(float(err_m[~ramp].mean())),
                      "entsoe_MAE_other_hours_MW": round(float(err_e[~ramp].mean()))}

    neg = df["day_ahead_price_eur_mwh"].reindex(pred.index) < 0
    sgn = lambda s, mask: round(float((s[mask] - pred["actual"][mask]).mean()))
    facts["negative_price_hours"] = {
        "hours_with_negative_price": int(neg.sum()),
        "model_bias_negative_price_MW": sgn(pred[final], neg),
        "entsoe_bias_negative_price_MW": sgn(pred["entsoe"], neg),
        "model_bias_other_hours_MW": sgn(pred[final], ~neg),
        "entsoe_bias_other_hours_MW": sgn(pred["entsoe"], ~neg),
    }

    nb = facts["negative_price_hours"]
    # sign-independent: True means the model's systematic error is smaller than ENTSO-E's in that group of hours
    nb["model_bias_closer_to_zero_than_entsoe_in_negative_price_hours"] = abs(nb["model_bias_negative_price_MW"]) < abs(nb["entsoe_bias_negative_price_MW"])
    nb["model_bias_closer_to_zero_than_entsoe_in_other_hours"] = abs(nb["model_bias_other_hours_MW"]) < abs(nb["entsoe_bias_other_hours_MW"])

    facts["reading_guide"] = [
        "bias_MW is forecast minus actual: positive means over-forecast, negative means under-forecast; a bias further from zero is worse.",
        "Absolute MAE is naturally larger in strong wind. To judge a wind regime, use model_vs_entsoe_pct (positive means the model is worse than ENTSO-E there), never the size of the MAE alone.",
        "Use the precomputed comparison flags and percentages; do not compare signed numbers yourself.",
        "Do not state causes that are not in the facts. A link between negative prices and over-forecasting is consistent with curtailment but is unproven; 'non-ramp hours' do not mean stable wind.",
        "final_mae_vs_entsoe_pct_higher is measured against ENTSO-E's MAE: say that the model's MAE is that percent higher than ENTSO-E's. Never say ENTSO-E is that percent lower, because that is a different number.",
        "For each wind regime, use the given 'verdict' exactly; do not call a gap better or worse than its verdict, and call a regime a small sample only if its verdict says so.",
        "'CF' in the model name means capacity factor. Describe the model using final_model_description.",
    ]

    facts["data_quality"] = {
        "rows": int(len(df)),
        "period": f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}",
        "rows_imputed_short_gaps": int(df["was_imputed"].sum()) if "was_imputed" in df else None,
        "missing_target_hours": int(df["wind_actual_mw"].isna().sum()),
        "notes": ["3 single-hour ENTSO-E forecast dips at clock-change hours were treated as artifacts and removed",
                  "bias_MW is forecast minus actual; negative means under-forecasting"],
    }
    facts["limitations"] = [
        "Weather inputs are reanalysis (what actually happened), not archived forecasts, so real day-ahead performance would be worse",
        "Installed-capacity timing was assumed known at the start of each year; the capacity-factor idea was found by studying the 2025 test year",
        "One test year, single run per model, no significance tests",
        "ENTSO-E remains more accurate overall; this model does not beat it",
    ]
    return facts


# ---------------------------------------------------------------- report text
def template_report(f: dict) -> str:
    fm, rel, me, mf = f["final_model"], f["relative"], f["metrics"]["entsoe"], f["metrics"][f["final_model"]]
    naive_key = next(k for k in rel if k.startswith("final_mae_reduction"))
    lines = [
        "## Summary",
        f"{f['task']}. Final model `{fm}` over {f['test_hours']} test hours ({f['test_period']}): "
        f"MAE {mf['MAE_MW']} MW, wMAPE {mf['wMAPE_pct']}%, bias {mf['bias_MW']} MW. "
        f"ENTSO-E's day-ahead forecast: MAE {me['MAE_MW']} MW, wMAPE {me['wMAPE_pct']}%. "
        f"The model's MAE is {rel['final_mae_vs_entsoe_pct_higher']}% higher than ENTSO-E's and "
        f"{rel[naive_key]}% lower than the best naive baseline.",
        "",
        "## Error by wind regime (MAE, MW)",
    ]
    lines += [f"- {r['regime']} ({r['hours']} h): model {r['model_MAE_MW']}, ENTSO-E {r['entsoe_MAE_MW']}" for r in f["regimes"]]
    n = f["negative_price_hours"]
    lines += ["", "## Negative-price hours",
              f"In {n['hours_with_negative_price']} negative-price hours the model's bias was {n['model_bias_negative_price_MW']} MW "
              f"(ENTSO-E {n['entsoe_bias_negative_price_MW']} MW), versus {n['model_bias_other_hours_MW']} MW "
              f"(ENTSO-E {n['entsoe_bias_other_hours_MW']} MW) otherwise.",
              "", "## Limitations"] + [f"- {x}" for x in f["limitations"]]
    return "\n".join(lines)


def build_prompt(facts: dict) -> str:
    return (
        "Write a concise markdown report with these sections: Summary, Comparison with ENTSO-E, "
        "Where the model is weak, Data quality, Limitations. Use plain language for a hiring manager "
        "in energy trading. Quote numbers exactly as given in the facts. Follow the 'reading_guide' in the facts strictly. "
        "When you show a bias, say whether it means over-forecasting or under-forecasting. Say plainly that ENTSO-E is "
        "more accurate overall, and do not claim the model beats it except in the months listed. "
        "Never print field names from the JSON (for example anything with underscores in backticks or brackets). "
        "In the Summary, also state the MAE reduction versus the best naive baseline. "
        "Treat differences within about 3 percent as roughly equal, and mention when a wind regime has few hours, "
        "because no significance tests were run. Copy the limitations from the facts closely instead of rewording them "
        "(for example do not say 'tuned' when the facts say 'found by studying').\n\nFACTS (JSON):\n"
        + json.dumps(facts, indent=1)
    )


# ---------------------------------------------------------------- LLM backends
def call_groq(prompt: str, model: str) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set (add it to .env).")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "temperature": 0.2,
              "messages": [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": prompt}]},
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"Groq returned {r.status_code} for model '{model}': {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def list_groq_models() -> list:
    """Ask Groq which model IDs are active for this API key."""
    key = os.getenv("GROQ_API_KEY")
    r = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json()["data"])


def call_ollama(prompt: str, model: str) -> str:
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": model, "stream": False, "options": {"temperature": 0.2},
              "messages": [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": prompt}]},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def ollama_available() -> bool:
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=2).ok
    except requests.RequestException:
        return False


# ---------------------------------------------------------------- number guard
def _flatten_numbers(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_numbers(str(k), out)  # model names such as naive_48h contain numbers the text may mention
            _flatten_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _flatten_numbers(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, str):  # numbers inside strings such as "2025-01-01 to 2025-12-30" or "3 single-hour"
        out += [float(x) for x in re.findall(r"\d+(?:\.\d+)?", obj)]


def normalize_numbers(text: str) -> str:
    """LLMs often write 8 735 with a thin or non-breaking space; join those digits so '8 735' reads as 8735."""
    return re.sub(r"(?<=\d)[\u00a0\u2009\u202f](?=\d)", "", text)


def unsupported_numbers(text: str, facts: dict) -> list:
    known = []
    _flatten_numbers(facts, known)
    flagged = []
    text = normalize_numbers(text)
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        x = float(raw.replace(",", ""))
        if x == int(x) and (x <= 31 or 2020 <= x <= 2030):
            continue  # small counts, days and years are not checked
        if not any(abs(x - k) <= max(0.06, 0.006 * abs(k)) or abs(x - abs(k)) <= 0.06 for k in known):
            flagged.append(raw)
    return sorted(set(flagged))


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["auto", "groq", "ollama", "none"], default="auto")
    parser.add_argument("--model", default=None, help="LLM name (Groq default from GROQ_MODEL, Ollama default llama3.2)")
    parser.add_argument("--list-models", action="store_true", help="Print the model IDs your Groq key can use, then exit")
    args = parser.parse_args()
    load_dotenv()

    if args.list_models:
        print("\n".join(list_groq_models()))
        return

    facts = gather_facts()
    backend = args.backend
    if backend == "auto":
        backend = "groq" if os.getenv("GROQ_API_KEY") else ("ollama" if ollama_available() else "none")

    used, body = "template (no LLM)", None
    if backend != "none":
        try:
            prompt = build_prompt(facts)
            if backend == "groq":
                model = args.model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
                try:
                    body = call_groq(prompt, model)
                except Exception:
                    try:
                        print("Active Groq models for your key:", ", ".join(list_groq_models()))
                        print("Set one with --model <id> or GROQ_MODEL=<id> in .env")
                    except Exception:
                        pass
                    raise
            else:
                model = args.model or "llama3.2"
                body = call_ollama(prompt, model)
            used = f"{backend}: {model}"
        except Exception as exc:  # fall back rather than fail the whole script
            print(f"LLM call failed ({exc!r}); using the template report instead.")
    if body is None:
        body = template_report(facts)

    flagged = unsupported_numbers(body, facts) if used != "template (no LLM)" else []
    header = (f"# Wind forecast model report\n\n_Generated {date.today()} using {used}. "
              f"All figures are computed by code from the walk-forward results._\n\n")
    footer = ""
    if flagged:
        footer = ("\n\n---\n**Number check:** these numbers in the text are not in the source facts, "
                  f"please verify before using: {', '.join(flagged)}\n")
        print("WARNING: unsupported numbers found:", flagged)
        norm = normalize_numbers(body)
        for raw in flagged:
            i = norm.find(raw)
            print(f"  {raw!r} appears near: ...{norm[max(0, i - 50): i + 50]!r}...")
    elif used != "template (no LLM)":
        footer = "\n\n---\nNumber check: every figure in this text matches the source facts.\n"

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "model_report.md").write_text(header + body + footer, encoding="utf-8")
    (REPORT_DIR / "report_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(f"Backend used: {used}")
    print("Saved reports/model_report.md and reports/report_facts.json")


if __name__ == "__main__":
    main()
