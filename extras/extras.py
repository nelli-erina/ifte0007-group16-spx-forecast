"""Supporting calculations for numbers quoted in the report that are not produced by spx_forecast_v2.py.

Run from this folder:  python extras.py
Inputs (all saved in this ZIP, no network needed):
  ../alternatives/members/                       members' own pilot forecasts (KNN workbook, historical mean / AR(1) slides)
  ../alternatives/pilot/data/gspc.csv            S&P 500 closes to 30 Oct 2026 (pilot snapshot)
  ../alternatives/pilot/results/                 pilot OLS forecast and event adjustment (pilot script output)
  ../alternatives/ridge/prices.csv               prices used by the Ridge re-implementation
  ../data_<tag>/gspc.csv                         main frozen S&P 500 closes (for the pilot-week path and tail counts)
  data/gspc_ohlc.csv, data/btc.csv, data/eth.csv opens/closes for the overnight and market-choice checks
Outputs (results/):
  pilot_models.csv        Table 'Pilot forecasts': every method's forecast for the 30 Oct -> 5 Nov window
  ridge_with_without.csv  Table 'Does the earnings factor add value?'
  market_choice.csv       median absolute move: S&P 500 (6 days) vs Bitcoin and Ethereum (7 days), last 2 years
  overnight.csv           effect of knowing the Thursday-to-Friday overnight move (2010-2026)
  tail_events.csv         daily falls of 4% or more over the last ten years
"""
import contextlib, io, re, runpy, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
TAG = sys.argv[1] if len(sys.argv) > 1 else sorted(p.name[5:] for p in ROOT.glob("data_*"))[-1]
score = lambda F, A: 50 - min(50, 5 * 100 * abs(F - A) / A)


def quiet_run(path):
    """Run a script in its own folder and return its namespace, hiding its printout."""
    import os
    cwd = os.getcwd(); os.chdir(path.parent)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return runpy.run_path(str(path))
    finally:
        os.chdir(cwd)


# ------------------------------------------------------------------ 1. pilot models, same window (data to 30 Oct, target 5 Nov)
# Members' own forecasts for this window are taken from their files in ../alternatives/members/:
#   KNN (5-day momentum + 40-day volatility, k=50): KNN_member_forecast_5Nov.xlsx, sheet "Forecast" -> 7,567.41
#   Historical mean and AR(1): HistMean_AR1_member_slides.pptx, slide 4 -> 7,603.91 and 7,610.22
# The member's Ridge model was built for a later one-day target (9 Nov), so it is re-estimated here by the
# agent on data to 30 Oct at the four-day horizon (../alternatives/ridge/ridge_4day.py).
import openpyxl
pilot = ROOT / "alternatives/pilot"
A = 7353.73                                                   # official close on Thu 5 Nov 2026
c = pd.read_csv(pilot / "data/gspc.csv", parse_dates=["Date"], index_col="Date")["Close"]
P0 = float(c[c.index <= pd.Timestamp("2026-10-30")].iloc[-1])
wb = openpyxl.load_workbook(ROOT / "alternatives/members/KNN_member_forecast_5Nov.xlsx", data_only=True)
F_knn = next(r[1] for r in wb["Forecast"].iter_rows(values_only=True) if r[0] == "Forecast level")
F_mean, F_ar = 7603.91, 7610.22                               # as reported on the member's slide 4
log = (pilot / "results/run_log.txt").read_text()
F_ols = float(re.search(r"Model forecast ([\d.]+)", log).group(1))
F_adj = float(re.search(r"adjusted forecast ([\d.]+)", log).group(1))
rg = quiet_run(ROOT / "alternatives/ridge/ridge_4day.py")
F_ridge = float(rg["Fc"])
rows = [("No change", "Benchmark", "-", P0, "benchmark"),
        ("Historical mean", "Benchmark", "Mean historical return", F_mean, "member"),
        ("AR(1)", "Time series", "Previous-day return", F_ar, "member"),
        ("OLS regression", "Linear regression", "Reversal, VIX gap", F_ols, "member"),
        ("Ridge regression", "Penalised regression", "Momentum, volatility, earnings", F_ridge, "re-estimated by agent"),
        ("k-nearest neighbours", "Machine learning", "Momentum, volatility (k=50)", F_knn, "member"),
        ("OLS + event adjustment", "Model + judgement", "as OLS, +0.59%", F_adj, "member (submitted)")]
pm = pd.DataFrame(rows, columns=["method", "type", "factors", "forecast", "source"])
pm["error_pct"] = 100 * (pm.forecast - A).abs() / A; pm["score"] = [score(f, A) for f in pm.forecast]
pm.to_csv(OUT / "pilot_models.csv", index=False)

# ------------------------------------------------------------------ 2. Ridge with and without the earnings factor
def ridge_bt(cols):
    P, known, Hr = rg["P"], rg["known"], rg["H"]; pos_cut = P.index.get_loc(rg["CUT"]); out = []
    for dte in known.index:
        if dte < pd.Timestamp("2026-07-10") or P.index.get_loc(dte) + Hr > pos_cut:
            continue
        p = P.index.get_loc(dte); tr = known[[P.index.get_loc(x) + Hr <= p for x in known.index]]
        if len(tr) < 80:
            continue
        Xt, yt = tr[cols].to_numpy(), tr["y"].to_numpy(); lam = rg["choose_lambda"](Xt, yt)
        pr = float(rg["ridge_pred"](rg["ridge_fit"](Xt, yt, lam), known.loc[dte, cols].to_numpy(dtype=float)))
        out.append((dte, pr, known.loc[dte, "y"], known.loc[dte, "earn"]))
    b_ = pd.DataFrame(out, columns=["o", "pred", "act", "earn"]).set_index("o")
    b_["E"] = 100 * np.abs((1 + b_.pred) / (1 + b_.act) - 1); b_["Erw"] = 100 * np.abs(1 / (1 + b_.act) - 1)
    return b_
full, base = ridge_bt(["mom5", "vol20", "earn"]), ridge_bt(["mom5", "vol20"])
act = full.earn.abs() > 1e-12; diff = full.E - base.E
rw = pd.DataFrame([
    {"model": "Ridge with earnings factor", "all": full.E.mean(), "active": full.E[act].mean(), "zero": full.E[~act].mean(), "better_than_no_change": (full.E < full.Erw).mean()},
    {"model": "Ridge without earnings factor", "all": base.E.mean(), "active": base.E[act].mean(), "zero": base.E[~act].mean(), "better_than_no_change": (base.E < base.Erw).mean()},
    {"model": "No change", "all": full.Erw.mean(), "active": full.Erw[act].mean(), "zero": full.Erw[~act].mean(), "better_than_no_change": np.nan},
    {"model": "n origins", "all": len(full), "active": int(act.sum()), "zero": int((~act).sum()), "better_than_no_change": np.nan},
    {"model": "with minus without (pp), t", "all": diff.mean(), "active": diff.mean() / (diff.std() / np.sqrt(len(diff))), "zero": np.nan, "better_than_no_change": np.nan}])
rw.to_csv(OUT / "ridge_with_without.csv", index=False)

# ------------------------------------------------------------------ 3. market choice
def med_move(f, h):
    s = pd.read_csv(HERE / "data" / f, parse_dates=["Date"], index_col="Date")["Close"]
    s = s[s.index >= s.index[-1] - pd.DateOffset(years=2)]
    return float((100 * (s.shift(-h) / s - 1)).abs().median())
pd.DataFrame([{"market": "S&P 500", "horizon_days": 6, "median_abs_move_pct": med_move("gspc_ohlc.csv", 6)},
              {"market": "Bitcoin", "horizon_days": 7, "median_abs_move_pct": med_move("btc.csv", 7)},
              {"market": "Ethereum", "horizon_days": 7, "median_abs_move_pct": med_move("eth.csv", 7)}]).to_csv(OUT / "market_choice.csv", index=False)

# ------------------------------------------------------------------ 4. overnight move (Thu close -> Fri open), 2010-2026
g = pd.read_csv(HERE / "data/gspc_ohlc.csv", parse_dates=["Date"], index_col="Date")
g = g[g.index >= "2010-01-01"]
on = np.log(g.Open / g.Close.shift(1)); fri = on[on.index.dayofweek == 4].dropna()
r6 = np.log(g.Close.shift(-6) / g.Close); thu = r6[r6.index.dayofweek == 3].dropna()
fo = fri.copy(); fo.index = fo.index - pd.Timedelta(days=1)
j = pd.concat([thu, fo], axis=1, join="inner").dropna(); j.columns = ["r6", "on"]
e0, e1 = 100 * j.r6.abs(), 100 * (j.r6 - j.on).abs()
sc = lambda e: (50 - np.minimum(50, 5 * e)).mean()
pd.DataFrame([{"weeks": len(j), "median_abs_overnight_pct": 100 * fri.abs().median(), "share_overnight_gt1pct": (fri.abs() > 0.01).mean(),
               "mae_no_change": e0.mean(), "mae_with_overnight": e1.mean(), "score_no_change": sc(e0), "score_with_overnight": sc(e1),
               "share_weeks_better": (e1 < e0).mean()}]).to_csv(OUT / "overnight.csv", index=False)

# ------------------------------------------------------------------ 5. tail events in the main frozen data
m = pd.read_csv(ROOT / f"data_{TAG}/gspc.csv", parse_dates=["Date"], index_col="Date")["Close"]
r = 100 * m.pct_change(); r = r[r.index >= r.index[-1] - pd.DateOffset(years=10)]
falls = r[r <= -4].round(2)
falls.rename("return_pct").to_csv(OUT / "tail_events.csv")
print(pm.round(2).to_string(index=False)); print(rw.round(3).to_string(index=False))
print(pd.read_csv(OUT / "market_choice.csv").round(2).to_string(index=False)); print(pd.read_csv(OUT / "overnight.csv").round(3).to_string(index=False))
print(f"falls of 4% or more in last 10 years: {len(falls)} of {len(r)} trading days")
