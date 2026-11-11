"""
S&P 500 practice-round forecast  (IFTE0007 group practice)
===========================================================

Target      : S&P 500 price index official close, Thursday 5 Nov 2026 (16:00 New York)
Info cutoff : Friday 30 Oct 2026 close (16:00 New York / 20:00 London)
Horizon     : 4 trading days (Fri close -> Thu close)

Method (two economically motivated factors + benchmark):
    y_t     = ln(P_{t+4} / P_t)                               (4-day log return)
    REV_t   = ln(P_t / P_{t-5})                               (past-week return; short-term reversal)
    VGAP_t  = VIX_t - mean(VIX over previous 252 trading days) (fear premium; risk compensation)

    y_t = a + b * REV_t + c * VGAP_t + e_t          (OLS, rolling window, refit at every origin)

    Forecast level  F = P_cutoff * exp(a + b*REV + c*VGAP)

Benchmarks: random walk (F = P_cutoff) and historical drift (F = P_cutoff * exp(mean y)).

Validation: forecasts made every Friday close from 2006 to 2026, each using ONLY
observations whose 4-day outcome was already known at that Friday (no look-ahead).

Run:
    pip install yfinance pandas numpy matplotlib
    python spx_forecast.py            # downloads data once, then reuses frozen CSVs in ./data
"""

from __future__ import annotations

import json
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- settings
CUTOFF = pd.Timestamp("2026-10-30")      # last usable close
TARGET = pd.Timestamp("2026-11-05")      # forecast target close
H = 4                                    # trading days ahead
WINDOW_YEARS = 10                        # rolling estimation window (main spec)
TEST_START = pd.Timestamp("2006-01-01")  # first backtest origin
VIX_LOOKBACK = 252

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

LOG_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    LOG_LINES.append(msg)


# ----------------------------------------------------------------------------- 1. data
def download_and_freeze() -> None:
    """Download daily closes from Yahoo Finance once and freeze them to CSV."""
    import yfinance as yf

    meta = {"retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "source": "Yahoo Finance via yfinance " + yf.__version__,
            "end_exclusive": "2026-10-31",
            "series": {}}
    for ticker, fname, start in [("^GSPC", "gspc.csv", "1950-01-01"),
                                 ("^VIX", "vix.csv", "1990-01-01")]:
        h = yf.Ticker(ticker).history(start=start, end="2026-10-31",
                                      auto_adjust=False, actions=False)
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        h = h[["Close"]].rename_axis("Date")
        h.to_csv(DATA / fname)
        meta["series"][ticker] = {"file": fname,
                                  "url": f"https://finance.yahoo.com/quote/{ticker}/history",
                                  "first": str(h.index[0].date()),
                                  "last": str(h.index[-1].date()),
                                  "last_close": round(float(h["Close"].iloc[-1]), 2),
                                  "rows": int(len(h))}
    (DATA / "data_snapshot_meta.json").write_text(json.dumps(meta, indent=2))


def load() -> pd.DataFrame:
    if not (DATA / "gspc.csv").exists() or not (DATA / "vix.csv").exists():
        log("Frozen CSVs not found -> downloading once from Yahoo Finance ...")
        download_and_freeze()
    spx = pd.read_csv(DATA / "gspc.csv", parse_dates=["Date"], index_col="Date")["Close"]
    vix = pd.read_csv(DATA / "vix.csv", parse_dates=["Date"], index_col="Date")["Close"]
    df = pd.concat({"spx": spx, "vix": vix}, axis=1).sort_index()
    df = df[df.index <= CUTOFF]            # information boundary: nothing after the cutoff
    return df


def checks(df: pd.DataFrame) -> None:
    log("=== DATA CHECKS ===")
    last = df.index[-1]
    log(f"Last observation           : {last.date()}  (must equal cutoff {CUTOFF.date()})")
    log(f"S&P 500 close at cutoff    : {df['spx'].iloc[-1]:.2f}  (press check: 7,580.07)")
    log(f"VIX close at cutoff        : {df['vix'].iloc[-1]:.2f}  (press check: 19.42)")
    assert last == CUTOFF, "Last data point is not the cutoff date"
    assert abs(df["spx"].iloc[-1] - 7580.07) < 0.5, "S&P close does not match press reports"
    assert df.index.is_unique
    r = np.log(df["spx"]).diff()
    big = r[r.abs() > 0.08]
    log(f"Daily moves > 8% (sanity)  : {len(big)} days, e.g. {[str(d.date()) for d in big.index[:4]]}")
    log("")


# ----------------------------------------------------------------------------- 2. factors
def build(df: pd.DataFrame) -> pd.DataFrame:
    lp = np.log(df["spx"])
    out = pd.DataFrame(index=df.index)
    out["spx"] = df["spx"]
    out["vix"] = df["vix"]
    out["REV"] = lp - lp.shift(5)                                     # past-week return
    out["R1"] = lp - lp.shift(1)                                      # last-day return (alt.)
    out["VGAP"] = df["vix"] - df["vix"].rolling(VIX_LOOKBACK).mean().shift(1)
    out["y"] = lp.shift(-H) - lp                                      # future 4-day return
    out["pos"] = np.arange(len(out))                                  # trading-day counter
    return out


def fit(train: pd.DataFrame, factors: list[str]) -> np.ndarray:
    X = np.column_stack([np.ones(len(train))] + [train[f].to_numpy() for f in factors])
    beta, *_ = np.linalg.lstsq(X, train["y"].to_numpy(), rcond=None)
    return beta


def train_set(d: pd.DataFrame, origin_pos: int, years: float, factors: list[str]) -> pd.DataFrame:
    """Rows whose outcome (t+H) is already observed at the origin, inside the window."""
    lo = origin_pos - int(years * 252)
    t = d[(d["pos"] + H <= origin_pos) & (d["pos"] >= lo)]
    return t.dropna(subset=factors + ["y"])


# ----------------------------------------------------------------------------- 3. backtest
def backtest(d: pd.DataFrame, factors: list[str], years: float, shrink: float = 1.0) -> pd.DataFrame:
    origins = d[(d.index >= TEST_START) & (d.index.dayofweek == 4)].dropna(subset=factors + ["y"])
    rows = []
    for date, o in origins.iterrows():
        tr = train_set(d, int(o["pos"]), years, factors)
        if len(tr) < 500:
            continue
        b = fit(tr, factors)
        mu = b[0] + sum(b[i + 1] * o[f] for i, f in enumerate(factors))
        drift = tr["y"].mean()
        P0, A = o["spx"], o["spx"] * np.exp(o["y"])
        rows.append({"origin": date, "P0": P0, "actual": A,
                     "F_model": P0 * np.exp(shrink * mu),
                     "F_rw": P0,
                     "F_drift": P0 * np.exp(drift),
                     "vix": o["vix"]})
    bt = pd.DataFrame(rows).set_index("origin")
    for m in ["model", "rw", "drift"]:
        bt[f"E_{m}"] = 100 * (bt[f"F_{m}"] - bt["actual"]).abs() / bt["actual"]
        bt[f"S_{m}"] = 50 - np.minimum(50, 0.5 * bt[f"E_{m}"] / 0.1)   # assignment scoring rule
    return bt


def summarise(bt: pd.DataFrame, label: str) -> dict:
    d = bt["E_model"] - bt["E_rw"]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))     # non-overlapping weekly windows
    return {"spec": label, "n": len(bt),
            "MAPE_model": bt["E_model"].mean(), "MAPE_rw": bt["E_rw"].mean(),
            "MAPE_drift": bt["E_drift"].mean(),
            "score_model": bt["S_model"].mean(), "score_rw": bt["S_rw"].mean(),
            "win_rate_vs_rw": (bt["E_model"] < bt["E_rw"]).mean(),
            "DM_t(model-rw)": t}


# ----------------------------------------------------------------------------- 4. midterm event study
MIDTERMS = [dt.date(y, 11, 1) + dt.timedelta(days=((0 - dt.date(y, 11, 1).weekday()) % 7) + 1)
            for y in range(1950, 2026, 4)]    # Tuesday after first Monday of November


def midterm_study(d: pd.DataFrame) -> pd.DataFrame:
    lp = np.log(d["spx"])
    rows = []
    for e in MIDTERMS:
        e = pd.Timestamp(e)
        before = d.index[d.index < e - pd.Timedelta(days=3)]       # last close on/before Friday prior
        if len(before) == 0:
            continue
        o = before[-1]
        i = d.index.get_loc(o)
        if i + H >= len(d):
            continue
        rows.append({"election": e.date(), "origin": o.date(),
                     "target": d.index[i + H].date(),
                     "ret_pct": 100 * (np.exp(lp.iloc[i + H] - lp.iloc[i]) - 1)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def main() -> None:
    log(f"Run at {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} UTC | "
        f"python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}")
    df = load()
    checks(df)
    d = build(df)

    # ---- main spec + alternatives (sensitivity)
    specs = {
        "MAIN  REV+VGAP, 10y": (["REV", "VGAP"], 10, 1.0),
        "REV only, 10y": (["REV"], 10, 1.0),
        "VGAP only, 10y": (["VGAP"], 10, 1.0),
        "R1+VGAP, 10y": (["R1", "VGAP"], 10, 1.0),
        "REV+VGAP, 5y": (["REV", "VGAP"], 5, 1.0),
        "REV+VGAP, 15y": (["REV", "VGAP"], 15, 1.0),
        "REV+VGAP, 10y, 50% shrink": (["REV", "VGAP"], 10, 0.5),
    }
    log("=== BACKTEST: Friday-close origins, 4-day horizon, 2006 -> last known outcome ===")
    summaries, final_rows, bts = [], [], {}
    o = d.loc[CUTOFF]
    for label, (f, yrs, s) in specs.items():
        bt = backtest(d, f, yrs, s)
        bts[label] = bt
        summaries.append(summarise(bt, label))
        tr = train_set(d, int(o["pos"]), yrs, f)
        b = fit(tr, f)
        mu = s * (b[0] + sum(b[i + 1] * o[x] for i, x in enumerate(f)))
        final_rows.append({"spec": label, "exp_ret_pct": 100 * (np.exp(mu) - 1),
                           "forecast": o["spx"] * np.exp(mu)})
    summ = pd.DataFrame(summaries).set_index("spec")
    sens = pd.DataFrame(final_rows).set_index("spec")
    pd.set_option("display.width", 160)
    log(summ.round(3).to_string())
    log("")
    log("Recent 2 years only (2024-11 -> 2026-10):")
    recent = pd.DataFrame([summarise(bt[bt.index >= "2024-11-01"], k) for k, bt in bts.items()]).set_index("spec")
    log(recent[["n", "MAPE_model", "MAPE_rw", "score_model", "score_rw", "win_rate_vs_rw"]].round(3).to_string())
    log("")
    main_bt = bts["MAIN  REV+VGAP, 10y"]
    hv = main_bt[main_bt["vix"] > 18]
    log(f"High-VIX origins (VIX>18, like now): n={len(hv)}, MAPE model {hv['E_model'].mean():.3f}% "
        f"vs RW {hv['E_rw'].mean():.3f}%")
    worst = main_bt.nlargest(5, "E_model")[["P0", "actual", "F_model", "E_model"]]
    log("Five worst model failures:\n" + worst.round(2).to_string())
    log("")

    # ---- final forecast (main spec), fully itemised
    f, yrs = ["REV", "VGAP"], 10
    tr = train_set(d, int(o["pos"]), yrs, f)
    b = fit(tr, f)
    contrib = {"intercept": b[0], "REV": b[1] * o["REV"], "VGAP": b[2] * o["VGAP"]}
    mu = sum(contrib.values())
    F = o["spx"] * np.exp(mu)
    resid = tr["y"] - (b[0] + b[1] * tr["REV"] + b[2] * tr["VGAP"])
    q10, q90 = np.quantile(resid, [0.10, 0.90])
    sig_vix = o["vix"] / 100 * np.sqrt(H / 252)
    log("=== FINAL FORECAST (main spec) ===")
    log(f"Training sample            : {tr.index[0].date()} -> {tr.index[-1].date()} ({len(tr)} obs)")
    log(f"Baseline P(30 Oct)         : {o['spx']:.2f}")
    log(f"Factor values              : REV = {100*o['REV']:.3f}%   VGAP = {o['VGAP']:.2f} VIX pts "
        f"(VIX {o['vix']:.2f} vs 1y avg {o['vix']-o['VGAP']:.2f})")
    log(f"Coefficients               : a = {b[0]:.6f}   b(REV) = {b[1]:.4f}   c(VGAP) = {b[2]:.6f}")
    log(f"Contributions (log-return) : " + ", ".join(f"{k} {100*v:+.3f}%" for k, v in contrib.items()))
    log(f"Expected 4-day return      : {100*(np.exp(mu)-1):+.3f}%")
    log(f"POINT FORECAST 5 Nov close : {F:.2f}")
    log(f"80% interval (residuals)   : {o['spx']*np.exp(mu+q10):.2f} - {o['spx']*np.exp(mu+q90):.2f}")
    log(f"VIX-implied 1-sigma (4d)   : +/-{100*sig_vix:.2f}%  -> adverse -2 sigma scenario "
        f"{o['spx']*np.exp(mu-2*sig_vix):.2f} (score would be ~{50-min(50, 5*100*(1-np.exp(-2*sig_vix))):.1f}/50)")
    log("")
    log("Sensitivity of the point forecast to specification:")
    log(sens.round(2).to_string())
    log("")

    # ---- midterm event study (context / judgement check, not in the model)
    ms = midterm_study(d)
    log("=== MIDTERM ELECTION WEEK (Fri before -> 4 trading days later), 1950-2022 ===")
    log(ms.round(2).to_string(index=False))
    allr = 100 * (np.exp(d["y"].dropna()) - 1)
    log(f"Midterm weeks: mean {ms['ret_pct'].mean():+.2f}%, median {ms['ret_pct'].median():+.2f}%, "
        f"positive {int((ms['ret_pct']>0).sum())}/{len(ms)} | all 4-day windows since 1950: "
        f"mean {allr.mean():+.2f}%, positive {(allr>0).mean():.0%}")
    # seasonality control: the same early-November window in NON-midterm years
    lp = np.log(d["spx"])
    ctrl = []
    for yr in range(1950, CUTOFF.year):
        if yr % 4 == 2:
            continue
        nov1 = dt.date(yr, 11, 1)
        e = pd.Timestamp(nov1 + dt.timedelta(days=((0 - nov1.weekday()) % 7) + 1))
        i = d.index.get_loc(d.index[d.index < e - pd.Timedelta(days=3)][-1])
        ctrl.append(100 * (np.exp(lp.iloc[i + H] - lp.iloc[i]) - 1))
    ctrl = np.array(ctrl)
    excess = ms["ret_pct"].mean() - ctrl.mean()
    log(f"Control (same window, non-midterm years): n={len(ctrl)}, mean {ctrl.mean():+.2f}%, "
        f"positive {(ctrl>0).mean():.0%}  -> midterm excess {excess:+.2f} pp")
    log("")

    # ---- JUDGEMENT (clearly separated from the model): half-weight midterm excess
    adj = 0.5 * excess / 100
    F_adj = F * (1 + adj)
    log("=== JUDGEMENTAL ADJUSTMENT (separate from model) ===")
    log(f"Midterm excess {excess:+.2f} pp x 0.5 shrink (small sample n={len(ms)}, outcome partly priced) "
        f"= {100*adj:+.2f}%")
    log(f"Model forecast {F:.2f} -> adjusted forecast {F_adj:.2f}")
    log("")

    # ---- save outputs
    summ.to_csv(OUT / "backtest_summary.csv")
    sens.to_csv(OUT / "sensitivity.csv")
    main_bt.to_csv(OUT / "backtest_main_spec.csv")
    ms.to_csv(OUT / "midterm_event_study.csv", index=False)
    record = {"market": "S&P 500 price index (SPX)", "unit": "index points",
              "target": "Official close, Thu 5 Nov 2026, 16:00 New York",
              "information_cutoff": "Fri 30 Oct 2026 close, 16:00 New York",
              "baseline_close_30Oct": round(float(o["spx"]), 2),
              "point_forecast": round(float(F), 2),
              "expected_return_pct": round(100 * (np.exp(mu) - 1), 3),
              "coefficients": {"a": b[0], "b_REV": b[1], "c_VGAP": b[2]},
              "factor_values": {"REV": o["REV"], "VGAP": o["VGAP"], "VIX": o["vix"]},
              "interval_80": [round(float(o["spx"] * np.exp(mu + q10)), 2),
                              round(float(o["spx"] * np.exp(mu + q90)), 2)]}
    (OUT / "forecast_record.json").write_text(json.dumps(record, indent=2, default=float))

    try:
        charts(d, main_bt, ms, F)
    except Exception as e:  # charts are optional
        log(f"(charts skipped: {e})")
    (OUT / "run_log.txt").write_text("\n".join(LOG_LINES))
    print(f"\nSaved results to {OUT}")


def charts(d, bt, ms, F) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    recent = d.loc["2026-07-01":, "spx"]
    ax.plot(recent.index, recent.values, color="#1f3b73", lw=1.6, label="S&P 500 close")
    ax.scatter([TARGET], [F], color="#d1495b", zorder=3, label=f"Forecast 5 Nov: {F:,.2f}")
    ax.axvline(CUTOFF, color="grey", ls="--", lw=1)
    ax.text(CUTOFF, ax.get_ylim()[1], " cutoff 30 Oct", va="top", fontsize=8, color="grey")
    ax.set_title("S&P 500 daily close and point forecast")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_price_forecast.png", dpi=160)

    fig, ax = plt.subplots(figsize=(9, 4))
    cum = (bt["S_model"] - bt["S_rw"]).cumsum()
    ax.plot(cum.index, cum.values, color="#1f3b73", lw=1.4)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Cumulative assignment-score gain of model vs 'no change' (weekly origins)")
    ax.set_ylabel("score points")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_backtest_cum_score.png", dpi=160)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar([str(e.year) for e in pd.to_datetime(ms["election"])], ms["ret_pct"],
           color=np.where(ms["ret_pct"] > 0, "#1f3b73", "#d1495b"))
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("S&P 500 return, Friday before midterm -> 4 trading days later (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_midterm_weeks.png", dpi=160)


if __name__ == "__main__":
    main()
