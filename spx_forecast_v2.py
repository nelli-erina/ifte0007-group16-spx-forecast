"""
S&P 500 forecast for the IFTE0007 group coursework (main task), version 2
=========================================================================

Target      : S&P 500 price index official close, Friday 20 Nov 2026, 16:00 New York
Info cutoff : Friday 13 Nov 2026, 12:00 London (07:00 New York) -> last usable close = Thu 12 Nov
Horizon     : 6 trading days (Thu close -> following Fri close)

Lessons from the practice round (30 Oct -> 5 Nov) built into this version:
  * no discretionary event adjustments in the point forecast;
  * VIX is used for the risk range, not for the centre;
  * the reversal mechanism can fail in sector-wide sell-offs -> reported as a failure condition.

Candidate specifications (all estimated by OLS on a rolling 10-year window, refit at every origin):
  M0  drift only                    y = a
  M1  reversal                      y = a + b*REV
  M2  reversal + fear gap           y = a + b*REV + c*VGAP
  M3  reversal + rates              y = a + b*REV + d*DTNX
where
  y     = ln(P_{t+6}/P_t)                       6-trading-day log return
  REV   = ln(P_t/P_{t-5})                        past-week return
  VGAP  = VIX_t - mean(VIX over previous 252 days)
  DTNX  = TNX_t - TNX_{t-5}                      5-day change in 10-year Treasury yield (pp)

Benchmark: unchanged level. Validation: every Thursday close since 2006, outcome 6 trading days later.

Usage:
  python spx_forecast_v2.py --cutoff 2026-11-12 --official-close X      # final run: downloads and FREEZES data
  python spx_forecast_v2.py --cutoff 2026-11-12 --no-download --official-close X   # reproduce from frozen data
  python spx_forecast_v2.py --cutoff 2026-11-10 --dry-run               # rehearsal on earlier data

Safety: a folder of frozen data is never overwritten unless --force-download is given, and the final
(non-dry) download refuses to run after the 12:00 London information cutoff on 13 Nov 2026.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = 6
WINDOW_YEARS = 10
TEST_START = pd.Timestamp("2006-01-01")
ROOT = Path(__file__).resolve().parent
LOG: list[str] = []


def log(s: str = "") -> None:
    print(s)
    LOG.append(s)


# --------------------------------------------------------------------------- data
def download(data_dir: Path, cutoff: pd.Timestamp) -> None:
    import yfinance as yf
    end = (cutoff + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    meta = {"retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "source": f"Yahoo Finance via yfinance {yf.__version__}", "cutoff_close": str(cutoff.date()),
            "series": {}}
    for tk, fn, start in [("^GSPC", "gspc.csv", "1990-01-01"), ("^VIX", "vix.csv", "1990-01-01"),
                          ("^TNX", "tnx.csv", "1990-01-01")]:
        h = yf.Ticker(tk).history(start=start, end=end, auto_adjust=False, actions=False)
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        h = h[["Close"]].rename_axis("Date")
        h.to_csv(data_dir / fn)
        meta["series"][tk] = {"file": fn, "url": f"https://finance.yahoo.com/quote/{tk}/history",
                              "first": str(h.index[0].date()), "last": str(h.index[-1].date()),
                              "last_close": round(float(h["Close"].iloc[-1]), 4), "rows": len(h),
                              "publication": {"^GSPC": "official close 16:00 New York",
                                              "^VIX": "Cboe close ~16:15 New York",
                                              "^TNX": "Cboe 10y yield index, close ~15:00 Chicago"}[tk]}
    (data_dir / "data_snapshot_meta.json").write_text(json.dumps(meta, indent=2))


def load(data_dir: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    rd = lambda f: pd.read_csv(data_dir / f, parse_dates=["Date"], index_col="Date")["Close"]
    df = pd.concat({"spx": rd("gspc.csv"), "vix": rd("vix.csv"), "tnx": rd("tnx.csv")}, axis=1).sort_index()
    df = df[df.index <= cutoff].dropna(subset=["spx"])
    df[["vix", "tnx"]] = df[["vix", "tnx"]].ffill()   # e.g. Veterans Day: bond market closed
    return df


def build(df: pd.DataFrame, rev_lb: int = 5, dtnx_lb: int = 5) -> pd.DataFrame:
    lp = np.log(df["spx"])
    d = pd.DataFrame(index=df.index)
    d["spx"], d["vix"] = df["spx"], df["vix"]
    d["REV"] = lp - lp.shift(rev_lb)
    d["VGAP"] = df["vix"] - df["vix"].rolling(252).mean().shift(1)
    d["DTNX"] = (df["tnx"] - df["tnx"].shift(dtnx_lb))          # ^TNX is quoted in percent
    d["y"] = lp.shift(-H) - lp
    d["pos"] = np.arange(len(d))
    return d


MAIN_SPEC = "M3 reversal + rates"   # fixed in advance by the group; do not change after seeing results
SPECS = {"M0 drift only": [], "M1 reversal": ["REV"], "M2 reversal + VIX gap": ["REV", "VGAP"],
         "M3 reversal + rates": ["REV", "DTNX"]}


def train(d, o_pos, f, window=None):
    w = WINDOW_YEARS if window is None else window
    t = d[(d["pos"] + H <= o_pos) & (d["pos"] >= o_pos - w * 252)]
    return t.dropna(subset=f + ["y"])


def fit(t, f):
    X = np.column_stack([np.ones(len(t))] + [t[c].to_numpy() for c in f])
    return np.linalg.lstsq(X, t["y"].to_numpy(), rcond=None)[0]


def predict(b, row, f):
    return b[0] + sum(b[i + 1] * row[c] for i, c in enumerate(f))


def backtest(d, f, window=None):
    orig = d[(d.index >= TEST_START) & (d.index.dayofweek == 3)].dropna(subset=f + ["y"])
    rows = []
    for date, o in orig.iterrows():
        t = train(d, int(o["pos"]), f, window)
        if len(t) < 500:
            continue
        mu = predict(fit(t, f), o, f)
        A = o["spx"] * np.exp(o["y"])
        rows.append({"origin": date, "P0": o["spx"], "actual": A, "F": o["spx"] * np.exp(mu), "vix": o["vix"]})
    bt = pd.DataFrame(rows).set_index("origin")
    bt["E"] = 100 * (bt["F"] - bt["actual"]).abs() / bt["actual"]
    bt["E_rw"] = 100 * (bt["P0"] - bt["actual"]).abs() / bt["actual"]
    sc = lambda e: 50 - np.minimum(50, 5 * e)
    bt["S"], bt["S_rw"] = sc(bt["E"]), sc(bt["E_rw"])
    return bt


def nw_se(x: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett) standard error of the mean of x."""
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    v = x @ x / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return float(np.sqrt(v / n))


def summary(bt, label):
    dif = (bt["E"] - bt["E_rw"]).to_numpy()
    return {"spec": label, "n": len(bt), "MAPE": bt["E"].mean(), "MAPE_rw": bt["E_rw"].mean(),
            "score": bt["S"].mean(), "score_rw": bt["S_rw"].mean(),
            "win_vs_rw": (bt["E"] < bt["E_rw"]).mean(),
            "t_vs_rw": dif.mean() / (dif.std(ddof=1) / np.sqrt(len(dif))),
            # 6-day outcomes from weekly origins overlap by about one week -> HAC with 2 lags
            "t_vs_rw_nw": dif.mean() / nw_se(dif, 2)}


def coef_tstats(t, f):
    """OLS coefficients with Newey-West t-stats (lags = H-1, daily overlapping 6-day returns)."""
    X = np.column_stack([np.ones(len(t))] + [t[c].to_numpy() for c in f]); y = t["y"].to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ b; n = len(y)
    XtXi = np.linalg.inv(X.T @ X); S = (X * u[:, None]).T @ (X * u[:, None])
    for k in range(1, H):
        w = 1 - k / H; G = (X[k:] * u[k:, None]).T @ (X[:-k] * u[:-k, None]); S += w * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi)); r2 = 1 - u.var() / y.var()
    return b, b / se, r2


def sensitivity(df: pd.DataFrame, f: list[str]) -> pd.DataFrame:
    """Re-run the backtest and the forecast changing ONE assumption at a time (all else at the main setting)."""
    rows = []
    cases = ([("Estimation window (years)", w, dict(window=w)) for w in (5, 10, 15)]
             + [("Reversal lookback (days)", k, dict(rev_lb=k)) for k in (1, 3, 5, 10, 20)]
             + [("Yield-change lookback (days)", k, dict(dtnx_lb=k)) for k in (5, 10, 20)])
    for name, val, kw in cases:
        d = build(df, kw.get("rev_lb", 5), kw.get("dtnx_lb", 5))
        bt = backtest(d, f, kw.get("window"))
        o = d.iloc[-1]
        b = fit(train(d, int(o["pos"]), f, kw.get("window")), f)
        F = o["spx"] * np.exp(predict(b, o, f))
        s = summary(bt, f"{name}={val}")
        rows.append({"assumption": name, "value": val, "main": (val == (10 if "window" in name else 5)),
                     "MAPE": s["MAPE"], "MAPE_rw": s["MAPE_rw"], "t_vs_rw": s["t_vs_rw"], "forecast": F})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- options (independent check)
def options_check(data_dir: Path, fetch: bool, expiry: str = "2026-11-20"):
    """SPXW (PM-settled, i.e. settles on the Friday CLOSE) contracts expiring on the target date.
    fetch=True: download the chain and freeze it in data_dir. fetch=False: recompute from the frozen file."""
    snap = data_dir / f"spxw_{expiry}_snapshot.csv"
    if fetch:
        import yfinance as yf
        oc = yf.Ticker("^SPX").option_chain(expiry)
        c = oc.calls[["contractSymbol", "strike", "lastPrice", "bid", "ask", "lastTradeDate"]]
        p = oc.puts[["contractSymbol", "strike", "lastPrice", "bid", "ask", "lastTradeDate"]]
        c, p = c[c.contractSymbol.str.startswith("SPXW")], p[p.contractSymbol.str.startswith("SPXW")]
        m = c.merge(p, on="strike", suffixes=("_c", "_p"))
        m["retrieved_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        m.to_csv(snap, index=False)
    m = pd.read_csv(snap)
    mid = lambda r, s: (r[f"bid_{s}"] + r[f"ask_{s}"]) / 2 if r[f"bid_{s}"] > 0 and r[f"ask_{s}"] > 0 else r[f"lastPrice_{s}"]
    m["C"] = m.apply(lambda r: mid(r, "c"), axis=1)
    m["P"] = m.apply(lambda r: mid(r, "p"), axis=1)
    m["fwd"] = m["strike"] + m["C"] - m["P"]
    m["absdiff"] = (m["C"] - m["P"]).abs()
    near = m.nsmallest(7, "absdiff")
    fwd = float(near["fwd"].median())
    atm = m.iloc[(m["strike"] - fwd).abs().argsort()[:1]].iloc[0]
    straddle = float(atm["C"] + atm["P"])
    return {"expiry": expiry, "contracts": "SPXW (PM-settled)", "implied_forward": round(fwd, 2),
            "atm_strike": float(atm["strike"]), "atm_straddle": round(straddle, 2),
            "expected_abs_move_pct": round(100 * straddle / fwd, 2),
            "quotes_last_trade": str(m["lastTradeDate_c"].max()), "snapshot_retrieved_utc": str(m["retrieved_utc"].iloc[0]) if "retrieved_utc" in m else None,
            "snapshot_file": snap.name}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True, help="last usable close (YYYY-MM-DD)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-options", action="store_true")
    ap.add_argument("--force-download", action="store_true", help="overwrite an existing frozen data folder")
    ap.add_argument("--official-close", type=float, default=None,
                    help="last close as published by S&P DJI, typed in by hand for the cross-check")
    a = ap.parse_args()
    cutoff = pd.Timestamp(a.cutoff)
    tag = ("dryrun_" if a.dry_run else "final_") + cutoff.strftime("%Y%m%d")
    data_dir, out_dir = ROOT / f"data_{tag}", ROOT / f"results_{tag}"
    data_dir.mkdir(exist_ok=True); out_dir.mkdir(exist_ok=True)

    log(f"RUN {tag} at {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} UTC | "
        f"python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}")
    if not a.no_download:
        if (data_dir / "gspc.csv").exists() and not a.force_download:
            raise SystemExit(f"STOPPED: {data_dir.name} already holds frozen data. Reproduce with --no-download, "
                             "or pass --force-download to overwrite it deliberately.")
        if not a.dry_run:
            now_london = pd.Timestamp.now(tz="Europe/London")
            deadline = pd.Timestamp("2026-11-13 12:00", tz="Europe/London")
            earliest = pd.Timestamp(cutoff.date()).tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=30)
            if not (earliest <= now_london <= deadline):
                raise SystemExit(f"STOPPED: final download must happen between the {cutoff.date()} close and "
                                 f"12:00 London on 13 Nov (now {now_london:%Y-%m-%d %H:%M} London).")
        download(data_dir, cutoff)
    df = load(data_dir, cutoff)
    d = build(df)
    import validate as V
    log("=== CHECKS: data, look-ahead, synthetic test ===")
    V.report(V.data_checks(df, cutoff, a.official_close) + V.lookahead_check(d, train, H, SPECS[MAIN_SPEC])
             + V.synthetic_test(fit), out_dir / "checks_data.txt", log)
    log("")
    o = d.iloc[-1]
    log(f"Baseline close {cutoff.date()}: {o['spx']:.2f} | VIX {o['vix']:.2f} | REV {100*o['REV']:+.2f}% | "
        f"VGAP {o['VGAP']:+.2f} | DTNX {o['DTNX']:+.3f} pp")
    log("")

    log(f"=== BACKTEST: Thursday origins since 2006, {H}-trading-day horizon, rolling {WINDOW_YEARS}y OLS ===")
    summ, fc, bts = [], [], {}
    for name, f in SPECS.items():
        bt = backtest(d, f); bts[name] = bt; summ.append(summary(bt, name))
        t = train(d, int(o["pos"]), f); b = fit(t, f); mu = predict(b, o, f)
        fc.append({"spec": name, "coef": [round(x, 6) for x in b], "exp_ret_pct": 100 * (np.exp(mu) - 1),
                   "forecast": o["spx"] * np.exp(mu)})
    S = pd.DataFrame(summ).set_index("spec"); FC = pd.DataFrame(fc).set_index("spec")
    pd.set_option("display.width", 170)
    log(S.round(3).to_string()); log("")
    rec = pd.DataFrame([summary(bt[bt.index >= "2024-11-01"], k) for k, bt in bts.items()]).set_index("spec")
    log("Last 2 years:"); log(rec[["n", "MAPE", "MAPE_rw", "score", "score_rw", "win_vs_rw"]].round(3).to_string()); log("")
    log("Forecasts by specification:"); log(FC.round(4).to_string()); log("")

    main_spec = MAIN_SPEC
    F = float(FC.loc[main_spec, "forecast"])
    log("=== CHECKS: forecast sanity ===")
    V.report(V.forecast_checks(o["spx"], F, float(FC.loc[main_spec, "exp_ret_pct"]), FC.loc[main_spec, "coef"]),
             out_dir / "checks_forecast.txt", log)
    log("")
    bt = bts[main_spec]
    resid_q = np.quantile(np.log(bt["actual"] / bt["F"]), [0.1, 0.9])
    sig = o["vix"] / 100 * np.sqrt(H / 252)
    log(f"=== MAIN SPEC ({main_spec}) ===")
    log(f"Coefficients (alpha, b_REV, b_DTNX): {FC.loc[main_spec, 'coef']}")
    log(f"Expected {H}-day return {FC.loc[main_spec, 'exp_ret_pct']:+.3f}%  ->  POINT FORECAST {F:.2f}")
    log(f"80% range from backtest errors: {F*np.exp(resid_q[0]):.2f} - {F*np.exp(resid_q[1]):.2f}")
    log(f"VIX-implied 1-sigma over {H} days: +/-{100*sig:.2f}%; adverse -2 sigma: {o['spx']*np.exp(-2*sig):.2f} "
        f"(score ~{50-min(50, 5*100*(np.exp(2*sig)-1)):.1f})")
    worst = bt.nlargest(5, "E")[["P0", "actual", "F", "E"]]
    log("Worst historical misses:\n" + worst.round(2).to_string())
    hv = bt[bt["vix"] > 20]
    log(f"When VIX>20 at origin: n={len(hv)}, MAPE {hv['E'].mean():.2f}% vs RW {hv['E_rw'].mean():.2f}%")
    log("")

    log("=== SENSITIVITY: one assumption changed at a time ===")
    SEN = sensitivity(df, SPECS[main_spec])
    SEN["dev_from_main_pts"] = SEN["forecast"] - F
    log(SEN.round(3).to_string(index=False)); log("")
    SEN.to_csv(out_dir / "sensitivity.csv", index=False)

    opt = None
    if not a.no_options and (not a.no_download or (data_dir / "spxw_2026-11-20_snapshot.csv").exists()):
        try:
            opt = options_check(data_dir, fetch=not a.no_download)
            log(f"=== OPTIONS CHECK (SPXW {opt['expiry']}) ===")
            log(json.dumps(opt, indent=2))
        except Exception as e:
            log(f"(options check failed: {e})")

    # ---------------- every number quoted in the report, in one file (read by the report build)
    fm = SPECS[main_spec]
    b_main, t_main, r2_main = coef_tstats(train(d, int(o["pos"]), fm), fm)
    over_time = []
    for end in ["2008-12-31", "2012-12-31", "2016-12-30", "2020-12-31", "2022-12-30", str(cutoff.date())]:
        p_ = int(d.loc[:end].iloc[-1]["pos"]); bb, tt, rr = coef_tstats(train(d, p_, fm), fm)
        over_time.append({"window_end": end[:4], "b_rev": bb[1], "t_rev": tt[1], "b_dtnx": bb[2], "t_dtnx": tt[2], "r2_pct": 100 * rr})
    P0 = float(o["spx"])
    contrib = {"drift": P0 * b_main[0], "rev": P0 * b_main[1] * o["REV"], "dtnx": P0 * b_main[2] * o["DTNX"]}
    sens_main = SEN.set_index(["assumption", "value"])["dev_from_main_pts"]
    numbers = {
        "data_to": str(cutoff.date()), "status": "dry run" if a.dry_run else "final",
        "P0": P0, "vix": float(o["vix"]), "rev_pct": 100 * float(o["REV"]), "dtnx_pp": float(o["DTNX"]),
        "alpha_pct": 100 * b_main[0], "b_rev": b_main[1], "b_dtnx": b_main[2],
        "t_alpha": t_main[0], "t_rev": t_main[1], "t_dtnx": t_main[2], "r2_pct": 100 * r2_main,
        "contrib_pts": contrib, "exp_ret_pct": float(FC.loc[main_spec, "exp_ret_pct"]), "forecast": F,
        "pi80": [F * np.exp(resid_q[0]), F * np.exp(resid_q[1])],
        "sigma6_pct": 100 * sig, "adverse_level": P0 * np.exp(-2 * sig),
        "backtest": S[["n", "MAPE", "MAPE_rw", "t_vs_rw", "t_vs_rw_nw", "score", "win_vs_rw"]].reset_index().to_dict("records"),
        "vix_gt20": {"n": int(len(hv)), "mae": float(hv["E"].mean()), "mae_rw": float(hv["E_rw"].mean())},
        "worst": [{"origin": str(i.date()), "error_pct": float(r.E)} for i, r in worst.iterrows()],
        "median_error_pct": float(bt["E"].median()),
        "sensitivity": SEN.to_dict("records"),
        "sens_window_max_pts": float(sens_main.loc["Estimation window (years)"].abs().max()),
        "sens_rev_3_20_max_pts": float(sens_main.loc["Reversal lookback (days)"].drop(1).abs().max()),
        "sens_rev_1_pts": float(sens_main.loc[("Reversal lookback (days)", 1)]),
        "sens_dtnx_max_pts": float(sens_main.loc["Yield-change lookback (days)"].abs().max()),
        "coef_over_time": over_time,
        "options": opt,
    }
    (out_dir / "report_numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
    log(f"Coefficient t-stats (Newey-West): alpha {t_main[0]:.2f}, REV {t_main[1]:.2f}, DTNX {t_main[2]:.2f}; R2 {100*r2_main:.2f}%")

    record = {"status": "DRY RUN - not the submitted forecast" if a.dry_run else "FINAL",
              "market": "S&P 500 price index", "unit": "index points",
              "target": "Official S&P DJI close, Fri 20 Nov 2026, 16:00 New York",
              "information_cutoff": "Fri 13 Nov 2026 12:00 London; last usable close " + str(cutoff.date()),
              "baseline_close": round(float(o["spx"]), 2), "model": main_spec,
              "coefficients": FC.loc[main_spec, "coef"],
              "expected_return_pct": round(float(FC.loc[main_spec, "exp_ret_pct"]), 4),
              "point_forecast": round(F, 2), "options_check": opt,
              "official_close_used_for_check": a.official_close,
              "reproduce_with": f"python spx_forecast_v2.py --cutoff {cutoff.date()} --no-download"
                                + (" --dry-run" if a.dry_run else "")
                                + (f" --official-close {a.official_close}" if a.official_close else "")}
    (out_dir / "forecast_record.json").write_text(json.dumps(record, indent=2, default=float))
    (out_dir / "FORECAST_RECORD.txt").write_text(
        f"FINAL FORECAST RECORD - Group 16 ({record['status']})\n"
        f"Market:              {record['market']}\n"
        f"Unit:                {record['unit']}\n"
        f"Target:              {record['target']}\n"
        f"Information cutoff:  {record['information_cutoff']}\n"
        f"Baseline close:      {record['baseline_close']:.2f}\n"
        f"Model:               {record['model']}\n"
        f"Expected return:     {record['expected_return_pct']:+.3f}%\n"
        f"POINT FORECAST:      {record['point_forecast']:.2f}\n"
        f"Software:            python {sys.version.split()[0]}, pandas {pd.__version__}, numpy {np.__version__}\n")
    S.to_csv(out_dir / "backtest_summary.csv"); FC.to_csv(out_dir / "forecasts_by_spec.csv")
    bt.to_csv(out_dir / "backtest_main_spec.csv")
    (out_dir / "run_log.txt").write_text("\n".join(LOG))
    write_checksums(data_dir); write_checksums(out_dir)
    print(f"\nSaved to {out_dir}")


def write_checksums(folder: Path) -> None:
    """Freeze proof: SHA-256 of every file in `folder`, saved as SHA256SUMS.txt.
    Anyone can re-check with:  sha256sum -c SHA256SUMS.txt  (run inside the folder)."""
    lines = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.name != "SHA256SUMS.txt":
            lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
    (folder / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
