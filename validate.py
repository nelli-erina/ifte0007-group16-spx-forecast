"""Automatic checks for spx_forecast_v2.py.

Every check returns (name, passed, detail). CRITICAL checks stop the run if they fail;
WARN checks are written to the log and to results/checks.txt for a human to review.
Run on its own:  python validate.py   (runs the synthetic-data test only)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CRITICAL, WARN = "CRITICAL", "WARN"


# ---------------------------------------------------------------- 1. data checks
def data_checks(df: pd.DataFrame, cutoff: pd.Timestamp, official_close: float | None = None):
    out = []
    out.append((CRITICAL, "no data after cutoff", df.index.max() <= cutoff, f"last date {df.index.max().date()}"))
    out.append((CRITICAL, "last row is the cutoff close", df.index[-1] == cutoff, f"{df.index[-1].date()} vs {cutoff.date()}"))
    out.append((CRITICAL, "no duplicate dates", not df.index.duplicated().any(), f"{int(df.index.duplicated().sum())} duplicates"))
    out.append((CRITICAL, "dates sorted", df.index.is_monotonic_increasing, ""))
    last = df.iloc[-30:]
    out.append((CRITICAL, "no missing values in last 30 rows", not last[["spx", "vix", "tnx"]].isna().any().any(),
                last[["spx", "vix", "tnx"]].isna().sum().to_dict()))
    gaps = df.index.to_series().diff().dt.days
    big = gaps[(gaps > 5) & (df.index >= "1995-01-01")]   # 4-day weekends are normal; >5 days is not (except 9/11)
    out.append((WARN, "no calendar gaps > 5 days since 1995", len(big) <= 1, f"{len(big)} gaps: {[d.date() for d in big.index[:5]]}"))
    r = np.log(df["spx"]).diff()
    jumps = r[r.abs() > 0.10]
    out.append((WARN, "no daily index moves > 10% (other than known crises)", len(jumps) <= 6,
                f"{len(jumps)} days: {[d.date() for d in jumps.index]}"))
    lo, hi = df["spx"].iloc[-1] * 0.8, df["spx"].iloc[-1] * 1.2
    out.append((CRITICAL, "last close within 20% of previous month", df["spx"].iloc[-22:].between(lo, hi).all(), ""))
    out.append((CRITICAL, "VIX in plausible range (5-90)", df["vix"].iloc[-252:].between(5, 90).all(), ""))
    out.append((CRITICAL, "10y yield in plausible range (0-10%)", df["tnx"].iloc[-252:].between(0, 10).all(), ""))
    if official_close is not None:
        diff = abs(df["spx"].iloc[-1] - official_close)
        out.append((CRITICAL, "last close matches S&P DJI official value", diff < 0.05,
                    f"Yahoo {df['spx'].iloc[-1]:.2f} vs S&P DJI {official_close:.2f}"))
    else:
        out.append((WARN, "last close cross-checked against S&P DJI", False, "not done: pass --official-close"))
    return out


# ---------------------------------------------------------------- 2. look-ahead check
def lookahead_check(d: pd.DataFrame, train_fn, H: int, f: list[str]):
    """For a sample of origins, the training set must contain only rows whose H-day outcome
    was already known at the origin (row position + H <= origin position)."""
    bad = 0
    for pos in np.linspace(3000, len(d) - 1, 25).astype(int):
        t = train_fn(d, pos, f)
        if len(t) and (t["pos"].max() + H > pos):
            bad += 1
    return [(CRITICAL, "no look-ahead in training windows (25 sampled origins)", bad == 0, f"{bad} violations")]


# ---------------------------------------------------------------- 3. synthetic-data test
def synthetic_test(fit_fn, seed: int = 0):
    """Generate data where the true coefficients are known and check that the estimator recovers them."""
    rng = np.random.default_rng(seed)
    n, true = 2500, np.array([0.003, -0.04, -0.006])
    X = pd.DataFrame({"REV": rng.normal(0, 0.02, n), "DTNX": rng.normal(0, 0.15, n)})
    X["y"] = true[0] + true[1] * X["REV"] + true[2] * X["DTNX"] + rng.normal(0, 0.0005, n)
    b = fit_fn(X, ["REV", "DTNX"])
    ok = np.allclose(b, true, atol=[2e-4, 3e-3, 4e-4])
    return [(CRITICAL, "estimator recovers known coefficients on synthetic data", ok,
             f"true {true.round(4).tolist()} vs est {np.round(b, 4).tolist()}")]


# ---------------------------------------------------------------- 4. forecast sanity
def forecast_checks(P0: float, F: float, exp_ret_pct: float, coef: list[float]):
    return [
        (CRITICAL, "forecast is a finite number", bool(np.isfinite(F)), f"{F:.2f}"),
        (CRITICAL, "expected 6-day return within +/-3%", abs(exp_ret_pct) < 3, f"{exp_ret_pct:+.3f}%"),
        (WARN, "reversal coefficient has expected sign (<0)", coef[1] < 0, f"{coef[1]:+.4f}"),
        (WARN, "rates coefficient has expected sign (<0)", coef[2] < 0, f"{coef[2]:+.5f}"),
    ]


def report(results, path=None, log=print):
    lines, failed = [], []
    for level, name, ok, detail in results:
        tag = "PASS" if ok else ("FAIL" if level == CRITICAL else "WARN")
        lines.append(f"[{tag:4}] {name}" + (f"  ({detail})" if detail != "" else ""))
        if not ok and level == CRITICAL:
            failed.append(name)
    for l in lines:
        log(l)
    if path is not None:
        path.write_text("\n".join(lines) + "\n")
    if failed:
        raise SystemExit(f"STOPPED: critical check(s) failed: {failed}")


if __name__ == "__main__":
    import spx_forecast_v2 as m
    report(synthetic_test(lambda t, f: m.fit(t, f)))
