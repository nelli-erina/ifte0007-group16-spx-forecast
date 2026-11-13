"""
Re-implementation of the member's Ridge + earnings-factor model, corrected to the
practice-round task: data up to Fri 30 Oct 2026 close, forecast the close on Thu 5 Nov
(4 trading days ahead). Same factors and method as the original report, horizon changed.

Approximation: the original used supplied earnings-date files. Here the latest earnings
dates come from the original report's table; earlier 2026 event dates are detected as the
largest market-adjusted absolute return inside each company's usual reporting season.
"""
import numpy as np
import pandas as pd

P = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
CUT = pd.Timestamp("2026-10-30")
H = 4
W = {"AAPL": 7.294, "NVDA": 7.404, "MSFT": 5.614, "AMZN": 3.894, "GOOGL": 5.548 * 0.53,
     "GOOG": 5.548 * 0.47, "AVGO": 2.544, "META": 2.524, "LLY": 1.634, "BRK-B": 1.604, "TSLA": 2.104}
W = {k: v / 100 for k, v in W.items()}
# first tradable event day of the latest events (from the original report's table)
LATEST = {"AAPL": "2026-11-03", "NVDA": "2026-08-27", "MSFT": "2026-10-29", "AMZN": "2026-10-30",
          "GOOGL": "2026-10-29", "GOOG": "2026-10-29", "AVGO": "2026-09-03", "META": "2026-10-22",
          "LLY": "2026-10-29", "BRK-B": "2026-08-10", "TSLA": "2026-10-22"}
# usual reporting seasons for earlier 2026 events
SEASONS = {"default": [("2026-01-20", "2026-02-10"), ("2026-04-20", "2026-05-08"), ("2026-07-20", "2026-08-07")],
           "NVDA": [("2026-02-18", "2026-03-04"), ("2026-05-18", "2026-06-02")],
           "AVGO": [("2026-03-02", "2026-03-16"), ("2026-06-08", "2026-06-19")],
           "BRK-B": [("2026-02-21", "2026-03-04"), ("2026-05-01", "2026-05-12")]}

r = P.pct_change().dropna()
spx = r["^GSPC"]
days = r.index


def loo_mkt(t):
    w = W[t]
    return (spx - w * r[t]) / (1 - w)


def mm_ar(t, tau):
    """market-model abnormal returns, model fitted on 120 days before event start tau"""
    m = loo_mkt(t)
    pos = days.get_loc(tau)
    est = slice(max(0, pos - 125), pos - 5)
    X = np.column_stack([np.ones(len(m.iloc[est])), m.iloc[est]])
    a, b = np.linalg.lstsq(X, r[t].iloc[est], rcond=None)[0]
    return r[t] - (a + b * m)


events = {t: [] for t in W}
for t in W:
    m = loo_mkt(t)
    for s, e in SEASONS.get(t, SEASONS["default"]):
        win = r.loc[s:e]
        if len(win) == 0:
            continue
        adj = (r[t] - m).loc[s:e].abs()
        events[t].append(adj.idxmax())
    events[t].append(pd.Timestamp(LATEST[t]))

F = pd.Series(0.0, index=days)          # daily weighted abnormal contribution
for t, evs in events.items():
    for tau in evs:
        if tau not in days:
            continue
        ar = mm_ar(t, tau)
        p = days.get_loc(tau)
        for d in days[p:p + 3]:
            F[d] += W[t] * ar[d]

lp = P["^GSPC"]
feat = pd.DataFrame({"mom5": lp / lp.shift(5) - 1,
                     "vol20": np.sqrt(252) * r["^GSPC"].rolling(20).std(),
                     "earn": F.rolling(5).sum()}).reindex(P.index)
feat["y"] = lp.shift(-H) / lp - 1
feat = feat[(feat.index >= "2026-03-02") & (feat.index <= CUT)]

LAMS = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0, ddof=0)
    Z = (X - mu) / sd
    g = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return mu, sd, g, y.mean()


def ridge_pred(fit, x):
    mu, sd, g, yb = fit
    return yb + ((x - mu) / sd) @ g


def choose_lambda(X, y):
    n = len(y); best, bl = None, None
    for lam in LAMS:
        errs = []
        for start in range(60, n - 10, 10):
            f = ridge_fit(X[:start], y[:start], lam)
            errs += list(np.abs(ridge_pred(f, X[start:start + 10]) - y[start:start + 10]))
        e = np.mean(errs)
        if best is None or e < best:
            best, bl = e, lam
    return bl


cols = ["mom5", "vol20", "earn"]
known = feat.dropna(subset=cols + ["y"])
# final forecast at 30 Oct: train on rows whose outcome (t+4) is known by 30 Oct
pos_cut = P.index.get_loc(CUT)
train = known[[P.index.get_loc(d) + H <= pos_cut for d in known.index]]
X, y = train[cols].to_numpy(), train["y"].to_numpy()
lam = choose_lambda(X, y)
fit = ridge_fit(X, y, lam)
x0 = feat.loc[CUT, cols].to_numpy(dtype=float)
rhat = float(ridge_pred(fit, x0))
P0, A = lp.loc[CUT], lp.loc["2026-11-05"]
Fc = P0 * (1 + rhat)
E = 100 * abs(Fc - A) / A
print("Events used:", {k: [str(x.date()) for x in v] for k, v in events.items()})
print(f"\nFactors at 30 Oct: mom5 {100*x0[0]:+.2f}% | vol20 {100*x0[1]:.2f}% | earnings {100*x0[2]:+.3f}%")
print(f"Training obs {len(train)} | lambda {lam} | coefs (standardised) {np.round(fit[2]*100,4)} % | mean y {100*fit[3]:+.3f}%")
print(f"Forecast 4-day return {100*rhat:+.3f}%  ->  {Fc:.2f}   (actual {A:.2f}, error {E:.2f}%, score {50-min(50,5*E):.2f})")

# backtest: daily origins Jul 10 .. Oct 26 (outcome known), lambda reselected each origin
rows = []
for d in known.index:
    if d < pd.Timestamp("2026-07-10") or P.index.get_loc(d) + H > pos_cut:
        continue
    p = P.index.get_loc(d)
    tr = known[[P.index.get_loc(x) + H <= p for x in known.index]]
    if len(tr) < 80:
        continue
    Xt, yt = tr[cols].to_numpy(), tr["y"].to_numpy()
    lm = choose_lambda(Xt, yt)
    pr = float(ridge_pred(ridge_fit(Xt, yt, lm), known.loc[d, cols].to_numpy(dtype=float)))
    act = known.loc[d, "y"]
    rows.append((d, lm, pr, act))
bt = pd.DataFrame(rows, columns=["origin", "lambda", "pred", "actual"]).set_index("origin")
e_m = 100 * np.abs((1 + bt.pred) / (1 + bt.actual) - 1)
e_rw = 100 * np.abs(1 / (1 + bt.actual) - 1)
print(f"\nBacktest {len(bt)} daily origins (Jul 10 - Oct 26), 4-day horizon:")
print(f"  Ridge MAPE {e_m.mean():.3f}%  vs  no change {e_rw.mean():.3f}%  | Ridge better in {(e_m<e_rw).mean():.0%} of origins")
print(f"  lambda chosen: {bt['lambda'].value_counts().to_dict()}")
