"""
KNN model (member's method) re-implemented on the same frozen data, and compared
with our regression model and the no-change benchmark.

Features at day t:  M = ln(P_t / P_{t-5})            (5-day momentum)
                    V = std of daily log returns over the last 20 days
Standardise with mean/sd of the training window (past only), Euclidean distance,
weights w_i = 1/(d_i + 0.05) normalised over the K nearest neighbours,
forecast return = sum w_i * r_i,  r_i = neighbour's h-day-ahead log return.
Neighbours must have their outcome known at the origin (no look-ahead).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CUTOFF = pd.Timestamp("2026-10-30")
s = pd.read_csv(ROOT / "data/gspc.csv", parse_dates=["Date"], index_col="Date")["Close"]
s = s[s.index <= CUTOFF]
lp = np.log(s)
r1 = lp.diff()
feat = pd.DataFrame({"M": lp - lp.shift(5), "V": r1.rolling(20).std()})
pos = np.arange(len(s))


def knn_forecast(o: int, h: int, K: int, years: float = 10, eps: float = 0.05) -> float:
    lo = o - int(years * 252)
    idx = np.arange(max(lo, 25), o - h + 1)            # outcome lp[i+h] known at origin o
    X = feat.iloc[idx].to_numpy()
    y = (lp.shift(-h).iloc[idx] - lp.iloc[idx]).to_numpy()
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    z0 = (feat.iloc[o].to_numpy() - mu) / sd
    d = np.sqrt(((Z - z0) ** 2).sum(1))
    nn = np.argsort(d)[:K]
    w = 1 / (d[nn] + eps)
    return float((w / w.sum()) @ y[nn])


o = len(s) - 1
P0 = s.iloc[-1]
print("=== Reproduce her 1-day forecast for 2 Nov (target 7,578.69) ===")
for K in [3, 5, 10, 15, 20, 30, 50, 100]:
    print(f"K={K:>3}: {P0*np.exp(knn_forecast(o, 1, K)):.2f}")

print("\n=== KNN, 4-day horizon, forecast for Thu 5 Nov ===")
for K in [5, 10, 20, 50, 100, 200]:
    print(f"K={K:>3}: {P0*np.exp(knn_forecast(o, 4, K)):.2f}")

# ---- backtest on the same Friday origins as our model
ours = pd.read_csv(ROOT / "results/backtest_main_spec.csv", parse_dates=["origin"], index_col="origin")
rows = []
for date, row in ours.iterrows():
    oo = s.index.get_loc(date)
    out = {"origin": date}
    for K in [10, 50, 200]:
        F = row["P0"] * np.exp(knn_forecast(oo, 4, K))
        out[f"E_knn{K}"] = 100 * abs(F - row["actual"]) / row["actual"]
    rows.append(out)
bt = ours.join(pd.DataFrame(rows).set_index("origin"))
score = lambda e: (50 - np.minimum(50, 5 * e)).mean()


def table(b, label):
    print(f"\n=== Backtest {label}: n={len(b)} Friday origins, 4-day horizon ===")
    print(f"{'method':<22}{'MAPE %':>8}{'score/50':>10}{'win vs RW':>11}{'t vs RW':>9}")
    for name, col in [("No change (RW)", "E_rw"), ("Our model REV+VGAP", "E_model"),
                      ("KNN K=10", "E_knn10"), ("KNN K=50", "E_knn50"), ("KNN K=200", "E_knn200")]:
        d = b[col] - b["E_rw"]
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if col != "E_rw" else np.nan
        print(f"{name:<22}{b[col].mean():>8.3f}{score(b[col]):>10.2f}"
              f"{(b[col] < b['E_rw']).mean():>11.0%}{t:>9.2f}")


table(bt, "2006-2026")
table(bt[bt.index >= "2024-11-01"], "last 2 years")
bt.to_csv(ROOT / "results/backtest_knn_vs_model.csv")
