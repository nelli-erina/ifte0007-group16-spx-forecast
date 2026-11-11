"""Figures for the Group 16 report, built only from the frozen data and the saved results of spx_forecast_v2.py.

Usage (from the folder that holds the data and results folders):
    python make_figures.py --tag final_20261112 --out figures
    python make_figures.py --tag dryrun_20261110 --out figures      # rehearsal
Reads  data_<tag>/gspc.csv, results_<tag>/backtest_main_spec.csv, results_<tag>/report_numbers.json
Writes fig_backtest.pdf, fig_path.pdf, fig_pilot.pdf, fig_cover_dark.pdf
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "legend.frameon": False})
INK, ACC, GREY = "#1d2b45", "#e07a1f", "#9aa0a8"

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
root = Path(__file__).resolve().parent
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
px = pd.read_csv(root / f"data_{a.tag}/gspc.csv", parse_dates=["Date"], index_col="Date")["Close"]
bt = pd.read_csv(root / f"results_{a.tag}/backtest_main_spec.csv", parse_dates=["origin"], index_col="origin")
num = json.loads((root / f"results_{a.tag}/report_numbers.json").read_text())
F, P0 = num["forecast"], num["P0"]
move = (num.get("options") or {}).get("expected_abs_move_pct") or num["sigma6_pct"]

# Figure: cumulative out-of-sample gain over no change (Welch-Goyal style)
gain = (bt["E_rw"] - bt["E"]).cumsum()
fig, ax = plt.subplots(figsize=(6.3, 2.4))
for s, e in [("2008-09-01", "2009-06-30"), ("2020-02-15", "2020-06-30"), ("2022-01-01", "2022-10-31")]:
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#e8ecf3", lw=0)
ax.fill_between(gain.index, 0, gain.values, where=gain.values > 0, color="#fbe3cc", lw=0)
ax.plot(gain.index, gain.values, color=ACC, lw=1.2); ax.axhline(0, color=GREY, lw=0.6)
ax.set_ylabel("Cumulative gain (pp)")
for x, t in [("2008-11-01", "GFC"), ("2020-04-15", "COVID"), ("2022-06-01", "2022")]:
    ax.text(pd.Timestamp(x), ax.get_ylim()[1] * 0.92, t, ha="center", fontsize=8, color="#555")
fig.tight_layout(); fig.savefig(out / "fig_backtest.pdf"); plt.close(fig)

# Figure: recent path, forecast and options-implied range
h = px.iloc[-65:]; tgt = pd.Timestamp("2026-11-20")
fig, ax = plt.subplots(figsize=(6.3, 2.5))
ax.plot(h.index, h.values, color=INK, lw=1.1)
lo, hi = F * (1 - move / 100), F * (1 + move / 100)
ax.fill_between([h.index[-1], tgt], [P0, lo], [P0, hi], color="#fbe3cc", lw=0)
ax.plot([h.index[-1], tgt], [P0, F], color=ACC, lw=1, ls="--"); ax.plot([tgt], [F], "o", color=ACC, ms=4)
ax.axvline(h.index[-1], color=GREY, lw=0.6, ls=":")
ax.annotate(f"Forecast\n{F:,.0f}", (tgt, F), xytext=(6, -3), textcoords="offset points", va="center", color="#c2610f", fontsize=8)
ax.annotate(f"±{move:.1f}% range", (tgt, hi), xytext=(6, 0), textcoords="offset points", va="center", color="#c2610f", fontsize=8)
ax.text(h.index[-1], ax.get_ylim()[1], " last close", va="top", fontsize=8, color="#555")
ax.set_xlim(h.index[0], tgt + pd.Timedelta(days=16)); ax.set_ylabel("Index level")
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator()); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
fig.tight_layout(); fig.savefig(out / "fig_path.pdf"); plt.close(fig)

# Figure: pilot week (forecasts from extras/results/pilot_models.csv, made with data to 30 Oct)
pm = pd.read_csv(root / "extras/results/pilot_models.csv")
models = pm[pm["method"] != "OLS + event adjustment"]["forecast"]
c = px.loc["2026-10-30":"2026-11-09"]
x = np.arange(len(c)); t = list(c.index).index(pd.Timestamp("2026-11-05"))
fig, ax = plt.subplots(figsize=(6.3, 2.7))
ax.axvspan(t + 0.5, len(c) - 0.5, color="#eef2f8", lw=0)
ax.axhspan(models.min(), models.max(), color="#fbe3cc", lw=0)
ax.axhline(c.iloc[0], color=GREY, lw=0.8, ls="--")
ax.plot(x, c.values, color=INK, lw=1.4, marker="o", ms=3.5); ax.plot([t], [c.iloc[t]], "o", color=ACC, ms=7)
ax.annotate(f"Target day: {c.iloc[t]:,.0f}\n({100*(c.iloc[t]/c.iloc[t-1]-1):+.1f}% that day)", (t, c.iloc[t]),
            xytext=(-8, -2), textcoords="offset points", ha="right", va="center", fontsize=8, color="#b8600f")
ax.annotate(f"Rebound: {c.iloc[-1]:,.0f}", (len(c) - 1, c.iloc[-1]), xytext=(-4, 8), textcoords="offset points", ha="right", fontsize=8)
ax.text(0.1, models.max() + 8, f"All six forecasts ({models.min():,.0f}–{models.max():,.0f})", fontsize=8, color="#b8600f", va="bottom")
ax.set_xticks(x); ax.set_xticklabels([d.strftime("%a %d %b") for d in c.index], fontsize=7.5); ax.set_ylabel("S&P 500 close")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")); ax.set_ylim(7300, 7720)
fig.tight_layout(); fig.savefig(out / "fig_pilot.pdf"); plt.close(fig)

# Cover chart: S&P 500 since 1990
fig, ax = plt.subplots(figsize=(6.2, 2.35)); fig.patch.set_alpha(0); ax.set_facecolor("none")
ax.fill_between(px.index, px.min() * 0.85, px.values, color=ACC, alpha=0.16, lw=0)
ax.plot(px.index, px.values, color="#f39a4a", lw=1.0)
ax.set_yscale("log"); ax.set_ylim(px.min() * 0.85, px.max() * 1.25); ax.set_yticks([])
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
for k in ["left", "right", "top"]: ax.spines[k].set_visible(False)
ax.spines["bottom"].set_color("#3a4a66"); ax.tick_params(axis="x", colors="#8d9bb3", labelsize=7, length=2, color="#3a4a66")
ax.set_xlim(px.index[0], px.index[-1] + pd.Timedelta(days=200)); ax.plot([px.index[-1]], [px.iloc[-1]], "o", color="white", ms=4)
fig.tight_layout(pad=0.1); fig.savefig(out / "fig_cover_dark.pdf", transparent=True); plt.close(fig)
print("figures written to", out)
