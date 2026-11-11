"""Write numbers.tex: every run-dependent number quoted in the report, as LaTeX macros.

Usage:  python make_report_numbers.py --tag final_20261112 --out numbers.tex
Reads results_<tag>/report_numbers.json and extras/results/*.csv, so the report text, tables and the
code outputs can never disagree. In a dry run every value is shown in a grey box.
"""
import argparse, json
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
root = Path(__file__).resolve().parent
n = json.loads((root / f"results_{a.tag}/report_numbers.json").read_text())
ex = root / "extras/results"
dry = n["status"] != "final"
M = {}
def f(x, d=2, sign=False, comma=True):
    s = f"{x:{'+' if sign else ''}{',' if comma else ''}.{d}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")
def put(name, val):
    M[name] = val
score = lambda e: 50 - min(50, 5 * e)

date = pd.Timestamp(n["data_to"])
put("dataTo", f"{date.day} {date:%B}")
put("Pzero", f(n["P0"])); put("forecast", f(n["forecast"])); put("forecastR", f(n["forecast"], 1))
put("expRet", f(n["exp_ret_pct"], 2, sign=True))
put("alpha", f(n["alpha_pct"], 3)); put("brev", f(n["b_rev"], 3)); put("bdtnx", f(n["b_dtnx"], 4))
put("talpha", f(n["t_alpha"])); put("trev", f(n["t_rev"])); put("tdtnx", f(n["t_dtnx"])); put("rtwo", f(n["r2_pct"]))
put("rev", f(n["rev_pct"], 2, sign=True)); put("dtnx", f(n["dtnx_pp"], 3, sign=True))
c = n["contrib_pts"]
put("cDrift", f(c["drift"], 1, sign=True)); put("cRev", f(c["rev"], 1, sign=True)); put("cDtnx", f(c["dtnx"], 1, sign=True))
put("cFactors", f(c["rev"] + c["dtnx"], 1, sign=True))
put("piLo", f(n["pi80"][0], 0)); put("piHi", f(n["pi80"][1], 0))
put("sigma", f(n["sigma6_pct"], 1))
adv_err = 100 * abs(n["forecast"] - n["adverse_level"]) / n["adverse_level"]
put("adverseMove", f(100 * (n["adverse_level"] / n["P0"] - 1), 1, sign=True)); put("adverseErr", f(adv_err, 1)); put("adverseScore", f(score(adv_err), 0))
put("medErr", f(n["median_error_pct"], 1)); put("medScore", f(score(n["median_error_pct"]), 1))
bt = {r["spec"][:2]: r for r in n["backtest"]}
put("maeRW", f(bt["M3"]["MAPE_rw"], 3))
for k, L in [("M0", "A"), ("M1", "B"), ("M2", "C"), ("M3", "D")]:
    put("mae" + L, f(bt[k]["MAPE"], 3)); put("t" + L, f(bt[k]["t_vs_rw"])); put("tnw" + L, f(bt[k]["t_vs_rw_nw"])); put("sc" + L, f(bt[k]["score"]))
put("nOrigins", f"{bt['M3']['n']:,}")
put("vixN", str(n["vix_gt20"]["n"])); put("vixMae", f(n["vix_gt20"]["mae"])); put("vixRw", f(n["vix_gt20"]["mae_rw"]))
w = n["worst"]
put("worst", "; ".join(f"{pd.Timestamp(x['origin']):%B %Y}, {x['error_pct']:.0f}\\%" for x in w[:3]))
# sensitivity
put("sensWin", f(n["sens_window_max_pts"], 1)); put("sensRev", f(n["sens_rev_3_20_max_pts"], 1))
put("sensRevOne", f(n["sens_rev_1_pts"], 1, sign=True)); put("sensDtnx", f(n["sens_dtnx_max_pts"], 1))
rows, last = [], None
for r in n["sensitivity"]:
    lab = {"Estimation window (years)": "Estimation window", "Reversal lookback (days)": "Reversal lookback",
           "Yield-change lookback (days)": "Yield-change lookback"}[r["assumption"]]
    unit = " years" if "window" in r["assumption"] else (" day" if r["value"] == 1 else " days")
    main = r["main"]
    if r["assumption"] != last and last is not None:
        rows.append("\\midrule")
    first = lab if r["assumption"] != last else ""
    v = f"{r['value']}{unit}" + (" (main)" if main else "")
    cells = [first, ("\\textbf{%s}" % v) if main else v, f(r["MAPE"], 3) + "\\%", f(r["t_vs_rw"]), "\\RV{%s}" % f(r["forecast"] - n["forecast"], 1, sign=True)]
    rows.append(("\\best " if main else "") + " & ".join(cells) + " \\\\")
    last = r["assumption"]
put("sensRows", "\n".join(rows))
# coefficients over time
ot = n["coef_over_time"]
put("coefHead", " & ".join("\\thead{%s}" % x["window_end"] for x in ot))
put("coefRev", " & ".join(f(x["b_rev"], 3) for x in ot)); put("coefRevT", " & ".join("(" + f(x["t_rev"]) + ")" for x in ot))
put("coefDtnx", " & ".join(("\\textcolor{accd}{%s}" if x["b_dtnx"] > 0 else "%s") % f(x["b_dtnx"], 4) for x in ot))
put("coefDtnxT", " & ".join("(" + f(x["t_dtnx"]) + ")" for x in ot)); put("coefRtwo", " & ".join(f(x["r2_pct"]) + "\\%" for x in ot))
put("revFirst", f(ot[0]["b_rev"], 3)); put("revLast", f(ot[-1]["b_rev"], 3))
# options
o = n.get("options") or {}
if o:
    put("optFwd", f(o["implied_forward"])); put("optGap", f(100 * (o["implied_forward"] / n["forecast"] - 1), 1, sign=True))
    put("optMove", f(o["expected_abs_move_pct"], 1)); put("optAbove", f(100 * (o["implied_forward"] / n["P0"] - 1), 1, sign=True))
    put("optTime", pd.Timestamp(o["snapshot_retrieved_utc"]).tz_convert("Europe/London").strftime("%H:%M London, %d %B"))
    put("optMoveScore", f(score(o["expected_abs_move_pct"]), 1))
# scenarios (Nvidia weight from the Ridge re-implementation's index weights)
nv = 7.404
put("nvWeight", f(nv, 0)); put("nvMove", f(nv * 0.10, 1)); put("nvLoss", f(5 * nv * 0.10, 1))
# extras
pm = pd.read_csv(ex / "pilot_models.csv")
pr = []
for _, r in pm.iterrows():
    if r["method"] == "OLS + event adjustment":
        pr.append("\\midrule")
    cells = [r["method"].replace("k-nearest", "$k$-nearest"), r["type"], str(r["factors"]).replace("-", "---") if r["factors"] == "-" else str(r["factors"]).replace("%", "\\%"),
             f(r["forecast"]), f(r["error_pct"]) + "\\%", f(r["score"])]
    if r["method"] == "OLS + event adjustment":
        cells[0] += "$^{a}$"
    if r["method"] == "Ridge regression":
        cells[0] += "$^{b}$"
    pr.append(("\\best " if r["method"] == "OLS regression" else "") + " & ".join(cells) + " \\\\")
put("pilotRows", "\n".join(pr))
mods = pm[pm.method != "OLS + event adjustment"].forecast
put("pilotLo", f(mods.min(), 0)); put("pilotHi", f(mods.max(), 0)); put("pilotSpread", f(100 * (mods.max() / mods.min() - 1), 2))
rw = pd.read_csv(ex / "ridge_with_without.csv").set_index("model")
for key, nm in [("Ridge with earnings factor", "W"), ("Ridge without earnings factor", "O"), ("No change", "N")]:
    for col, cn in [("all", "all"), ("active", "act"), ("zero", "zero")]:
        put(f"rg{nm}{cn}", f(rw.loc[key, col], 3))
    if nm != "N":
        put(f"rg{nm}better", f(100 * rw.loc[key, "better_than_no_change"], 0))
put("rgCntAll", f(rw.loc["n origins", "all"], 0, comma=False)); put("rgCntAct", f(rw.loc["n origins", "active"], 0)); put("rgCntZero", f(rw.loc["n origins", "zero"], 0))
put("rgDiff", f(rw.loc["with minus without (pp), t", "all"], 3, sign=True)); put("rgT", f(rw.loc["with minus without (pp), t", "active"]))
mc = pd.read_csv(ex / "market_choice.csv").set_index("market")["median_abs_move_pct"]
put("moveSPX", f(mc["S&P 500"], 1)); put("moveBTC", f(mc["Bitcoin"], 1)); put("moveETH", f(mc["Ethereum"], 1))
on = pd.read_csv(ex / "overnight.csv").iloc[0]
put("onWeeks", f"{int(on.weeks):,}"); put("onGain", f(on.score_with_overnight - on.score_no_change, 1)); put("onBetter", f(100 * on.share_weeks_better, 0))
tail = pd.read_csv(ex / "tail_events.csv")
put("tailN", str(len(tail)))

with open(a.out, "w") as fh:
    fh.write(f"% generated by make_report_numbers.py from results_{a.tag} -- do not edit by hand\n")
    fh.write("\\newcommand{\\RV}[1]{%s}\n" % ("\\ph{#1}" if dry else "#1"))
    fh.write("\\newcommand{\\RunStatus}{%s}\n" % ("dry run on data to " + M["dataTo"] if dry else "final run on data to " + M["dataTo"]))
    for k, v in M.items():
        raw = k.endswith("Rows") or k.startswith("coef") or k in ("worst",)
        fh.write("\\newcommand{\\N%s}{%s}\n" % (k, v if raw else "\\RV{%s}" % v))
print(f"wrote {len(M)} macros to {a.out}")
