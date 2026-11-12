"""Three additional robustness checks quoted in the report (Sections 7 and 8).

Usage (from the extras/ folder):
    python robustness_checks.py final_20261112

Reads only the frozen inputs  ../data_<tag>/  and uses the functions of ../spx_forecast_v2.py unchanged.
Writes ../results_<tag>/robustness_checks.json and prints the same numbers.

1. Interval coverage. The report's 80% interval applies the 10th and 90th percentiles of signed log
   backtest errors to the forecast. Here the interval is rebuilt for every backtest origin from EARLIER
   errors only: an error counts as known at an origin when its own target date (origin + H trading days)
   is on or before that origin. At least MIN_HISTORY known errors are required. Coverage is the share of
   outcomes that fell inside the interval, in total and split by the VIX at the origin.
2. Constant shift. The pilot's discretionary adjustment (+0.59%) applied to every backtest forecast:
   share of weeks in which the shifted forecast was closer, and the mean absolute error with and without.
3. Split sample. Mean absolute error of the four specifications in 2006-2015 and 2016-2026.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import spx_forecast_v2 as M  # noqa: E402

MIN_HISTORY = 104          # two years of weekly errors before the first interval is scored
SHIFT = 0.0059             # the pilot adjustment, +0.59%
SPLIT = pd.Timestamp("2016-01-01")

tag = sys.argv[1] if len(sys.argv) > 1 else "final_20261112"
cutoff = pd.Timestamp(f"{tag[-8:-4]}-{tag[-4:-2]}-{tag[-2:]}")
df = M.load(ROOT / f"data_{tag}", cutoff)
d = M.build(df)
bts = {name: M.backtest(d, f) for name, f in M.SPECS.items()}
bt = bts[M.MAIN_SPEC].copy()

# 1. interval coverage from earlier errors only
pos = d["pos"]
bt["pos"] = pos.reindex(bt.index).to_numpy()
bt["err"] = np.log(bt["actual"] / bt["F"])
rows = []
for date, o in bt.iterrows():
    known = bt[bt["pos"] + M.H <= o["pos"]]["err"]
    if len(known) < MIN_HISTORY:
        continue
    lo, hi = np.quantile(known, [0.1, 0.9])
    rows.append({"origin": date, "vix": o["vix"],
                 "inside": bool(o["F"] * np.exp(lo) <= o["actual"] <= o["F"] * np.exp(hi))})
cov = pd.DataFrame(rows).set_index("origin")
coverage = {
    "n": int(len(cov)), "min_history": MIN_HISTORY, "first_origin": str(cov.index[0].date()),
    "all_pct": float(100 * cov["inside"].mean()),
    "vix_le20_n": int((cov["vix"] <= 20).sum()), "vix_le20_pct": float(100 * cov[cov["vix"] <= 20]["inside"].mean()),
    "vix_gt20_n": int((cov["vix"] > 20).sum()), "vix_gt20_pct": float(100 * cov[cov["vix"] > 20]["inside"].mean()),
}

# 2. constant +0.59% shift on every backtest forecast
E_shift = 100 * (bt["F"] * (1 + SHIFT) - bt["actual"]).abs() / bt["actual"]
shift = {"shift_pct": 100 * SHIFT, "n": int(len(bt)), "share_weeks_better_pct": float(100 * (E_shift < bt["E"]).mean()),
         "mae_model": float(bt["E"].mean()), "mae_shifted": float(E_shift.mean())}

# 3. split sample
split = {}
for label, part in [("2006-2015", lambda x: x[x.index < SPLIT]), ("2016-2026", lambda x: x[x.index >= SPLIT])]:
    mae = {name: float(part(b)["E"].mean()) for name, b in bts.items()}
    split[label] = {"n": int(len(part(bt))), "mae_no_change": float(part(bt)["E_rw"].mean()), "mae": mae,
                    "ranking": sorted(mae, key=mae.get), "max_gap_pp": float(max(mae.values()) - min(mae.values()))}
split["same_ranking_in_both_halves"] = split["2006-2015"]["ranking"] == split["2016-2026"]["ranking"]
split["best_in_both_halves"] = [split["2006-2015"]["ranking"][0], split["2016-2026"]["ranking"][0]]

out = {"tag": tag, "interval_coverage": coverage, "constant_shift": shift, "split_sample": split}
(ROOT / f"results_{tag}" / "robustness_checks.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
