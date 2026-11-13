"""Gain over the no-change forecast by calendar year, split by factor (report Section 5).

Usage (from the extras/ folder):
    python factor_gain_by_year.py final_20261112

Reads only the frozen inputs ../data_<tag>/ and uses the functions of ../spx_forecast_v2.py unchanged.
For each backtest origin the gain is E(no change) - E(model) in percentage points, as in Figure 3.
Four variants are compared: drift only, drift + reversal, drift + rates, and the main model (both factors).
Writes ../results_<tag>/factor_gain_by_year.csv and prints the years 2007-2013.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import spx_forecast_v2 as M  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "final_20261112"
cutoff = pd.Timestamp(f"{tag[-8:-4]}-{tag[-4:-2]}-{tag[-2:]}")
d = M.build(M.load(ROOT / f"data_{tag}", cutoff))

variants = {"drift_only": [], "drift_reversal": ["REV"], "drift_rates": ["DTNX"], "main_both": ["REV", "DTNX"]}
out = {}
for name, f in variants.items():
    bt = M.backtest(d, f)
    out[name] = (bt["E_rw"] - bt["E"]).groupby(bt.index.year).sum()
res = pd.DataFrame(out).round(3)
res.index.name = "year"
res.to_csv(ROOT / f"results_{tag}" / "factor_gain_by_year.csv")
print(res.loc[2007:2013].to_string())
print("totals:", res.sum().round(2).to_dict())
