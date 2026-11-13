# Group 16 — S&P 500 forecast for 20 November 2026: code and evidence

IFTE0007 Decentralised Finance and Blockchain, group coursework.
Authors: Nelli Erina, Yaxi Gu, Chloe Wei Wu, Taoyan Xia (University College London).

## Final forecast record

See `results_<tag>/FORECAST_RECORD.txt` (also `forecast_record.json`). `<tag>` is `final_20261112` for the submitted forecast; `dryrun_20261110` is the rehearsal on data to 10 November.

| Item | Value |
|---|---|
| Market | S&P 500 price index |
| Unit | index points |
| Target | Official S&P Dow Jones Indices close, Friday 20 November 2026, 16:00 New York |
| Information cutoff | 12:00 London, Friday 13 November 2026 (last usable close: Thursday 12 November) |
| Target time | Friday 20 November 2026, 4:00 pm New York (EST) = 9:00 pm London (21:00 UTC) |
| Point forecast | see `FORECAST_RECORD.txt` (two decimal places) |

## Contents

| Path | Contents |
|---|---|
| `spx_forecast_v2.py` | Main calculation: data download and freeze, factors, rolling OLS, backtest with Newey–West tests, sensitivity, options check, forecast record, `report_numbers.json` |
| `validate.py` | 17 automated checks (data, look-ahead, synthetic-data test, forecast sanity); a critical failure stops the run |
| `make_figures.py` | Figures in the report, built from the saved data and results |
| `make_report_numbers.py` | Writes `numbers.tex`, the file from which the report takes every run-dependent number |
| `requirements.txt` | Software versions |
| `data_<tag>/` | **Frozen inputs**: `gspc.csv`, `vix.csv`, `tnx.csv`, `spxw_2026-11-20_snapshot.csv`, `data_snapshot_meta.json` (source URL, retrieval time, publication time, last observation), `SHA256SUMS.txt` |
| `results_<tag>/` | Outputs: forecast record, backtest, sensitivity, `report_numbers.json`, check reports, run log, `SHA256SUMS.txt` |
| `extras/` | `extras.py` and its frozen inputs (`extras/data/`) and outputs (`extras/results/`): pilot-window comparison of all methods, Ridge with/without the earnings factor, market choice, overnight-move test, tail events |
| `extras/robustness_checks.py` | Interval coverage from earlier errors only, the pilot's +0.59% shift applied to every backtest week, and the split-sample comparison of the four specifications; writes `results_<tag>/robustness_checks.json` |
| `extras/factor_gain_by_year.py` | Gain over no change by calendar year for drift only, drift + reversal, drift + rates and the main model (the 2009 and 2012 explanation in Section 5); writes `results_<tag>/factor_gain_by_year.csv` |
| `alternatives/` | Re-implementations of other group methods (KNN, Ridge) and the pilot-round script with its data and results |
| `figures/` | Figures used in the report |
| `agent_log/` | `agent_instructions.md` (goal, data boundary, permitted tools), `decision_log.md` (decisions, tool inputs and outputs, corrections, checks, approval), `transcript_excerpts.md` |

## How to reproduce (no account, network or AI subscription needed)

Clone or download this repository (Code → Download ZIP), then:

1. Install Python 3.11+ and: `pip install numpy pandas matplotlib`  (yfinance is only needed to download, not to reproduce).
2. **First** check that the frozen inputs are unchanged:
   ```
   cd data_<tag> && sha256sum -c SHA256SUMS.txt && cd ..
   ```
   Every line should say OK. (Windows PowerShell: `Get-FileHash <file> -Algorithm SHA256`.)
3. Re-run the main calculation from the frozen inputs:
   ```
   python spx_forecast_v2.py --cutoff 2026-11-12 --no-download --official-close <value>
   ```
   The exact command is stored in `results_<tag>/forecast_record.json` under `reproduce_with`. About 20 seconds.
   Without `--no-download` the script refuses to touch an existing data folder, so the frozen inputs cannot be overwritten by mistake.
4. Re-run the supporting calculations and figures:
   ```
   cd extras && python extras.py <tag> && python robustness_checks.py <tag> && python factor_gain_by_year.py <tag> && cd ..
   python make_figures.py --tag <tag> --out figures
   python make_report_numbers.py --tag <tag> --out numbers.tex
   ```
5. Compare with the shipped files: all outputs are identical; only `run_log.txt` (and therefore its line in `SHA256SUMS.txt`) differs, because it records the time of the run.

## Software

Python 3.13.18; numpy 2.5.3; pandas 3.0.5; matplotlib 3.11.2; yfinance 1.7.0 (download only). Versions are also written at the top of every `run_log.txt` and in `FORECAST_RECORD.txt`.

## Data notes

- Prices and yields are from Yahoo Finance, which redistributes the official index levels. The 12 November close was checked by hand against S&P Dow Jones Indices (`--official-close`; result in `checks_data.txt`).
- The SPXW option snapshot was taken before 12:00 London on 13 November; its retrieval time is in the file and in `report_numbers.json`.
- Data files are included only so that the marker can reproduce the calculation; they remain subject to the providers' terms of use.
- No credentials, API keys or personal data are included. No paid data or trading account was used.
