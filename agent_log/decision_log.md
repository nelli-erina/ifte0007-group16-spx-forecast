# Agent decision log — Group 16, S&P 500 forecast for 20 November 2026

**Agent:** Claude (Anthropic, Claude Opus 5.5), with web search, a Python sandbox and Yahoo Finance access.
**Human supervisors:** Group 16 (see Contribution Statement).
**Source:** compiled from the agent session transcript (excerpts in `transcript_excerpts.md`); every number links to a file in this ZIP.
**How to read this log:** each row is one decision the agent made. It shows *what* was decided, *why*, *which tool was called with which input*, and *what came out* (a file you can open in this ZIP and the key number in it). Values marked `[final]` are filled in by the frozen run of 13 November.

---

## 1. Decisions

| # | Decision | Why | Tool and input | Output (file → key result) | Human check |
|---|---|---|---|---|---|
| D1 | Use **Yahoo Finance** instead of FRED for prices and yields | FRED requests timed out repeatedly; Yahoo redistributes the same official S&P DJI index levels and has history back to 1990 | Python `yfinance.download(["^GSPC","^TNX","^VIX"], start="1990-01-01")` | `data/gspc.csv`, `tnx.csv`, `vix.csv`; `data_snapshot_meta.json` → source URL, retrieval time (UTC), publication time and last observation of each series | `[to do 13 Nov]` last close checked on the S&P DJI website by: `[name]` |
| D2 | Last usable close = **Thu 12 Nov**; horizon = **6 trading days**; backtest origins on **Thursdays** | Cutoff 12:00 London is 07:00 New York, before the US open; the evaluation should mirror the task | Brief + exchange hours (web search) | `spx_forecast_v2.py`: `H = 6`; check *"last row is the cutoff close"* → PASS | Nelli Erina checked against the brief |
| D3 | Factors **REV + DTNX**; **VIX gap rejected** as a third factor | Brief allows extra factors only if justified; VIX gap made the backtest slightly worse | Python: same backtest for M0–M3, 1,056 Thursday origins 2006–2026 | `backtest_summary.csv` → MAE with VIX gap 1.829% vs 1.824% without; t vs no change −0.50 vs −1.15 | Nelli Erina agreed 11 Nov; `[group to confirm]` |
| D4 | **Main model M3 chosen after the backtest, before the final run** | The brief asks for an economic mechanism; M3 costs < 0.01 pp of accuracy. Reported openly: drift-only M0 is marginally more accurate (1.816% vs 1.825%) and the only significant specification | Code constant `MAIN_SPEC = "M3 reversal + rates"` | `forecast_record.json` → `"model": "M3 reversal + rates"` | Proposed by agent; Nelli Erina agreed 11 Nov; `[group to confirm]` |
| D5 | Use **SPXW** options, not standard SPX, for the market check | SPXW settles on the Friday **close** (our target); monthly SPX settles on the Friday **open** | Python `yfinance.Ticker("^SPX").option_chain("2026-11-20")`, filtered to SPXW | `data_<tag>/spxw_2026-11-20_snapshot.csv` → `report_numbers.json` → implied forward and expected move (`options` block) | `[to do 13 Nov]` snapshot time checked by: `[name]` |
| D6 | **No discretionary adjustment** | In the pilot an event-based adjustment (+0.59%) cost 3.0 points; the Nvidia risk is two-sided | Python: pilot midterm event study (19 elections, 1950–2022) | `pilot/run_log.txt` → adjusted 7,643.97 vs model 7,599.23; actual 7,353.73 | Nelli Erina agreed 11 Nov; `[group to confirm]` |
| D7 | **Overnight futures not used** | Average gain only about +0.2 points over 824 weeks; cannot be backtested at 07:00 NY with free data | Python `extras/extras.py`: ^GSPC open vs previous close, 2010–2026 | `extras/results/overnight.csv` → MAE 1.740% → 1.696%; score +0.2; better in 54% of weeks | Nelli Erina agreed 11 Nov; `[group to confirm]` |
| D8 | **Earnings-factor (Ridge) model not used** for the final forecast | Factor is zero at the cutoff; with vs without the factor makes no measurable difference | Python `extras/extras.py` (runs `alternatives/ridge/ridge_4day.py`), 76 origins, with and without the factor | `extras/results/ridge_with_without.csv` → 1.036% vs 1.026% (t = 0.49) | Nelli Erina agreed 11 Nov; `[group to confirm]` |

## 2. Corrections (errors found and fixed)

| # | What was wrong | How it was found | Fix |
|---|---|---|---|
| C1 | Preliminary 2 Nov close 7,541.56 | Re-download showed a revised value | Use 7,541.40 |
| C2 | Preliminary 5 Nov VIX close 26.9 | Re-download showed a revised value | Use 28.18 |
| C3 | A classmate's KNN forecast targeted 2 Nov (1 day) instead of 5 Nov (4 days) | Agent re-implemented the model and compared horizons | Member re-ran it at the 4-day horizon (7,567.41, `alternatives/members/KNN_member_forecast_5Nov.xlsx`) |
| C4 | Code used model M1 while the report described M3 | Agent compared the script with the report | Code set to M3 (D4); forecasts now identical |
| C5 | Report overstated the earnings factor ("beats no change 47% vs 33%") | Agent re-tested with vs without the factor | Section 6.3 rewritten with Table 7 |
| C6 | Window sensitivity was marked "to be confirmed" | Agent noticed the claim was untested | Script now runs the sensitivity automatically (`sensitivity.csv`) |
| C7 | Agent's own re-implementation of KNN (20-day volatility) and of the historical-mean and AR(1) benchmarks gave different numbers from the members' specifications | User checked the members' files | Report uses the members' own forecasts for the pilot window; only Ridge, built for a different target, is re-estimated |

## 3. Independent checks

| Check | Tool | Result |
|---|---|---|
| Last close vs S&P DJI official value | `--official-close` + `validate.py` | `[final]` (must be PASS, else the run stops) |
| 17 automated data, look-ahead and forecast checks | `validate.py` | `checks_data.txt`, `checks_forecast.txt` → all PASS in dry run |
| Estimator recovers known coefficients on synthetic data | `validate.py` | true −0.04 / −0.006 → estimated −0.0394 / −0.006 |
| Options-implied forward vs model forecast | SPXW put–call parity | `report_numbers.json` → `options`; gap reported in the report, Section 6.4 |
| Reproducibility | offline re-run of the whole package in an empty folder | identical outputs; `SHA256SUMS.txt` in each data/results folder |
| Coefficient significance | Newey–West t-statistics (`report_numbers.json`) | REV and DTNX not individually significant; reported in the report |

## 4. Human approval

The group reviewed the code, the checks and the outputs above and approved the submitted forecast of **`[final]`** index points on `[date, time]` London.

Approved by: Nelli Erina, `[Member 2]`, `[Member 3]`, `[Member 4]`.
