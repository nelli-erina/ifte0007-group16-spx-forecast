# Independent reproduction of the final forecast, 13 November 2026

**Who:** Claude (Anthropic, model Claude Opus 5.5), in the main claude.ai session used for the project, at the request of Nelli Erina.
**When:** 13 November 2026, about 09:05 London (before the 12:00 cutoff).
**What:** the final forecast of 12 November was re-computed from scratch in a fresh clone of this repository, offline, from the frozen inputs only.

## Steps
1. `git clone` of the public repository into an empty folder; `results_final_20261112/` removed.
2. Checksums: `sha256sum -c data_final_20261112/SHA256SUMS.txt` - all five input files OK.
3. `python spx_forecast_v2.py --cutoff 2026-11-12 --no-download --official-close 7592.40`
4. `cd extras && python robustness_checks.py final_20261112 && python extras.py`
5. Byte-by-byte comparison of every output file with the committed version.

## Result
- Point forecast **7,596.27** (identical).
- Identical files: `FORECAST_RECORD.txt`, `forecast_record.json`, `report_numbers.json`, `backtest_main_spec.csv`, `backtest_summary.csv`, `forecasts_by_spec.csv`, `sensitivity.csv`, `checks_data.txt`, `checks_forecast.txt`, `robustness_checks.json`, and all files in `extras/results/`.
- Different only in time stamps: `run_log.txt` (and therefore `SHA256SUMS.txt` of the results folder).
- Validation: 17 of 17 checks passed.

## Data revision check
Yahoo Finance was queried again on 13 November at about 09:05 London. The closes for 3-12 November of ^GSPC, ^TNX and ^VIX are identical to the frozen files (12 November: S&P 500 7,592.40; 10-year yield 5.103; VIX 18.70). No revision; the frozen data were not changed.
