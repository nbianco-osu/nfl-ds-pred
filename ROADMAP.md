# NFL Predictor Roadmap

Updated: 2026-10-13. Keep this file current in each iteration. Record evidence,
limitations, and the next acceptance check before marking an item complete.

## Current Iteration

| ID | Priority | Status | Scope and acceptance |
| --- | --- | --- | --- |
| R1 | 1 | Implemented and verified | Weekly expanding-window baseline backtest; fixed settings, strictly earlier training dates, fold manifest, source hash, per-game exports, separate research results. Initial 2023-2025 run: 815 games in 54 folds. |
| R2 | 2 | Implemented and verified | Observed archive dashboard: model-version filter, weekly winner accuracy/log loss/Brier, calibration counts, ensemble score MAE, spread/total counts and exclusions. Automatically exported on refresh. |
| R3 | 3 | Implemented and verified | Default to earliest unfinished week, compact team-logo matchups, expected scores, accessible game details, correctness labels, existing detailed table retained. |

## Next Options

| ID | Priority | Status | Acceptance check |
| --- | --- | --- | --- |
| R4 | High | Planned | Audit feature availability and leakage; timestamp every original input and winner forecast. Never reconstruct old forecasts from final results. |
| R5 | High | Planned | Extend rolling evaluation to advanced production candidates with nested time-based tuning; compare Elo and market-informed benchmarks separately. Include confidence intervals and regular-season/tie-complete coverage. |
| R6 | High | Planned | Joint score/margin/total distributions; validate interval coverage, score coherence, and cover/over/push probabilities on unseen weeks. No arbitrary probability noise or win cap. |
| R7 | High | Planned | Per-game source freshness: QB certainty, injury/roster coverage, forecast weather, opponent-adjusted EPA, pace and OL availability; ablation test each addition. |
| R8 | Medium | Planned | Challenger review: version-isolated prospective cohorts, minimum evidence threshold, explicit human promotion; no automatic production replacement. |
| R9 | Medium | Partial | Expanded matchup detail with local explanations and input provenance. Current iteration adds score, line and model-version details only. |
| R10 | Medium | Planned | Compare individual score models in observed performance, with independent score-version filters and calibration plots with uncertainty. |
| R11 | Medium | Partial | Reproducible validation-gated publishing, visible last successful refresh, stale-data alerts, atomic data writes and scheduled CI tests. Existing weekly automation remains unchanged. |

## Evidence and Boundaries

- `rolling_backtest.py` is a research baseline harness, not a refit or evaluation of the deployed Random Forest. Production model files and archived picks are unchanged.
- Historical CSV inputs are revised snapshots and omit ties. Initial published results describe available source rows, not an independently audited full schedule.
- Observed performance is computed from saved forecasts. Older winner records lack original creation timestamps; legacy provenance is incomplete. The default view isolates the latest known winner version.
- Score errors require an archive timestamp. Spread/total hit rates exclude pushes, missing picks and no-edge decisions. Winner ties do not enter probability metrics.
- A small observed sample or one favorable backtest does not demonstrate future improvement or betting profitability.

## Commands

```powershell
.\.venv\Scripts\python.exe rolling_backtest.py --start-season 2023 --end-season 2025
.\.venv\Scripts\python.exe performance.py
.\.venv\Scripts\python.exe -B -m unittest discover -p 'test*.py'
node test_dashboard.cjs
node public_site/build.mjs
```

Backtest outputs: `backtests/weekly_predictions.csv` and
`public_site/data/backtest.json`. Observed outputs:
`public_site/data/performance.json`. The weekly refresh regenerates observed
results, but never runs backtests or retrains production automatically.

## Iteration Log

- 2026-10-13: Implemented R1-R3 and initialized the options tracker. All 31 Python tests passed; desktop/mobile browser checks passed, including all 28 Week 6 team-logo images, tab navigation, model-version filtering and no page overflow. Recent-form baseline: 62.5% accuracy / 0.645 log loss; home-field baseline: 54.2% / 0.691. These retrospective results do not establish production improvement.
