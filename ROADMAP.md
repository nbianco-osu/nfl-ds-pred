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
| R4 | High | Implemented prospectively; legacy limits remain | Raw source receipts and local snapshots, exact winner inputs and model hashes, forecast timestamps, kickoff freeze and local SHAP reconstruction. See DATA_AUDIT.md. Original historical publication times cannot be recovered. |
| R5 | High | Implemented research baseline | 36 nested weekly RF/HGB selection folds for 2024-2025, 544 regular-season games including one tie; Elo and separately labeled market benchmark, fixed-estimator EPA/QB/rest/NGS ablations and 95% accuracy intervals. Not a replica of the production 700-tree fit. |
| R6 | High | Implemented shadow experiment | Paired-residual Ridge score distribution; 1,030 prequential evaluations, interval coverage and three-class spread/total scoring. Future probabilities archived by version; no serving-model promotion. Scoring-process and prospective calibration remain future research. |
| R7 | High | Partial; data dependencies explicit | Current matchup source coverage, unconfirmed QB status/previous usage, prior play volume, opponent-adjusted EPA and same-week OL context. Timestamped weather adapter added, but no provider configured. Historical starter/weather snapshots and new-context ablations still needed; current features stay research-only. |
| R8 | Medium | Implemented review gate | Version-isolated prospective cohorts, 64-game minimum, paired bootstrap log-loss interval and Brier check; human decision CLI and audit trail. Approval records do not automatically change production. |
| R9 | Medium | Implemented prospectively | Per-game SHAP contributions, reference probability, source availability, input fingerprint and research distributions. Never reconstruct explanations for legacy finals. |
| R10 | Medium | Implemented | Independent score-model/version comparisons and calibration chart with Wilson 95% intervals and bin sample sizes. Empty/legacy samples remain explicit. |
| R11 | Medium | Implemented; operational hardening continues | Staged validation, atomic per-file writes with rollback, health/stale indicator, Pages validation gate and scheduled Python/build CI. Weekly automation updated to semantic comparison and new audit exports. Whole-directory crash atomicity and remote raw-snapshot backup remain open. |

## R12-R17 Iteration

R12-R17 are implemented with the limits below. Finish R7's historical inputs and
review the legacy roster-status correction before adding more model families.

| ID | Priority | Status | Scope and acceptance |
| --- | --- | --- | --- |
| R12 | High | Implemented prospectively | Timestamped winner and combined-score history, same-version deltas and changed inputs/source hashes. Immutable observations preserved in audit/dashboard_history.json; unchanged refreshes deduplicate. No reconstructed legacy history. |
| R13 | High | Implemented | Classifier/score-ensemble disagreement on cards; per-family expected scores, versions and margin range in game details. Dispersion is not a calibrated interval. |
| R14 | High | Implemented with provider limits | Coverage summary and detailed missing/fallback counts, QB/weather/injury limitations, source observation age and roster caveat. Freshness unknown where publication times absent; no probability penalty. |
| R15 | Medium | Implemented | 32 profiles, opponent-adjusted EPA trend, complete schedule/scores, record and version-scoped observed winner accuracy with counts. Trends use revised research context, not live historical availability. |
| R16 | Medium | Implemented | Browser-local favorite, URL-linked week/team/confidence or game, reset, back-navigation handling and storage-failure fallback. No accounts or server-stored preferences. |
| R17 | Medium | Implemented; accumulating prospective evidence | Winner/score-version-isolated horizon and season-stage metrics, calibration bias, score MAE and sample-gated drift alerts. No settled verified observations initially; no automatic production changes. |

## R18-R22 Iteration

R18 is the current matched-evaluation iteration. R19-R22 remain proposals, not
authorization for implementation or model promotion.

| ID | Priority | Status | Acceptance check |
| --- | --- | --- | --- |
| R18 | High | Implemented retrospectively | Re-tuned 64 recipes on 621 identical decisive games across annual nested rolling folds for 2024-2026; paired log-loss/Brier intervals, counts, information-set filters and cutoff audit. Prospective evidence remains required; no automatic promotion. |
| R19 | High | Proposed | Evaluate Poisson/Tweedie and football-scoring-process models for scores, margins and totals; compare calibration, proper scoring rules and interval coverage against current score baselines. |
| R20 | High | Proposed | Archive confirmed starters and weather forecasts with observation/kickoff timestamps; measure incremental value without historical backfilling of availability. |
| R21 | Medium | Proposed | Train regularized stacking only on temporal out-of-fold predictions; isolate its calibration and prospective forecasts by model version. |
| R22 | Medium | Proposed | Normalize warehouse team/player/venue dimensions and automate verified incremental sync after privilege, storage-cost and source-license review. Keep raw snapshots private. |

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

- 2026-10-14, recovery and R18: Completed and fingerprint-verified all 10,089 private Supabase records using the connected tool, preserving existing rows and RLS. Added matched evaluation of 64 re-tuned research recipes across 272 / 271 / 78 decisive games in 2024 / 2025 / 2026. The new tab separates information sets and reports paired season-week bootstrap intervals, fold cutoffs, missingness and selected settings. Original AutoML searches are bounded at 4/6 trials; revised source data and recipe-selection optimism remain limitations. Production models, forecasts and shadow learning are unchanged. R19-R22 remain proposed.

- 2026-10-13, database and research expansion: Trained and saved 32 additional temporally calibrated configurations (16 method families x 2 feature views), bringing the library to 64. New research evaluates 78 decisive 2026 games through October 12, not the earlier library's 48-game cohort. Added private FTN/snap-count source collection, lagged charted features, 7,031-row research dataset (2026 tie retained), private Supabase schema and verified-batch importer. Database import status is recorded separately in database/README.md; no production promotion. Next priorities: matched rolling evaluation, score-distribution models, temporal stacking, and confirmed-starter/weather archives.

- 2026-10-13, R12-R17: Added archive-backed insights, team profiles, preferences and monitoring; refresh exporter runs inside existing staged validation. Production models and forecasts unchanged. Monitor uses latest observation per game/version/horizon, with no retrospective score backfill. Descriptive drift thresholds require at least 32 decisive games in both recent and earlier windows; no statistical significance or improved-performance claim.

- 2026-10-13: Added blue projected-winner emphasis to weekly team rows, with explicit text and a pregame-pick label for finals. Green/red remain outcome correctness indicators. Highlight uses the classifier's saved pick even when the score ensemble disagrees. Added R12-R17 proposals; no model or forecast changes.

- 2026-10-13: Implemented R1-R3 and initialized the options tracker. All 31 Python tests passed; desktop/mobile browser checks passed, including all 28 Week 6 team-logo images, tab navigation, model-version filtering and no page overflow. Recent-form baseline: 62.5% accuracy / 0.645 log loss; home-field baseline: 54.2% / 0.691. These retrospective results do not establish production improvement.
- 2026-10-13, R4-R11: Added provenance, nested evaluation, shadow score distributions, review gates, independent score results and staged publishing. Nested RF/HGB log loss 0.640 versus Elo 0.638 on 543 decisive games; no basis for automatic promotion. Nominal 80% research intervals achieved roughly 79-81% coverage. Preserved all 79 completed production forecasts. R7 remains partial for confirmed-starter/weather history and contextual ablations. An audit identified the legacy active-roster status-code mismatch; corrected research active-OL counts are separate from production.
- Validation: 41 Python tests pass, including temporal boundaries, mixed-version rejection, staged-failure preservation, atomic-write failures and probability conservation. Desktop/mobile browser checks cover local explanations, all 28 current-week logos, score-model filtering and nonblank calibration intervals. Production classifier/score artifacts and existing calibration forecasts remain unchanged.
