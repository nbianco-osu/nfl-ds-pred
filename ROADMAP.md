# NFL Predictor Roadmap

Updated: 2026-10-15. Keep this file current in each iteration. Record evidence,
limitations, and the next acceptance check before marking an item complete.

## Current Iteration

### R41-R48 and 120 Configurations

See `RELEASE_R41_R48.md` for protocols, results, commands and limitations.

| ID | Status | Delivered scope |
| --- | --- | --- |
| R41 | Implemented; inconclusive | Matched production-estimator recipe comparison and prospective production pairing; separate from actual historical live performance. |
| R42 | Implemented shadow-only | 120 configurations (80 winner, 40 score) plus two temporal ensembles; next-eight-day predictions and exact inputs archived. Added 48 small neural-network configurations. |
| R43 | Implemented; awaiting outcomes | Immutable model/generation/game/horizon cohorts, ties/pushes/no-edge handling, score errors and interval coverage. No retrospective backfill. |
| R44 | Implemented research | Early-2025 convex weights, independent 2026 evaluation; score blend selected Ridge alone. No promotion or proven gain. |
| R45 | Implemented with calibration limits | Late-2025 residual intervals, 2026 coverage and prospective coverage tracking. Not joint guarantees; season drift remains. |
| R46 | Implemented | Game-level observed ages, stale/missing fields, frozen charting status; starter/weather context remains outside the fitted feature set. |
| R47 | Active schedule; capture availability limited | Ten-minute due-window heartbeat, shared refresh/model-activation lock, genuine 24h/90m observations only. Host and runtime availability required. |
| R48 | Review implemented; promotion blocked | Read-only checklist, local approval/rollback controls, explicit incompatible feature contracts and no prospective results yet. No automatic promotion. |

The original 60 artifacts were retained unchanged on identical input hashes;
60 additional configurations were trained. Generation
`expanded120-20261015T191351669323Z` is pinned only for shadow inference.

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

R18-R22 now include implemented research and private warehouse improvements.
Provider coverage and prospective evaluation remain incomplete; no model promotion.

| ID | Priority | Status | Acceptance check |
| --- | --- | --- | --- |
| R18 | High | Implemented retrospectively | Re-tuned 64 recipes on 621 identical decisive games across annual nested rolling folds for 2024-2026; paired log-loss/Brier intervals, counts, information-set filters and cutoff audit. Prospective evidence remains required; no automatic promotion. |
| R19 | High | Implemented research | Four score distributions on 623 matched games; CRPS, interval coverage and three-class line scoring. Scoring-event assumptions are not a full football engine. Saved research artifacts; production unchanged. |
| R20 | High | Import support implemented | Strict timestamped starter/weather imports, immutable observations and explicit missing inputs. No provider configured, zero observations; incremental-value evaluation awaits prospective data. |
| R21 | Medium | Retrospective implementation | Regularized stack fit on 2024 temporal OOF predictions, calibrated in 2025, evaluated on 78 decisive 2026 games. Paired interval is inconclusive. Future forecasts withheld until matching live base artifacts are available. |
| R22 | Medium | Private dimensions and assisted sync implemented | Private team/venue/schedule-QB dimensions, insert normalization and verified incremental connector queue. 580 records matched with zero duplicate inserts. Full roster coverage and raw-object storage remain pending. |

## Evidence and Boundaries

## R23-R27 Iteration

| ID | Status | Scope and remaining limits |
| --- | --- | --- |
| R23 | Implemented with explicit source gaps | Automatic non-commercial Open-Meteo forecasts, stadium-ID coordinate matching, immutable response/observation archive and source-backed confirmed-QB import. No fabricated issue timestamps or historical weather; no automatic confirmed-starter provider. Initial 23 weather observations; confirmations still missing. |
| R24 | Implemented prospectively | Three frozen pre-2026 score challengers, 579 first-observed forecasts for 193 future games, exact feature/version archive and matched production-probability comparisons by cohort. No retrospective scores counted as prospective evidence. Human review required; stacking remains withheld. |
| R25 | Implemented research | Independent Poisson, NB2 overdispersion and correlated Gaussian-copula NB2; identical 623-game chronological evaluation, calibration only on preceding season, CRPS/coverage/interval scores. Counts are not a football/overtime engine. No production promotion. |
| R26 | Implemented | Compact weekly matchup default with expected scores and separately labeled research uncertainty. Library/global SHAP moved to research; season bars moved to season view. Detailed views retained, including clearly labeled full-season replay. |
| R27 | Implemented operational checks | Failed refresh/import, missing predictions/starters/weather, stale refresh/weather, missing shadow forecasts and unverified database queue. Deterministic issue fingerprint for meaningful-change notifications in weekly automation. No external paging service or continuous uptime guarantee. |

Separate season replay covers all 272 scheduled 2026 games for three models.
Completed-game reconstructions never replace archived pregame production picks.

## Evidence and Boundaries (Continued)

## R28-R40 Iteration

| ID | Status | Delivered scope |
| --- | --- | --- |
| R28 | Implemented | Historical opponent-adjusted EPA/sack/big-play features for 2017-2026, franchise aliases, coverage fingerprints and week-frozen inputs. |
| R29 | Implemented research | Five matched views on 621 decisive holdout games; paired season-week intervals. All differences inconclusive; production unchanged. |
| R30 | Capture implemented; scheduler added in R47 | Immutable 24h/90m snapshots with freshness and window gates. Weekly and due-window refresh invoke capture; missed windows stay explicit. |
| R31 | Implemented sensitivity tool | User-assumed QB/availability point shifts over archived shadow distributions; not player valuations or confirmations. |
| R32 | Implemented | Version-separated saved-pick errors by confidence, team, stage and provenance; sample warnings and tie exclusions. |
| R33 | Implemented local controls | Content-addressed model bundles, human approval, integrity/feature-contract gates, dry run and exception-safe activation/rollback. Baseline snapshot only; no approval or activation performed. |
| R34 | Implemented | Feature keys, coverage, finite-value checks and missingness inventory; publication contract gate. |
| R35 | Implemented | Eastern/local kickoff displays with DST-aware UTC source timestamps. |
| R36 | Implemented | Filtered CSV export with versions/timestamps and formula-string protection. |
| R37 | Implemented | Model/source registry with artifact fingerprints and source observation provenance. |
| R38 | Implemented research | Missing-feature stress tests using trained imputers and identical held-out games. |
| R39 | Implemented audit | Original archived input replay; one of ten sampled records differs slightly. Old precision limitation preserved; future serialization fixed. |
| R40 | Implemented | Keyboard-accessible game finder, datalist and game deep links. |

See RELEASE_R28_R40.md for commands, evidence, safety boundaries and release-blob
backup requirements. No new recurring development task was created.

## Evidence and Boundaries (Research)

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

- 2026-10-17, R61-R66: Installed the corrected 240-model cohort and robust ensemble
  in the private shadow runner with all old model/forecast files preserved and no
  production promotion. Added immutable core training vintages, fixed prospective
  review looks, training-only representative-member pruning, a possession-event
  simulator with matched Poisson comparisons, and approval-bound fallback/rollback
  rehearsal. These are research and operational safeguards, not evidence of improved
  prospective accuracy. See RELEASE_R61_R66.md for delivered scope and limitations.
  R58 remains deferred pending sufficient prospective evidence.

- 2026-10-17, R59-R60: Trained a 240-configuration library (140 winner, 100 score),
  preserving the original 120 artifacts. Built and saved a six-member-per-task
  constrained/calibrated ensemble with chronological OOF fitting, paired-residual
  ranges and fail-closed inference. Historical advantage over equal weighting is
  inconclusive; production and the cloud's existing 120-model shadow generation
  remain unchanged. New model binaries are local, pending private-runner migration.
  Training inputs end October 12. See RELEASE_R59_R60.md for evidence, reproduction
  commands and proposed R61-R66; R58 remains deferred.

- 2026-10-15, R49-R58: Added strict timestamp-gated accuracy, model disagreement,
  joint score summaries, complete weekly recaps, source availability, matched
  baselines, true grouped ablations and publication transition gates. R57 private
  runner is active with a successful live refresh/publication; R58 is deliberately
  deferred pending sufficient prospective evidence and human review. Special-teams
  EPA remains descriptive, and weather ablation is unavailable. See
  RELEASE_R49_R58.md for exact delivered scope and limitations.

- 2026-10-15, R28-R40: Completed historical coverage and feature ablations, checkpoint capture, sensitivity scenarios, diagnostics, release controls and Evidence Lab. All 84 Python tests and desktop/mobile browser checks passed, including CSV download and keyboard navigation. No production model/pick changes. Historical added-feature differences are inconclusive. Replay audit reports one small mismatch among ten sampled forecasts; future archives preserve full numeric precision.

- 2026-10-15, R23-R27: Added observed-time weather collection, frozen live challenger cohorts, negative-binomial uncertainty research, compact default view, full-season research replay and actionable reliability checks. Production artifacts and pregame picks unchanged. Confirmed starters remain supplied inputs, and prospective challenger outcomes are not yet available. Full implementation notes: RELEASE_R23_R27.md.

- 2026-10-15, reproducible DVOA alternative: Calculated joint ridge Opponent-Adjusted EPA (not DVOA), sack differentials and explosive-play differentials from the archived 2026 PBP snapshot. Saved 32-team rankings and week-frozen features for 272 matchups using 79 completed games. Week 1 remains missing; no fabricated prior-season data. Added source fingerprints, coverage, methodology and temporal tests; all 73 tests passed. Weekly refresh regenerates research files only. No production refit, dashboard publication or predictive-improvement claim; broader R23-R27 work remains pending.

- 2026-10-14, R19-R22: Added Experiments & Inputs dashboard, four saved score distributions, retrospective temporal stacking, strict supplied-input archive and private normalized warehouse. Browser desktop/mobile checks and 69 Python tests passed. Existing production models, picks and learning cohorts unchanged. Stacking improvement is inconclusive; confirmed starter/weather inputs remain unavailable.

- 2026-10-14, recovery and R18: Completed and fingerprint-verified all 10,089 private Supabase records using the connected tool, preserving existing rows and RLS. Added matched evaluation of 64 re-tuned research recipes across 272 / 271 / 78 decisive games in 2024 / 2025 / 2026. The new tab separates information sets and reports paired season-week bootstrap intervals, fold cutoffs, missingness and selected settings. Original AutoML searches are bounded at 4/6 trials; revised source data and recipe-selection optimism remain limitations. Production models, forecasts and shadow learning are unchanged. R19-R22 remain proposed.

- 2026-10-13, database and research expansion: Trained and saved 32 additional temporally calibrated configurations (16 method families x 2 feature views), bringing the library to 64. New research evaluates 78 decisive 2026 games through October 12, not the earlier library's 48-game cohort. Added private FTN/snap-count source collection, lagged charted features, 7,031-row research dataset (2026 tie retained), private Supabase schema and verified-batch importer. Database import status is recorded separately in database/README.md; no production promotion. Next priorities: matched rolling evaluation, score-distribution models, temporal stacking, and confirmed-starter/weather archives.

- 2026-10-13, R12-R17: Added archive-backed insights, team profiles, preferences and monitoring; refresh exporter runs inside existing staged validation. Production models and forecasts unchanged. Monitor uses latest observation per game/version/horizon, with no retrospective score backfill. Descriptive drift thresholds require at least 32 decisive games in both recent and earlier windows; no statistical significance or improved-performance claim.

- 2026-10-13: Added blue projected-winner emphasis to weekly team rows, with explicit text and a pregame-pick label for finals. Green/red remain outcome correctness indicators. Highlight uses the classifier's saved pick even when the score ensemble disagrees. Added R12-R17 proposals; no model or forecast changes.

- 2026-10-13: Implemented R1-R3 and initialized the options tracker. All 31 Python tests passed; desktop/mobile browser checks passed, including all 28 Week 6 team-logo images, tab navigation, model-version filtering and no page overflow. Recent-form baseline: 62.5% accuracy / 0.645 log loss; home-field baseline: 54.2% / 0.691. These retrospective results do not establish production improvement.
- 2026-10-13, R4-R11: Added provenance, nested evaluation, shadow score distributions, review gates, independent score results and staged publishing. Nested RF/HGB log loss 0.640 versus Elo 0.638 on 543 decisive games; no basis for automatic promotion. Nominal 80% research intervals achieved roughly 79-81% coverage. Preserved all 79 completed production forecasts. R7 remains partial for confirmed-starter/weather history and contextual ablations. An audit identified the legacy active-roster status-code mismatch; corrected research active-OL counts are separate from production.
- Validation: 41 Python tests pass, including temporal boundaries, mixed-version rejection, staged-failure preservation, atomic-write failures and probability conservation. Desktop/mobile browser checks cover local explanations, all 28 current-week logos, score-model filtering and nonblank calibration intervals. Production classifier/score artifacts and existing calibration forecasts remain unchanged.
