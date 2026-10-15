# R28-R40: Evidence Before Promotion

## Implemented Scope

| ID | Capability | Boundaries |
| --- | --- | --- |
| R28 | Historical opponent metrics, 2017-2026 | Revised nflverse snapshots, franchise alias normalization, prior-week cutoffs; Week 1 missing. 2026 uses the saved current snapshot. Raw files private/local. |
| R29 | Five matched feature views | Fixed logistic/Ridge pipelines; annual 2024-2026 holdouts, training-only imputation/scaling, season-week paired bootstrap intervals. No selected winner or production promotion. |
| R30 | Immutable 24h and 90m checkpoint capture | Actual observation windows +/-30m and +/-15m; validated refresh must be within 30m. Missed windows stay missed. Weekly refresh calls capture, but no additional scheduler or guarantee of capture frequency is installed. |
| R31 | QB/availability scenario sensitivity | User-assumed point shifts applied to the archived correlated shadow margin distribution. Not learned player values, confirmed starters or causal effects. |
| R32 | Saved-pick error analysis | Confidence/team/stage/provenance groups, version-separated, ties excluded, small-sample warnings. Team groups overlap. No reconstructed historical injury coverage. |
| R33 | Model release controls | Content-addressed local model bundles, immutable named-reviewer approval, dry-run plans, activation/rollback with exception recovery. Existing feature contracts enforced. No model approved or activated in this release. |
| R34 | Data contracts | Duplicate IDs, feature coverage and finite-value checks plus missingness inventory; publication gate rejects failed checks. No fabricated missing data. |
| R35 | Timezone-aware kickoffs | Schedule Eastern times converted to UTC; browser renders Eastern or local timezone with DST. Unknown times must remain unavailable, not guessed. |
| R36 | Filtered CSV export | Visible matchups only, saved model/version/forecast fields, quoted CSV and string-formula neutralization. |
| R37 | Model/source registry | Artifact SHA-256 and sizes, source receipts and retrieval/publication distinction. Inventory is not a performance ranking. |
| R38 | Missing-feature robustness | Same all-feature model and test games; fixed trained imputer handles simulated missing groups. No refitting or causal claims. |
| R39 | Original-input replay audit | Up to ten exact source records with matching classifier artifact. Missing records/artifacts and mismatches remain visible; no historical record repair. |
| R40 | Keyboard-accessible game finder | Native datalist, game deep links, slash-to-focus outside editable fields, Escape dismissal. Existing team/week filters retained. |

## Results and Limits

Historical feature-value comparison uses 621 decisive holdout games. Opponent-
adjusted EPA changes mean log loss by about -0.0017 versus score-form/raw-EPA
baseline, with a 95% paired interval approximately [-0.0049, 0.0018]. The all-
metric view changes it by -0.0013, interval [-0.0057, 0.0029]. All added-feature
intervals include zero. These are inconclusive retrospective results, designed
after observing these seasons; multiple comparisons are not corrected.

Historical backfill counts represent available regular-season PBP. The canceled
2022 Buffalo-Cincinnati game is not a completed missing observation. Ratings
reset each season; missing Week 1 inputs are imputed only inside fitted research
pipelines, never invented in the source dataset. Nothing uses exact DVOA.

Some original production input archives used pandas' 10-decimal JSON precision.
The replay audit detects small probability mismatches on those records. This
release preserves them and fixes future archive serialization to retain full
numeric precision. Original unrounded inputs are unavailable for those records,
so rounding is a known limitation, not proof that every discrepancy has that cause.
The reliability monitor reports observed mismatches. No production picks change.

## Commands

```powershell
.\.venv\Scripts\python.exe historical_metrics.py
.\.venv\Scripts\python.exe research_operations.py --audit --scenarios
.\.venv\Scripts\python.exe release_control.py snapshot
.\.venv\Scripts\python.exe release_control.py plan --release RELEASE_ID
.\.venv\Scripts\python.exe -B -m unittest discover -p 'test*.py'
```

Historical collection/evaluation is explicit, not rerun automatically each week.
Source caches are under `data/historical_metrics/`; derived matchup features and
coverage fingerprints are tracked. Latest 2026 research coverage should be rerun
explicitly when refreshing this historical study; it is a frozen evaluation.

Release manifests live in `releases/`; model bytes are content-addressed under
ignored `.release_blobs/`. Back up that local directory privately for rollback
on another machine. A Git clone alone does not include rollback blobs. Only load
trusted local model files: joblib deserialization is executable. Human approval
requires `approve --release ID --reviewer NAME --reason REASON`; activation or
rollback is a separate explicit action. Reviewers must inspect evidence first.
This tool does not prove statistical readiness or authenticate reviewer identity.

Activation restores files after caught exceptions, but a machine crash between
individual file replacements is not bundle-atomic. Do not run activation during
forecast generation. Saved forecasts, data, model cohorts and hosted deployments
are not rolled back. Regenerate future forecasts and validate/publish separately
after a separately authorized production model change.

No new recurring development task was scheduled; implementation was completed
in the requested session. Existing Tuesday refresh remains the ongoing cadence.
