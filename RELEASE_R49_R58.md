# R49-R58 evidence and operations

Production artifacts and existing winner/score picks are unchanged. R58 remains
deferred: the expanded generation has no completed prospective results yet, and
its feature contracts still require an explicit serving adapter and human review.

| ID | State | Delivered scope |
| --- | --- | --- |
| R49 | Implemented | Forecast Accuracy tab: independent winner/score versions, weekly chart, log loss/Brier/MAE, strict timezone-aware pregame eligibility, separate unverified history. Line hit rates additionally require an archived pregame line. |
| R50 | Implemented | Matchup Evidence: model dispersion by game, generation, task and observation window. Correlated models are not independent votes or calibrated uncertainty. |
| R51 | Implemented with boundary | Offense/defense opponent-adjusted EPA plus matched 543-game Elo, simple-form, raw-EPA and adjusted-EPA comparison. Special-teams EPA is descriptive only, not opponent-adjusted or prospectively validated. |
| R52 | Implemented using existing research | Joint paired-residual score forecasts: same draws produce score ranges, margin/total and win/tie/line probabilities. No new production distribution or unsupported joint PMF. |
| R53 | Implemented research | Eight genuine drop-group refits on 544 historical games: EPA, sacks, big plays, QB, injury, roster, all opponent groups, full model. Prior-game personnel proxies; no claim of current confirmed availability. |
| R54 | Implemented | Per-game input status, receipt and provider timestamps; absent evidence is explicit. Starter/weather inputs remain context-only for this candidate generation. |
| R55 | Implemented | Full weekly recap, including wrong picks, missing scores, ties and pending games. Winner/score provenance separate; correct green, wrong red plus text. |
| R56 | Implemented | Staged and Pages publication gates: valid integer final scores, kickoff-locked picks/archived lines, exact-input hashes/timing, training cutoffs and explained coverage drops. |
| R57 | Active; end-to-end verified | Private runner is the sole refresh writer. Live run 38814252099 saved private generation 2 and published c30fbf3; Pages succeeded. Local pregame writer is paused and Tuesday local automation is a read-only health/readiness monitor. |
| R58 | Deferred | Existing read-only readiness and manual release controls retained. Sufficient matched prospective evidence, calibration and serving compatibility are required before considering approval. No automatic promotion. |

## Interpretation

- All 79 completed winner picks currently lack verified pregame timestamps. The
  strict view correctly has no graded prospective winner outcomes. Legacy results
  remain accessible but never fill the prospective view.
- Matched baseline log loss: adjusted EPA 0.6345, raw EPA 0.6364, Elo 0.6383,
  simple form 0.6408 on the same 543 decisive games. Their paired intervals against
  simple form include zero. These recipes use different chronological fit windows;
  the match is in held-out games, not an identical training experiment.
- Personnel sources end in 2025. The personnel ablation therefore uses 2024-2025,
  not unsupported 2026 personnel history. Same-season prior-game snapshots are
  used before target-week cutoffs. Upstream revised data/zero fills and post-hoc
  group choices remain limitations. Weather and opponent-adjusted special-teams
  ablations are unavailable.
- Special-teams summary covers 2,118 valid plays. EPA credits follow recorded
  possession-side ownership, which on kickoffs may be the receiving team; this is
  not a kicker valuation. Minimum coverage is a display rule, not significance.
- Research refits require explicit `iteration_evidence.py --research`. Scheduled
  exports reuse the frozen experiment and flag changed sources; they never refit it.

## Verification and publication

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -p 'test*.py'
.\.venv\Scripts\python.exe -B forecast_scorecard.py
.\.venv\Scripts\python.exe -B iteration_evidence.py --season 2026
.\.venv\Scripts\python.exe -B publication_gate.py --against HEAD
.\.venv\Scripts\python.exe -B dashboard_snapshot.py
node test_dashboard.cjs
node test_cloud_data.cjs
node public_site/build.mjs
```

The hosted Site reads a single public `live_snapshot.json` from GitHub Pages.
All panels and downloads share that in-memory revision. If unavailable, all data
falls back to the bundled release with a visible warning. No credentials or raw
private sources enter this public bundle. Snapshot-only operational timestamp
changes do not independently trigger publication.

R57 preserves missed windows, never invents observations, and never guarantees
exact scheduling. Private Actions consume account quotas; no billing settings are
changed. See `runner/README.md` for state preservation, bounded schedules, manual
rehearsal, activation, private storage and recovery.
