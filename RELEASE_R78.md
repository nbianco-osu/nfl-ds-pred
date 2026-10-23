# R78: Regulation semi-Markov prototype

This is the first implementation stage, not a production or final-game model.

## Delivered

- Empirical joint possession transitions: elapsed time, offensive and defensive
  score increments, next field position and possession retention.
- State conditioning on field-position bin, score-margin bin and late-game clock;
  league smoothing and a separate 35%-team-conditioned component.
- Half-clock censoring, opposite second-half receiver and explicit regulation ties.
- Whole-trajectory generative mixture. Component weight selected on a fixed grid
  using only earlier-week out-of-fold three-outcome Brier losses; equal weights
  until 16 earlier evaluations exist. No current-week labels in weight selection.
- Saved transition table, model specification, source hashes, per-game cutoffs,
  mean quarter scores, 80% score bands, lead probabilities and sample paths.
- Offline interactive report and deterministic temporal/clock/scoring tests.

## Data and limitations

The locally available archived PBP contains 79 games through Week 5. The newer
latest.json pointer refers to a missing local parquet. The run explicitly uses
the older named archive; it neither changes that pointer nor claims current data.
Later-week projections therefore reuse stale transition evidence. Historical
results use revised PBP and must not be called prospective performance.

This is an empirical state-conditioned bootstrap, not a complete NFL rules engine.
No play-level down/distance, explicit timeouts, kicker decisions, calibrated home
advantage, injuries, QB inputs or time-varying team strength are modeled. A drive
overrunning a half is censored without scoring; scores are placed at drive end.
Initial field position is fixed at 70 yards. League smoothing and team shrinkage
are assumptions, not optimized hyperparameters. Quarter bands omit parameter
uncertainty. Regulation lead probability is not in-game win probability.

## Not yet implemented

Rules-validated overtime and final-score market grading; blending with existing
production score distributions; shared-strength season trajectories; prospective
checkpoint capture; public-dashboard deployment. These require additional work
before the full proposed architecture is complete. No production models, picks,
shadow cohorts or writer schedules are changed.

## Reproduction

Run `python -B -m unittest test_markov_research`. Run `markov_research.py` with
explicit `--pbp`, `--schedule`, `--season`, `--trials` and a new `--output` folder.
Existing output folders are refused. `python -B markov_report.py PATH/report.json`
creates an offline HTML viewer in the run folder (expected under backtests/markov).
The initial smoke run uses 200 trials, not enough for production probability
precision. More simulation trials do not correct structural model errors.

## Verified smoke run

`backtests/markov/r78-prototype/` contains 256 generated matchups, 16 skipped
Week 1 games, 63 retrospective regulation evaluations, model.json,
transitions.csv, report.json and index.html. All probability vectors sum to one
and all per-game training maximum weeks precede their target weeks.
Seventeen new/existing scoring-research tests pass using workspace temporary
storage. Offline Chrome checks at 1440px and 390px passed matchup selection,
four-quarter table rendering, nonblank chart pixels, no JavaScript exceptions
and no page overflow. No public deployment or production integration was made.
