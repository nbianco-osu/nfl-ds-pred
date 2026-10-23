# R79-R84: Generative shadow ensemble

## Delivered scope

- R79: Fresh, separate nflreadpy research snapshot observed October 23, 2026.
  94 completed games, all with PBP; source hashes and observation timestamps.
  Production source pointers are untouched. Provider revisions/cache remain possible.
- R80: Final-score endpoint with 2026 regular-season overtime stopping rules:
  600-second limit, reply opportunities, sudden death, safety/defensive TD stops,
  retained ties and no unnecessary conversion on a winning TD. Tests cover these
  scenarios. This is not a complete play-level NFL rules engine: penalties,
  onside recoveries, timeouts, untimed downs and end-of-possession scoring times
  remain approximations. OT transition frequencies use regulation drives.
- R81: Semi-Markov, existing Ridge paired-residual and Poisson score recipes,
  plus a joint-score generative mixture. Baselines fit 2017-2024 and calibrate
  in 2025. Transition fits and mixture weights use only earlier weeks. Weights
  optimize three-outcome Brier loss with fixed shrinkage toward equal weights;
  no current-week labels. Week 1 skipped for insufficient earlier PBP.
  Matched score CRPS, winner losses, market losses, calibration and paired
  week-bootstrap intervals are exported. Existing production artifacts are not
  refitted or used as falsely historical predictions.
- R82: 10,000 season simulations, seed 42, with shared Gaussian team-strength
  latent variables. The Gaussian copula preserves per-game input marginals;
  strength SD 0.35 is an explicit assumption, not fitted. Completed outcomes
  fixed, ties separate, no win caps; per-trial league conservation checked.
- R83: Frozen generation/artifact hashes, exact feature/line snapshots, actual
  observation timestamps, 24h/90m windows, exclusive-write forecasts/results,
  no missed-window backfill, no mixed generations, ties/no-edge excluded from
  winner accuracy, result-correction flags. Candidate-only readiness requires
  100 matched 90m games over six weeks; never automatic promotion. The sole
  private runner captures; local scripts cannot masquerade as live forecasts.
- R84: Lazy Generative Forecasts tab with final scores, matchup distributions,
  regulation component paths, calibration, paired comparisons and team season
  win bands. Research and prospective evidence remain clearly separated.

## Current artifact and evidence

Generation: `generative-b19f03a93e952fcc`.
Research run: `backtests/generative/r79-r84-20261023`.
1,000 draws/component/game; 256 generated matchups, 16 skipped, 78 evaluated.
Pinned live transition bank ends Week 6, so Thursday Week 7 does not leak into
the remaining Week 7 predictions. Frozen mixture weights: Markov 35.7%, Ridge
60.9%, Poisson 3.4%. Frozen models do not automatically train on later games.

Ensemble three-outcome log loss 0.735 versus Ridge 0.730; paired difference
0.0046, descriptive 95% week-bootstrap interval -0.0109 to 0.0209. Ensemble
margin CRPS 6.579 versus Ridge 6.590; total CRPS 7.633 versus Ridge 7.599.
No demonstrated improvement. Only six evaluation week blocks; revised-source
and model-design-selection optimism remain. No prospective outcomes yet.

## Validation and operations

190 Python tests and 10 Node tests passed. Desktop 1440px and mobile 390px
checks passed matchup/team switching, 256 games, 32 teams, charts with nonblank
pixels, 18-week tables, paired comparisons, no page errors or page overflow.
The test temp directory is set to the workspace because the sandbox OS temp
directory cannot reliably create nested test paths.

Cloud installation uses `runner/generative_workflow.yml` and the existing
`nfl-private-state` concurrency group, installs only a separate model generation
and public research exports, and validates all old state before CAS persistence.
Runtime refreshes infer with pinned artifacts; they never call training here.
Source/evaluation freshness is distinct from future forecast observation time.

Sites publication was blocked by auto-review rejecting credential input to the
official helper. No attempt was made to bypass it. GitHub deployment and cloud
installation outcomes are recorded separately after execution.

## Commands

`python -B generative_sources.py --season 2026 --output data/generative/NEW_RUN`

`python -B generative_pipeline.py --sources data/generative/NEW_RUN --output backtests/generative/NEW_RUN --season 2026 --trials 1000`

`python -B export_generative.py backtests/generative/NEW_RUN`

`python -B -m unittest test_generative test_markov_research`

Existing run folders are never overwritten. Raw source snapshots and model
joblib remain local/private; public exports contain research summaries.

Rules reference: https://operations.nfl.com/rules-officiating/2026-nfl-rulebook

Next: evaluate OT-specific transition data, fit team-strength uncertainty,
improve kickoff/clock modeling, and assess matched prospective performance
before considering any production change. R58 remains deferred.
