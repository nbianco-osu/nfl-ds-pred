# R19-R22 Evidence and Remaining Work

## Score Distributions (R19)

`python score_experiments.py` compares four fixed recipes on identical 2024,
2025 and completed 2026 games, including available ties. Base fitting uses
2017 through Y-2; residual/dispersion calibration uses Y-1; Y is untouched testing.
Four lagged score-form features are frozen before each week. This is a controlled
research experiment, not a refit of the production five-model score ensemble.

Ridge resamples paired calibration residuals; Poisson assumes independent point
counts. Tweedie power 1.5 uses compound Poisson-Gamma draws with calibration-only
dispersion. The scoring-event model uses assumed marks 2/3/6/7/8 with probabilities
.01/.34/.04/.59/.02 and shared Gamma(shape=20, scale=.05) pace. It is not a fitted
drive/possession/overtime engine. All use 10,000 draws and seed 42.

Report CRPS, point MAE, nominal 80% coverage, width and interval score, plus
three-class line Brier/log loss (below/push/above). Monte Carlo line probabilities
use .5 pseudocount smoothing. A wide interval can cover well but still score
poorly. Market lines are revised historical snapshots, not verified pregame lines.
No predictive-performance guarantee or automatic promotion follows.

623 games (272/272/79) are covered, whereas R18 has 621 decisive games because
its historical source omits the 2025 tie. Do not compare metrics across those
different cohorts as if they were matched. Final research artifacts use a base
fit through 2024 and 2025-2026 calibration, and stay separate under models/research.

## Temporal Stacking (R21)

`python temporal_stack.py` consumes R18 OOF predictions from 60 no-market recipes.
Fixed logistic C=.01 fits 2024 logits, sigmoid calibration uses 2025, and 78
decisive 2026 games are held out. Equal-weight averaging of the same bases is the
comparator. No random cross-validation or selecting the best retrospective bases.
The paired log-loss interval spans zero; improvement is inconclusive.

Artifact and report are version-isolated. Prospective forecasts are deliberately
withheld until the matching R18 base-fit artifacts and live feature inputs exist.
Substituting older saved library fits would silently change the evaluated model.
Thus R21 is implemented retrospectively, not complete prospective validation.

## Warehouse (R22)

Private dimensions cover 36 team identifiers, 38 schedule venues and 42 schedule
QB identities, with 272 immutable schedule observations. Schedule QB names are
not verified starters. No complete roster upload, raw-object bucket or license
expansion was performed. Existing 10,089 records remain intact.

`warehouse_queue.py` stages 10-row SQL batches for the connected Supabase tool.
It does not hold credentials or claim a remote write succeeded. The weekly
assistant executes batches and verifies full payload/metadata equality. Existing
identical snapshots are reused, timestamp-only changes ignored, and new inserts
normalize transactionally through a private SECURITY INVOKER trigger. This is
assistant-mediated incremental sync, not a standalone unattended service.

Security: private schema, RLS on every table, no anon/authenticated grants. The
setup RLS event trigger remains enabled, but unnecessary public execute grants
were revoked. The advisor's [RLS without policies notice](https://supabase.com/docs/guides/database/database-linter?lint=0008_rls_enabled_no_policy)
is intentional deny-by-default behavior, not a reason to add public policies.

References: [TweedieRegressor 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.TweedieRegressor.html),
[Supabase functions](https://supabase.com/docs/guides/database/functions),
[Supabase RLS](https://supabase.com/docs/guides/database/postgres/row-level-security).
