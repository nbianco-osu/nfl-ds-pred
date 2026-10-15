# R23-R27

## Inputs

`weather_inputs.py --season 2026` retrieves Open-Meteo outdoor forecasts for
scheduled stadium IDs within 15 days (16-day API request). Public free endpoint
is for non-commercial use; retain Open-Meteo attribution. Coordinates come from
greerreNFL/Stadiums; missing venues are never mapped to a home team's usual venue.
Closed/dome games skip outdoor collection. Other roof states remain outdoor
conditions, not confirmed field conditions. Invalid values or unavailable source
responses preserve the old archive. Exact response hashes and first observations
are retained; provider issue time stays null. Historical weather is not backfilled.

Confirmed starters use `pregame_inputs.py --input PATH --season 2026`, with an
explicit confirmation, team/player identity, source URL and published timestamp.
No automated starter provider is configured. Current gaps are shown in monitoring.
Input availability does not mean these variables have been added to production.

## Models and Evaluation

`live_challengers.py --train --season 2026` is an explicit research-training command.
It compares Poisson, independent negative binomial (NB2) and Gaussian-copula NB2
score distributions on the same 623 available games across 2024-2026. All use
the same Poisson-regression mean features: prior-eight-game points for/against.
Mean fits end two seasons before the test year. Dispersion and rank correlation
use only the preceding season. No test-label tuning or best-model selection.
The final frozen generation uses 2017-2024 means and 2025 calibration, not 2026
outcomes. Saved artifacts have input/code-derived version IDs.

NB2 variance is mean + dispersion * mean squared; dispersion is a calibration
method-of-moments estimate. Gaussian-copula correlation uses calibration residual
ranks, clipped to [-0.8, 0.8]. It is not a fitted discrete copula likelihood.
Scoring uses 10,000 draws and seed 42, nonnegative integer outcomes, CRPS, 80%
coverage/width/interval score and available historical line metrics. Football
scoring constraints and overtime rules are not modeled. Thus simulated tie mass
is not a calibrated NFL tie forecast. Winner probabilities condition on no tie.

Total coverage across these historical games is approximately 51.2% for Poisson,
81.9% for NB2 and 81.5% for correlated NB2 against a nominal 80%. Total CRPS is
about 8.00, 7.46 and 7.45 respectively. These descriptive research results are
not proof of future improvement or significance, and do not justify promotion.

`live_challengers.py --season 2026` exports from pinned artifacts without training.
First saved observations per game/version/model are immutable under
`learning/live_challengers/<version>/forecasts.json`. Upcoming form inputs use
only completed games before that week's first date. Captured production
probabilities/versions are matched comparators. Evaluation excludes ties, requires
the archive timestamp precede actual kickoff, and separates production cohorts.
No archived results are reconstructed from replay outputs. Prospective sample
starts October 15, 2026; 193 future games x 3 challengers were initially saved.
First snapshots may be far before kickoff; these are not closing forecasts.

The separate 272-game season replay uses the frozen pre-2026 artifacts and
week-frozen features. Completed-game predictions are labeled retrospective;
future games use current known history, never invented future outcomes.
Revised historical source snapshots remain a limitation. This does not turn
new Opponent-Adjusted EPA features into trained classifier inputs; that ablation
still needs enough historical source coverage. R21 stacking is not substituted.

## Monitoring and Operations

`reliability.py --season 2026` emits actionable issue IDs, severity, next action
and a deterministic fingerprint. It checks eight-day refresh age, 48-hour weather
age, missing upcoming predictions/inputs/shadows, failed imports/refreshes,
weather failures and queue hash against the last verified remote receipt.
It does not authenticate to the database or claim a sync has succeeded.

Staged refresh adds weather collection, opponent ratings and live export. It
records failed refresh status without publishing partial forecast replacements.
Monitoring runs after success/failure. The Tuesday automation reruns monitoring
after remote sync and notifies on changed actionable conditions, not timestamps.
No new notification channel, daily schedule or automatic model promotion is added.

The dashboard keeps technical details in secondary views. Weekly score intervals
are explicitly a separate shadow model, not uncertainty around the production
score ensemble. Production model artifacts and saved pregame picks are unchanged.
