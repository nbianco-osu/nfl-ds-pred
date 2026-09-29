# Expanded Team Metrics

Research and data refresh: September 29, 2026. Sources were checked online and
loaded through nflreadpy. Raw downloads remain in its cache; per-game aggregate
caches live in `data/expanded_cache/` and are not published.

## Sources and Choices

- [nflfastR play-by-play definitions](https://nflfastr.com/reference/fast_scraper.html):
  early-down EPA, sacks and recorded QB hits per dropback, third-down conversion
  rates, rushes gaining no yards, and red-zone drive touchdown rates. QB hits are
  not all pressures. Red-zone rates count drives reaching the opposing 20, not
  touchdowns per red-zone play. Offensive and defensive versions are included.
- [NFL Next Gen Stats via nflverse](https://nflreadr.nflverse.com/reference/load_nextgen_stats.html):
  available from 2016, with qualifying-player minimums and weekly summaries.
  We exclude season-summary rows (`week == 0`). Team passing/rushing metrics are
  attempt-weighted and receiving metrics target-weighted among reported players.
- [NGS passing definitions](https://nextgenstats.nfl.com/stats/passing): time to
  throw, intended air yards, aggressiveness (tight-window rate), and completion
  percentage above expectation. Rates are retained in their source units.
- [NGS rushing definitions](https://nextgenstats.nfl.com/stats/rushing): rushing
  yards above expectation per attempt and the proportion exceeding expectation.
- [NGS receiving definitions](https://nextgenstats.nfl.com/stats/receiving):
  separation and yards after catch above expectation.
- [FTN charting availability](https://nflreadr.nflverse.com/reference/load_ftn_charting.html)
  was investigated but is not incorporated: its history starts in 2022 and it
  has a separate CC-BY-SA attribution requirement. No FTN or proprietary DVOA/PFF
  scores were scraped or represented as available in this dataset.

Custom Elo uses a 1500 starting rating, K=20, a fixed 55-point home adjustment,
and 25% regression to 1500 each offseason. Opponent strength averages previously
faced opponents' pregame Elo within the season. These are explicit modeling
assumptions, not an NFL or FiveThirtyEight rating reproduction. Elo includes
ties as half outcomes. Neutral venues are not separately adjusted in this first
version; this is a limitation for international/neutral games.

## Timing and Coverage

All games in a season/week get features before any scores or statistics from
that week update state. PBP and tracking features average the last five team
games, allowing prior-season observations but not those more than a season old.
Missing observations do not become zero. Model imputation is fit on training
data only. Opponent ratings use the values available before the week.

The expanded dataset retains 1999-2026 history. PBP/NGS additions were collected
for 2016-2026; earlier rows retain missing values for them. The sixteen new
configurations train on 2017+, while original model schemas retain their
1999+ history. Existing GPR candidates now include 23 new numeric contrasts
alongside the previous 20 inputs; their regression window remains capped.

Current source coverage and download errors are in
`data/expanded_coverage_2026.json`. At this update, all 48 completed games have
PBP aggregates; NGS passing covers 96 team-weeks, rushing 85, and receiving 95,
through Week 3. Coverage is not a claim that every statistic is non-missing.
Historical inputs are revised public snapshots, not archived publication-time
data; retrospective results must not be described as live pregame accuracy.

## Evaluation and Models

The sixteen additions are eight estimator families with two feature views:
logistic regression, random forest, Extra Trees, histogram gradient boosting,
gradient boosting, AdaBoost, SGD logistic regression, and Gaussian Naive Bayes.
The views are compact team contrasts or full matchup context. Each configuration
has three validation trials. This is 16 configurations, not 16 distinct algorithms.
No market inputs are used in these additions.

Tune on 2025 after fitting earlier years; evaluate on the 48 completed 2026 games
with estimators fitted only through 2025; then refit with completed 2026 outcomes.
All original and GPR comparisons were refreshed to the same 48-game evaluation
set. Time windows and information sets still differ and are labeled. Weak models
are retained in the comparison; no winner is automatically promoted. A 48-game
retrospective sample cannot establish superiority. The production Random Forest
keeps its feature schema and algorithm, with a new Week 3 fit/version.

## Reproduce

```powershell
python refresh_predictions.py --season 2026
python retrain_current_season.py --original-only
python expanded_features.py --season 2026
python train_gpr.py --input data/nfl_matchups_1999_2026_expanded.csv
python train_expanded.py
python export_model_comparison.py
python refresh_predictions.py --season 2026
python -m unittest test_active_learning.py test_season_simulation.py test_gpr_models.py test_expanded_features.py
```

The regular Tuesday automation refreshes predictions, not model training.
Rerun expanded feature collection and training explicitly to update research
candidates. Model binaries are local except the existing production artifact;
code, coverage, datasets and evaluation reports are versioned. Chart data export
is part of the weekly refresh. It uses archived pregame probabilities and raw
future probabilities, not Monte Carlo output: the cumulative series fixes actual
completed wins and adds future probabilities; byes add nothing and ties add no win.
