# R18 Matched Evaluation

This is retrospective research, not a change to production models or saved picks.
All 64 recipes are compared on the same decisive games in each outer season:
2024, 2025 and available completed 2026 games. Ties are excluded. The historical
source already omits older ties and is not a separately audited complete schedule.

## Protocol

For outer season Y, use 2017 through Y-3 for inner base fitting, Y-2 for sigmoid
calibration, and Y-1 for hyperparameter selection by log loss. Refit the selected
base through Y-2, calibrate on Y-1, and evaluate Y. No outer labels enter fitting
or selection. Date checks reject overlapping partitions. GPR uses a recent
1,000-row regression window and the entire later calibration season.

Original AutoML recipes use 4 standard or 6 deep candidates drawn reproducibly
from the existing search spaces, not the original exhaustive search. Other
families retain their existing candidate sets. All non-GPR recipes receive the
same sigmoid-calibration protocol, including originally uncalibrated recipes.
These results evaluate re-tuned recipes, not the saved fitted production weights.

The baseline is the home-win frequency among pre-test training/calibration rows.
Paired log-loss and Brier differences use 1,000 bootstrap draws of season-week
blocks, seed 42. Negative differences favor the recipe. Intervals are not
adjusted for searching 64 configurations; choosing the best displayed result
introduces selection optimism. Prospective version-isolated evidence is still
required before any promotion. No model is promoted automatically.

Historical feature sources are revised snapshots, not point-in-time archives.
Market lines and weather availability are not proven pregame. Charted sources
start in 2022; missing features are imputed using training partitions. The
dashboard reports test missingness but this does not establish source freshness.

## Reproduction

Run `python matched_backtest.py` using the pinned environment in requirements.txt
(scikit-learn 1.8.0). `--seasons 2024 --limit 1` is a smoke run and cannot publish.
The full run writes `public_site/data/matched_comparison.json` and
`backtests/matched_predictions.csv`; per-fold caches under `backtests/matched/`
are ignored. Registry equality, shared game IDs and unique labels are checked.
Caches are keyed by source CSV hashes, seasons and the runner script, not by
transitive imported modules or artifact metadata. Use a fresh cache directory
after changing estimator factories, feature metadata or dependency versions.

Local prerequisites include `data/research_matchups.csv` (build with
`research_dataset.py` from privately retained source inputs), the expanded CSV,
and original/expanded joblib artifacts for their feature-column metadata. Some
private research sources and local artifacts are deliberately not in Git; a
fresh clone alone is not sufficient to rerun training. Only load trusted joblib
files. Public exports contain metrics, fold cutoffs, selected settings and source
hashes; the prediction CSV enables independent metric recomputation.

R19-R22 remain separate proposed work. Weekly refresh does not rerun this study.
