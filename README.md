# NFL Matchup Prediction Starter

This project uses `nflreadpy` to collect historical NFL schedule/results data and build a clean matchup-level dataset for modeling whether the home team wins.

`nflreadpy` returns Polars DataFrames from nflverse data. The dataset script converts schedules to pandas and creates pre-game rolling team features. Separate scripts train/save a model and load that model for predictions.

## Setup

Install Python 3.10+ first, then from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Build Dataset

```powershell
python build_matchup_dataset.py --start-season 1999 --end-season 2025
```

Outputs:

- `data/nfl_matchups_1999_2025.csv`

The script uses only information available before each game for team form features. That matters: using final season stats or game stats from the same matchup would leak the answer into the model.

## Build Advanced Dataset

```powershell
python build_matchup_dataset.py --start-season 1999 --end-season 2025 --advanced
```

Output:

- `data/nfl_matchups_1999_2025_advanced.csv`

Advanced features include:

- Rest days and rest advantage
- Rolling offensive/defensive EPA, success rate, explosive play rate, turnover/takeaway rate, pass rate, pass EPA, and rush EPA from play-by-play
- Prior-season EPA carryover features
- Rolling QB passing volume, efficiency, EPA, CPOE, interceptions, sacks, and prior-starter continuity
- Injury report counts, including QB/skill/line injury flags where available
- Weekly roster strength proxies such as roster size, active count, average experience, rookie count, QB count, OL count, and skill-position count

Source coverage is not identical across all years. Play-by-play and player stats cover the full 1999-2025 build here, weekly rosters start in 2002, and injuries start in 2009. Missing early-year feature values are handled by the model imputer.

## Train And Save A Model

### Refit Through 2026

Run `python retrain_current_season.py` after refreshing completed scores. This refits all eight saved production/baseline configurations (excluding the smoke-test artifact), using the same feature columns and tuned hyperparameters. It reconstructs 2026 pregame score-based form with week cutoffs and carries historical advanced snapshots forward. It does not fetch fresh 2026 EPA, injury, QB, or roster inputs.

Each configuration is first fitted through 2025 and evaluated on completed 2026 games. It is then refitted on all 6,984 games through September 21, 2026, including 32 current-season games. The reported pre-refit metrics are NOT an independent evaluation of the final model, which now includes those labels. Historical pregame picks remain frozen. Models before replacement are backed up under `models/archive/`; the full comparison is `models/retraining_2026.json`.

Run `python refresh_predictions.py --season 2026` afterward to score upcoming games. Each new forecast records its model version. Calibration learning starts a separate cohort under `learning/generations/<model_version>/`, leaving old learning history intact. No old-model seed labels or shadow metrics are reused as evidence for the new version.

```powershell
python train_model.py --input data/nfl_matchups_1999_2025.csv
```

Outputs:

- `models/home_win_logreg.joblib`
- `models/home_win_logreg.metrics.json`

By default, the latest season in the dataset is used as the holdout test set. Add `--exclude-market` if you do not want the model to use `spread_line` or `total_line`.

## AutoML Search

Run an AutoML-style search across logistic regression, random forest, extra trees, and histogram gradient boosting:

```powershell
python train_automl.py --input data/nfl_matchups_1999_2025.csv --n-iter 40
```

Outputs:

- `models/home_win_automl.joblib`
- `models/home_win_automl.metrics.json`

The search optimizes validation log loss using a time-aware split: it holds out the latest season for final testing and uses the prior season for model selection.

Advanced AutoML:

```powershell
python train_automl.py --input data/nfl_matchups_1999_2025_advanced.csv --n-iter 40 --model-name home_win_advanced_automl.joblib
python train_automl.py --input data/nfl_matchups_1999_2025_advanced.csv --n-iter 40 --exclude-market --model-name home_win_advanced_automl_no_market.joblib
python train_automl.py --input data/nfl_matchups_1999_2025_advanced.csv --n-iter 20 --complexity deep --exclude-market --model-name home_win_advanced_automl_no_market_deep.joblib
```

Use the no-market model for schedule-only future predictions, because future spreads and totals are often unavailable.

## Make Predictions

### Active Learning

`refresh_predictions.py` now also runs `active_learning.py`. The learner queries up to eight games in the next unplayed week: six nearest 50% win probability plus two deterministic random exploration picks. Official results supply labels; the query controls which later games enter candidate training. All completed games remain in production accuracy reporting, including games outside the query.

The first 16 available results seed a regularized logistic probability-calibration candidate using the archived baseline probability as its sole input. This is an active-learning-inspired calibration experiment, not a retrained Random Forest or updated EPA/QB model. `learning/candidate.joblib`, `state.json`, `summary.json`, and `evaluation.csv` persist training selections and shadow forecasts. Repeated refreshes do not duplicate labels. Initial seed games are never counted as prospective validation. Unknown outcomes and ties are excluded from binary training and accuracy.

Future candidate forecasts are saved before results, then scored against the baseline before refitting. Query selection never uses final scores or correctness. Same-day games are excluded from new shadow forecasts because source kickoff times are not consistently available. At least 64 prospective evaluated games and lower log loss AND Brier score flag a candidate for review; this is a screening rule, not statistical proof. Production stays on the original model until a separate evaluation and promotion. No improvement is claimed from the seed sample.

Run `python active_learning.py` to update the candidate from existing prediction files, or `python -m unittest test_active_learning.py` for grading, tie, sampling, and prospective-evaluation checks.

Refresh completed scores, score-based team form, upcoming forecasts, and public dashboard data:

```powershell
python refresh_predictions.py --season 2026
```

This preserves saved pregame picks for completed games and grades them Yes, No, Tie (no winner), or Pending. It does not retrain the production model or refresh advanced EPA, QB, injury, and roster snapshots; that limitation is shown on the public dashboard. It updates the separate shadow learner. The weekly Codex refresh runs Tuesday at 9 a.m. America/New_York and publishes updated assets.

Score a known game row:

```powershell
python predict_matchup.py --game-id 2024_01_BAL_KC
```

Score a manual matchup using each team's current-season form before that week:

```powershell
python predict_matchup.py --home-team BUF --away-team KC --season 2025 --week 12
```

## Predict A Full Season Schedule

Use `auto` to try `nflreadpy` first and then fall back to ESPN's public scoreboard schedule API:

```powershell
python predict_schedule.py --season 2026 --model models/home_win_automl.joblib
```

Advanced no-market schedule prediction:

```powershell
python predict_schedule.py --season 2026 --history data/nfl_matchups_1999_2025_advanced.csv --model models/home_win_advanced_automl_no_market.joblib --output predictions/nfl_2026_predictions_advanced_no_market.csv
python predict_schedule.py --season 2026 --history data/nfl_matchups_1999_2025_advanced.csv --model models/home_win_advanced_automl_no_market_deep.joblib --output predictions/nfl_2026_predictions_advanced_no_market_deep.csv
```

## SHAP Explanations

Generate global and local SHAP explanation tables:

```powershell
python explain_model.py --model models/home_win_advanced_automl_no_market.joblib --input data/nfl_matchups_1999_2025_advanced.csv --game-id 2025_01_DAL_PHI
```

Outputs:

- `explanations/global_shap_2025.csv`
- `explanations/local_shap_2025_01_DAL_PHI.csv`

SHAP values explain movement toward the home team winning. Positive values help the home team; negative values help the away team. The `feature_value` column is the transformed model input value, so numeric fields may be standardized and categorical fields may appear as one-hot encoded columns.

## Web Dashboard

Run the local Streamlit dashboard:

```powershell
python -m streamlit run app.py --server.port 8501
```

Then open:

- `http://localhost:8501`

The dashboard includes schedule filters, prediction confidence, predicted-winner summaries, global SHAP feature importance, and selected-game SHAP drivers.

Team logo metadata is cached locally in `data/team_metadata.csv` so the dashboard does not need to download logo metadata at startup.

To force a source:

```powershell
python predict_schedule.py --season 2026 --source nflreadpy
python predict_schedule.py --season 2026 --source espn
```

You can also provide a CSV with at least `season`, `week`, `game_id`, `home_team`, and `away_team`:

```powershell
python predict_schedule.py --season 2026 --schedule-csv data/2026_schedule.csv --model models/home_win_automl.joblib
```

Output:

- `predictions/nfl_2026_predictions.csv`

## Good Next Features

- Play-by-play EPA/offensive success rate from `nfl.load_pbp(...)`
- Weekly rosters/injuries from `nfl.load_rosters_weekly(...)` and `nfl.load_injuries(...)`
- QB starter continuity and passing efficiency
- Rest/travel/time-zone effects
- Market features such as spread and total, if you want a model that incorporates betting-market information
