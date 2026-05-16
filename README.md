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
