# Deployment

This app is ready for Streamlit Community Cloud.

## Required Files

The deployed app needs these generated artifacts in the repository:

- `data/nfl_matchups_1999_2025_advanced.csv`
- `data/team_metadata.csv`
- `models/home_win_advanced_automl_no_market_deep.joblib`
- `models/home_win_advanced_automl_no_market_deep.metrics.json`
- `predictions/nfl_2026_predictions_advanced_no_market_deep.csv`
- `explanations/global_shap_2025.csv`

## Streamlit Community Cloud

1. Push this folder to GitHub.
2. Go to `https://share.streamlit.io/`.
3. Choose the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.

No secrets are required for the current app. The app reads local model/data artifacts from the repo.

## Important Notes

This is not a betting app. It is an educational predictive analytics dashboard.

Team names and logos belong to their respective owners and are used only for team identification in a non-commercial, informational context.
