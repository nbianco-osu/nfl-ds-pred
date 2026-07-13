# Deployment

This app is ready for Streamlit Community Cloud.

The repository also contains a static public dashboard in `public_site/` that can be deployed with GitHub Pages.

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

## GitHub Pages Static Site

The static public dashboard is generated from precomputed prediction and explanation artifacts and does not require a Python server.

Public site source:

- `public_site/index.html`
- `public_site/styles.css`
- `public_site/app.js`
- `public_site/data/*.json`

The GitHub Pages workflow is:

- `.github/workflows/pages.yml`

After pushing to `main`, open the repository settings in GitHub and make sure Pages is enabled for GitHub Actions. The expected public URL is:

- `https://nbianco-osu.github.io/nfl-ds-pred/`
