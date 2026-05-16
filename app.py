from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from prediction_features import fill_latest_team_state, make_prediction_row
from shap_utils import global_importance, local_top_features


ROOT = Path(__file__).parent
DEFAULT_MODEL = ROOT / "models" / "home_win_advanced_automl_no_market_deep.joblib"
DEFAULT_HISTORY = ROOT / "data" / "nfl_matchups_1999_2025_advanced.csv"
DEFAULT_PREDICTIONS = ROOT / "predictions" / "nfl_2026_predictions_advanced_no_market_deep.csv"
DEFAULT_TEAMS = ROOT / "data" / "team_metadata.csv"


st.set_page_config(page_title="NFL Prediction Dashboard", layout="wide")


@st.cache_resource
def load_model(path: str) -> dict:
    return joblib.load(path)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_team_metadata() -> pd.DataFrame:
    teams = pd.read_csv(DEFAULT_TEAMS)
    return teams[["team_abbr", "team_name", "team_logo_espn"]].drop_duplicates("team_abbr")


@st.cache_data(show_spinner=False)
def cached_global_importance(model_path: str, history_path: str, season: int) -> pd.DataFrame:
    cached_path = ROOT / "explanations" / f"global_shap_{season}.csv"
    if cached_path.exists():
        return pd.read_csv(cached_path)
    artifact = joblib.load(model_path)
    history = pd.read_csv(history_path)
    rows = history[history["season"].eq(season)].copy()
    return global_importance(artifact, rows, max_rows=750)


def prediction_feature_row(artifact: dict, history: pd.DataFrame, game: pd.Series) -> pd.DataFrame:
    row = make_prediction_row(
        matchups=history,
        feature_cols=artifact["feature_cols"],
        home_team=str(game["home_team"]),
        away_team=str(game["away_team"]),
        season=int(game["season"]),
        week=int(game["week"]),
    )
    return fill_latest_team_state(
        row,
        history,
        artifact["feature_cols"],
        home_team=str(game["home_team"]),
        away_team=str(game["away_team"]),
        season=int(game["season"]),
        week=int(game["week"]),
    )


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def enrich_predictions(predictions: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    team_names = teams.set_index("team_abbr")["team_name"].to_dict()
    team_logos = teams.set_index("team_abbr")["team_logo_espn"].to_dict()
    enriched = predictions.copy()
    enriched["away_team_name"] = enriched["away_team"].map(team_names).fillna(enriched["away_team"])
    enriched["home_team_name"] = enriched["home_team"].map(team_names).fillna(enriched["home_team"])
    enriched["predicted_winner_name"] = enriched["predicted_winner"].map(team_names).fillna(enriched["predicted_winner"])
    enriched["away_logo"] = enriched["away_team"].map(team_logos)
    enriched["home_logo"] = enriched["home_team"].map(team_logos)
    enriched["winner_logo"] = enriched["predicted_winner"].map(team_logos)
    enriched["matchup"] = enriched["away_team_name"] + " at " + enriched["home_team_name"]
    enriched["confidence"] = enriched[["home_win_probability", "away_win_probability"]].max(axis=1)
    enriched["home_win_pct"] = 100 * enriched["home_win_probability"]
    enriched["away_win_pct"] = 100 * enriched["away_win_probability"]
    enriched["confidence_pct"] = 100 * enriched["confidence"]
    return enriched


FEATURE_KEY = pd.DataFrame(
    [
        {
            "Term": "SHAP value",
            "Meaning": "How much a feature moves the prediction toward the home team or away team.",
        },
        {
            "Term": "Positive SHAP",
            "Meaning": "Pushes the model toward the home team winning.",
        },
        {
            "Term": "Negative SHAP",
            "Meaning": "Pushes the model toward the away team winning.",
        },
        {
            "Term": "diff",
            "Meaning": "Home team value minus away team value. Positive means the home team has the edge.",
        },
        {
            "Term": "EPA",
            "Meaning": "Expected Points Added. Higher offensive EPA is better; lower defensive EPA allowed is better.",
        },
        {
            "Term": "Success rate",
            "Meaning": "Share of plays with positive EPA.",
        },
        {
            "Term": "Explosive rate",
            "Meaning": "Share of high-gain plays: 20+ yard passes or 10+ yard runs.",
        },
        {
            "Term": "last 3 / last 5",
            "Meaning": "Rolling average from the team's previous 3 or 5 games before the matchup.",
        },
        {
            "Term": "Previous season",
            "Meaning": "Prior-year team or QB signal carried into the next season.",
        },
    ]
)


FEATURE_REPLACEMENTS = [
    ("team_pbp_prev_season_", "previous season play-by-play "),
    ("team_qb_prev_season_", "previous season QB "),
    ("team_pbp_", "play-by-play "),
    ("team_qb_", "QB "),
    ("team_injury_", "injury report "),
    ("team_roster_", "roster "),
    ("off_epa_per_play", "offensive EPA per play"),
    ("def_epa_allowed_per_play", "defensive EPA allowed per play"),
    ("off_success_rate", "offensive success rate"),
    ("def_success_allowed_rate", "defensive success rate allowed"),
    ("off_explosive_rate", "offensive explosive play rate"),
    ("def_explosive_allowed_rate", "defensive explosive play rate allowed"),
    ("off_turnover_rate", "offensive turnover rate"),
    ("def_takeaway_rate", "defensive takeaway rate"),
    ("off_pass_rate", "offensive pass rate"),
    ("off_pass_epa", "offensive pass EPA"),
    ("off_rush_epa", "offensive rush EPA"),
    ("point_diff_per_game", "point differential per game"),
    ("points_for_per_game", "points scored per game"),
    ("points_against_per_game", "points allowed per game"),
    ("last_5_point_diff", "last five game point differential"),
    ("games_played", "games played"),
    ("win_pct", "win percentage"),
    ("rest_days", "rest days"),
    ("passing_yards", "passing yards"),
    ("passing_tds", "passing touchdowns"),
    ("passing_interceptions", "passing interceptions"),
    ("sacks_suffered", "sacks taken"),
    ("passing_epa", "passing EPA"),
    ("passing_cpoe", "completion percentage over expected"),
    ("prev_starter_same_as_prev_game", "QB starter continuity"),
    ("report_count", "players listed on injury report"),
    ("out_count", "players listed out"),
    ("questionable_count", "players listed questionable"),
    ("qb_out", "QB listed out"),
    ("skill_out_count", "skill-position players listed out"),
    ("line_out_count", "line players listed out"),
    ("active_count", "active roster count"),
    ("avg_years_exp", "average years of experience"),
    ("rookie_count", "rookie count"),
    ("qb_count", "QB count"),
    ("ol_count", "offensive line count"),
    ("skill_count", "skill-position count"),
]


def readable_feature_name(feature: str) -> str:
    name = feature.replace("num__", "").replace("cat__", "")
    side = ""
    if name.startswith("diff_"):
        side = "Home edge in "
        name = name.removeprefix("diff_")
    elif name.startswith("home_"):
        side = "Home team "
        name = name.removeprefix("home_")
    elif name.startswith("away_"):
        side = "Away team "
        name = name.removeprefix("away_")

    for old, new in FEATURE_REPLACEMENTS:
        name = name.replace(old, new)

    name = name.replace("_last_3", ", last 3 games")
    name = name.replace("_last_5", ", last 5 games")
    name = name.replace("_avg", ", season-to-date average")
    name = name.replace("_", " ")
    return (side + name).strip().title()


def add_feature_labels(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()
    labeled["Readable Feature"] = labeled["feature"].map(readable_feature_name)
    return labeled


st.title("NFL Matchup Prediction Dashboard")

with st.sidebar:
    st.header("Data")
    model_path = st.text_input("Model artifact", str(DEFAULT_MODEL))
    history_path = st.text_input("Advanced history CSV", str(DEFAULT_HISTORY))
    predictions_path = st.text_input("Predictions CSV", str(DEFAULT_PREDICTIONS))
    st.caption("Use the advanced no-market model for future schedules unless you have spreads/totals.")

artifact = load_model(model_path)
history = load_csv(history_path)
teams = load_team_metadata()
predictions = enrich_predictions(load_csv(predictions_path), teams)
metrics = artifact.get("metrics", {})

top = st.columns(4)
top[0].metric("Predicted Games", f"{len(predictions):,}")
top[1].metric("Model", type(artifact["model"].named_steps["model"]).__name__)
top[2].metric("Holdout Log Loss", f"{metrics.get('log_loss', float('nan')):.3f}")
top[3].metric("Holdout ROC AUC", f"{metrics.get('roc_auc', float('nan')):.3f}")

schedule_tab, model_tab, game_tab = st.tabs(["Schedule", "Model Signals", "Game Explanation"])

with schedule_tab:
    left, right = st.columns([1, 3])
    with left:
        weeks = sorted(predictions["week"].dropna().astype(int).unique())
        selected_weeks = st.multiselect("Weeks", weeks, default=weeks[:3])
        teams = sorted(set(predictions["home_team"]).union(set(predictions["away_team"])))
        selected_teams = st.multiselect("Teams", teams)
        min_confidence = st.slider("Minimum confidence", 0.50, 0.90, 0.50, 0.01)

    filtered = predictions[predictions["week"].isin(selected_weeks)].copy()
    if selected_teams:
        filtered = filtered[
            filtered["home_team"].isin(selected_teams) | filtered["away_team"].isin(selected_teams)
        ]
    filtered = filtered[filtered["confidence"].ge(min_confidence)]

    with right:
        st.plotly_chart(
            px.histogram(filtered, x="predicted_winner_name", title="Predicted Winners In Filtered Schedule")
            .update_layout(xaxis_title=None, yaxis_title="Games"),
            width="stretch",
        )

    table = filtered[
        [
            "week",
            "gameday",
            "away_logo",
            "away_team_name",
            "home_logo",
            "home_team_name",
            "winner_logo",
            "predicted_winner_name",
            "away_win_pct",
            "home_win_pct",
            "confidence_pct",
        ]
    ].sort_values(["week", "gameday"])
    st.dataframe(
        table.style.format(
            {
                "away_win_pct": "{:.1f}%",
                "home_win_pct": "{:.1f}%",
                "confidence_pct": "{:.1f}%",
            }
        ),
        column_config={
            "week": st.column_config.NumberColumn("Week", format="%d"),
            "gameday": st.column_config.TextColumn("Game Date"),
            "away_logo": st.column_config.ImageColumn("Away"),
            "away_team_name": st.column_config.TextColumn("Away Team"),
            "home_logo": st.column_config.ImageColumn("Home"),
            "home_team_name": st.column_config.TextColumn("Home Team"),
            "winner_logo": st.column_config.ImageColumn("Pick"),
            "predicted_winner_name": st.column_config.TextColumn("Predicted Winner"),
            "away_win_pct": st.column_config.NumberColumn("Away Win %", format="%.1f%%"),
            "home_win_pct": st.column_config.NumberColumn("Home Win %", format="%.1f%%"),
            "confidence_pct": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
        },
        hide_index=True,
        width="stretch",
        height=520,
    )

with model_tab:
    season = st.selectbox("Global explanation season", sorted(history["season"].unique(), reverse=True), index=0)
    with st.expander("Feature Key", expanded=True):
        st.dataframe(FEATURE_KEY, hide_index=True, width="stretch")
    importance = add_feature_labels(cached_global_importance(model_path, history_path, int(season))).head(30)
    st.plotly_chart(
        px.bar(
            importance.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="Readable Feature",
            orientation="h",
            title=f"Top Global SHAP Features, {season}",
        ).update_layout(xaxis_title="Mean absolute SHAP value", yaxis_title=None),
        width="stretch",
    )
    importance_table = importance.rename(
        columns={
            "mean_abs_shap": "Average Impact",
            "mean_value": "Average Model Input Value",
            "feature": "Raw Feature Name",
        }
    )[["Readable Feature", "Average Impact", "Average Model Input Value", "Raw Feature Name"]]
    st.dataframe(
        importance_table,
        column_config={
            "Average Impact": st.column_config.NumberColumn("Average Impact", format="%.4f"),
            "Average Model Input Value": st.column_config.NumberColumn("Average Model Input Value", format="%.3f"),
        },
        hide_index=True,
        width="stretch",
        height=420,
    )

with game_tab:
    game_labels = predictions.sort_values(["week", "gameday"])["matchup"] + " | Week " + predictions["week"].astype(str)
    selected_label = st.selectbox("Game", game_labels.tolist())
    game = predictions.loc[game_labels[game_labels.eq(selected_label)].index[0]]

    logo_left, summary, logo_right = st.columns([1, 2, 1])
    with logo_left:
        st.image(str(game["away_logo"]), width=96)
        st.subheader(str(game["away_team_name"]))
    with summary:
        st.metric("Predicted Winner", str(game["predicted_winner_name"]))
        st.caption(f"Week {int(game['week'])} | {game['gameday']}")
    with logo_right:
        st.image(str(game["home_logo"]), width=96)
        st.subheader(str(game["home_team_name"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", pct(float(game["confidence"])))
    c2.metric(f"{game['home_team_name']} Win Probability", pct(float(game["home_win_probability"])))
    c3.metric(f"{game['away_team_name']} Win Probability", pct(float(game["away_win_probability"])))

    with st.expander("How To Read This Explanation", expanded=True):
        st.markdown(
            """
            SHAP values show why this specific game prediction moved toward the home or away team.
            Positive bars help the home team. Negative bars help the away team. The raw feature name is kept in the table for auditability.
            """
        )
        st.dataframe(FEATURE_KEY, hide_index=True, width="stretch")

    if st.button("Generate Explanation For Selected Game"):
        row = prediction_feature_row(artifact, history, game)
        local = add_feature_labels(local_top_features(artifact, row, top_n=20))
        st.plotly_chart(
            px.bar(
                local.sort_values("shap_value"),
                x="shap_value",
                y="Readable Feature",
                color="impact",
                orientation="h",
                title="Top SHAP Drivers For Selected Game",
                color_discrete_map={"Helps home team": "#1f77b4", "Helps away team": "#d62728"},
            ).update_layout(xaxis_title="SHAP value toward home-team win probability", yaxis_title=None),
            width="stretch",
        )
        local_table = local.rename(
            columns={
                "shap_value": "SHAP Impact",
                "feature_value": "Model Input Value",
                "impact": "Direction",
                "feature": "Raw Feature Name",
            }
        )[["Readable Feature", "Direction", "SHAP Impact", "Model Input Value", "Raw Feature Name"]]
        st.dataframe(
            local_table,
            column_config={
                "SHAP Impact": st.column_config.NumberColumn("SHAP Impact", format="%.4f"),
                "Model Input Value": st.column_config.NumberColumn("Model Input Value", format="%.3f"),
            },
            hide_index=True,
            width="stretch",
            height=460,
        )
    else:
        st.info("Select a game, then generate the SHAP explanation when you want to inspect its top drivers.")

st.divider()
st.subheader("Disclaimers And Fair Use")
st.markdown(
    """
    This dashboard is an educational predictive analytics tool for exploring NFL data science workflows.
    It is not a betting site, sportsbook, gambling service, financial advice service, or guarantee of future outcomes.
    Predictions are probabilistic estimates from historical public data and should be treated as research outputs, not instructions.

    Team names, logos, marks, and related identifiers belong to their respective owners. Logos are displayed only to identify
    teams and improve readability in a non-commercial, informational context. This project is not affiliated with, endorsed by,
    or sponsored by the NFL, any NFL club, ESPN, or nflverse. If you reuse or publish this dashboard, keep attribution to the
    underlying data sources, avoid implying endorsement, and remove or replace trademarked assets if your use requires permission.
    """
)
