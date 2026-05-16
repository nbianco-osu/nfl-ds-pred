from __future__ import annotations

import pandas as pd
import numpy as np


TEAM_FEATURES = [
    "games_played",
    "win_pct",
    "point_diff_per_game",
    "points_for_per_game",
    "points_against_per_game",
    "last_5_point_diff",
]


def prior_games(matchups: pd.DataFrame, team: str, season: int, week: int) -> pd.DataFrame:
    before_cutoff = (matchups["season"] < season) | (
        matchups["season"].eq(season) & matchups["week"].lt(week)
    )
    rows = matchups[before_cutoff & (matchups["home_team"].eq(team) | matchups["away_team"].eq(team))].copy()
    rows = rows.sort_values(["season", "week", "game_id"])
    return rows[rows["season"].eq(season)]


def summarize_team(matchups: pd.DataFrame, team: str, season: int, week: int) -> dict[str, float]:
    games = prior_games(matchups, team, season, week)
    if games.empty:
        return {
            "games_played": 0,
            "win_pct": 0.0,
            "point_diff_per_game": 0.0,
            "points_for_per_game": 0.0,
            "points_against_per_game": 0.0,
            "last_5_point_diff": 0.0,
        }

    is_home = games["home_team"].eq(team)
    points_for = games["home_score"].where(is_home, games["away_score"])
    points_against = games["away_score"].where(is_home, games["home_score"])
    point_diff = points_for - points_against
    wins = (points_for > points_against).astype(int)

    return {
        "games_played": int(len(games)),
        "win_pct": float(wins.mean()),
        "point_diff_per_game": float(point_diff.mean()),
        "points_for_per_game": float(points_for.mean()),
        "points_against_per_game": float(points_against.mean()),
        "last_5_point_diff": float(point_diff.tail(5).mean()),
    }


def make_prediction_row(
    matchups: pd.DataFrame,
    feature_cols: list[str],
    home_team: str,
    away_team: str,
    season: int,
    week: int,
    game_type: str = "REG",
    roof: str | None = None,
    surface: str | None = None,
    temp: float | None = None,
    wind: float | None = None,
    div_game: int | None = None,
    spread_line: float | None = None,
    total_line: float | None = None,
) -> pd.DataFrame:
    row: dict[str, object] = {col: np.nan for col in feature_cols}
    row.update(
        {
            "season": season,
            "week": week,
            "game_type": game_type,
            "home_team": home_team,
            "away_team": away_team,
            "roof": roof,
            "surface": surface,
            "temp": temp,
            "wind": wind,
            "div_game": div_game,
            "spread_line": spread_line,
            "total_line": total_line,
        }
    )

    home_summary = summarize_team(matchups, home_team, season, week)
    away_summary = summarize_team(matchups, away_team, season, week)
    for feature in TEAM_FEATURES:
        row[f"home_team_{feature}"] = home_summary[feature]
        row[f"away_team_{feature}"] = away_summary[feature]
        row[f"diff_{feature}"] = home_summary[feature] - away_summary[feature]

    return pd.DataFrame([row], columns=feature_cols)


def _latest_team_row(matchups: pd.DataFrame, team: str, season: int, week: int) -> pd.Series | None:
    before_cutoff = (matchups["season"] < season) | (
        matchups["season"].eq(season) & matchups["week"].lt(week)
    )
    rows = matchups[before_cutoff & (matchups["home_team"].eq(team) | matchups["away_team"].eq(team))].copy()
    if rows.empty:
        return None
    return rows.sort_values(["season", "week", "game_id"]).iloc[-1]


def fill_latest_team_state(
    row: pd.DataFrame,
    matchups: pd.DataFrame,
    feature_cols: list[str],
    home_team: str,
    away_team: str,
    season: int,
    week: int,
) -> pd.DataFrame:
    home_latest = _latest_team_row(matchups, home_team, season, week)
    away_latest = _latest_team_row(matchups, away_team, season, week)

    def source_value(latest: pd.Series | None, team: str, suffix: str) -> object:
        if latest is None:
            return np.nan
        side = "home" if latest.get("home_team") == team else "away"
        return latest.get(f"{side}_{suffix}", np.nan)

    updates: dict[str, object] = {}
    for col in feature_cols:
        if col.startswith("home_team_") and pd.isna(row.at[0, col]):
            updates[col] = source_value(home_latest, home_team, col.removeprefix("home_"))
        elif col.startswith("away_team_") and pd.isna(row.at[0, col]):
            updates[col] = source_value(away_latest, away_team, col.removeprefix("away_"))

    for col, value in updates.items():
        row.at[0, col] = value

    for col in feature_cols:
        if not col.startswith("diff_") or not pd.isna(row.at[0, col]):
            continue
        suffix = col.removeprefix("diff_")
        home_col = f"home_{suffix}"
        away_col = f"away_{suffix}"
        if home_col in row.columns and away_col in row.columns:
            home_value = row.at[0, home_col]
            away_value = row.at[0, away_col]
            if not pd.isna(home_value) and not pd.isna(away_value):
                row.at[0, col] = home_value - away_value

    return row
