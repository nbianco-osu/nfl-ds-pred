from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

from advanced_features import add_advanced_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an NFL matchup training dataset from nflreadpy schedules."
    )
    parser.add_argument("--start-season", type=int, default=1999)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument(
        "--include-playoffs",
        action="store_true",
        help="Include postseason games in addition to regular season games.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Add EPA/play-by-play, QB, rest, injury, and roster-strength features.",
    )
    return parser.parse_args()


def load_games(start_season: int, end_season: int, include_playoffs: bool) -> pd.DataFrame:
    seasons = list(range(start_season, end_season + 1))
    schedules = nfl.load_schedules(seasons).to_pandas()

    games = schedules.copy()
    games = games[games["home_score"].notna() & games["away_score"].notna()]

    if "game_type" in games.columns and not include_playoffs:
        games = games[games["game_type"].eq("REG")]

    games["home_score"] = games["home_score"].astype(int)
    games["away_score"] = games["away_score"].astype(int)
    games = games[games["home_score"].ne(games["away_score"])].copy()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    sort_cols = [col for col in ["season", "week", "gameday", "game_id"] if col in games.columns]
    return games.sort_values(sort_cols).reset_index(drop=True)


def make_team_games(games: pd.DataFrame) -> pd.DataFrame:
    base_cols = [col for col in ["game_id", "season", "week", "gameday"] if col in games.columns]

    home = games[base_cols + ["home_team", "away_team", "home_score", "away_score"]].copy()
    home = home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "points_for",
            "away_score": "points_against",
        }
    )
    home["is_home"] = 1

    away = games[base_cols + ["away_team", "home_team", "away_score", "home_score"]].copy()
    away = away.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "points_for",
            "home_score": "points_against",
        }
    )
    away["is_home"] = 0

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["win"] = (team_games["points_for"] > team_games["points_against"]).astype(int)
    team_games["point_diff"] = team_games["points_for"] - team_games["points_against"]
    team_games = team_games.sort_values(["team", "season", "week", "game_id"]).reset_index(drop=True)
    return team_games


def add_pregame_team_features(team_games: pd.DataFrame) -> pd.DataFrame:
    grouped = team_games.groupby(["team", "season"], group_keys=False)
    features = team_games[["game_id", "team", "season", "week"]].copy()
    prior_games = features["team_games_played"] = grouped.cumcount()
    denominator = prior_games.mask(prior_games.eq(0))

    features["team_win_pct"] = (
        grouped["win"].cumsum().groupby([team_games["team"], team_games["season"]]).shift(fill_value=0)
        / denominator
    )
    features["team_point_diff_per_game"] = (
        grouped["point_diff"].cumsum().groupby([team_games["team"], team_games["season"]]).shift(fill_value=0)
        / denominator
    )
    features["team_points_for_per_game"] = (
        grouped["points_for"].cumsum().groupby([team_games["team"], team_games["season"]]).shift(fill_value=0)
        / denominator
    )
    features["team_points_against_per_game"] = (
        grouped["points_against"].cumsum().groupby([team_games["team"], team_games["season"]]).shift(fill_value=0)
        / denominator
    )

    shifted_diff = grouped["point_diff"].shift(1)
    features["team_last_5_point_diff"] = (
        shifted_diff.groupby([team_games["team"], team_games["season"]])
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    return features.fillna(0)


def build_matchups(games: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    home_features = team_features.add_prefix("home_").rename(
        columns={
            "home_game_id": "game_id",
            "home_team": "home_team",
            "home_season": "season",
            "home_week": "week",
        }
    )
    away_features = team_features.add_prefix("away_").rename(
        columns={
            "away_game_id": "game_id",
            "away_team": "away_team",
            "away_season": "season",
            "away_week": "week",
        }
    )

    keep_cols = [
        col
        for col in [
            "game_id",
            "season",
            "week",
            "gameday",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "home_win",
            "roof",
            "surface",
            "temp",
            "wind",
            "div_game",
            "spread_line",
            "total_line",
        ]
        if col in games.columns
    ]
    matchups = games[keep_cols].copy()
    matchups = matchups.merge(home_features, on=["game_id", "season", "week", "home_team"], how="left")
    matchups = matchups.merge(away_features, on=["game_id", "season", "week", "away_team"], how="left")

    for feature in [
        "win_pct",
        "point_diff_per_game",
        "points_for_per_game",
        "points_against_per_game",
        "last_5_point_diff",
        "games_played",
    ]:
        home_col = f"home_team_{feature}"
        away_col = f"away_team_{feature}"
        if home_col in matchups.columns and away_col in matchups.columns:
            matchups[f"diff_{feature}"] = matchups[home_col] - matchups[away_col]

    return matchups


def main() -> None:
    args = parse_args()
    games = load_games(args.start_season, args.end_season, args.include_playoffs)
    team_games = make_team_games(games)
    team_features = add_pregame_team_features(team_games)
    matchups = build_matchups(games, team_features)
    if args.advanced:
        seasons = list(range(args.start_season, args.end_season + 1))
        matchups = add_advanced_features(matchups, games, seasons)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_advanced" if args.advanced else ""
    output_path = args.output_dir / f"nfl_matchups_{args.start_season}_{args.end_season}{suffix}.csv"
    matchups.to_csv(output_path, index=False)

    print(f"Wrote {len(matchups):,} games to {output_path}")


if __name__ == "__main__":
    main()
