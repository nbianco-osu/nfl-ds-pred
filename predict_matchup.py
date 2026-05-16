from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from prediction_features import fill_latest_team_state, make_prediction_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the home-team win probability for an NFL matchup.")
    parser.add_argument("--model", type=Path, default=Path("models/home_win_logreg.joblib"))
    parser.add_argument("--input", type=Path, default=Path("data/nfl_matchups_1999_2025.csv"))
    parser.add_argument("--game-id", help="Score an existing row from the matchup dataset.")
    parser.add_argument("--home-team", help="Home team abbreviation, for example BUF.")
    parser.add_argument("--away-team", help="Away team abbreviation, for example KC.")
    parser.add_argument("--season", type=int, help="Season for a manual matchup.")
    parser.add_argument("--week", type=int, help="Week for a manual matchup.")
    parser.add_argument("--roof", default="outdoors")
    parser.add_argument("--surface", default="fieldturf")
    parser.add_argument("--temp", type=float)
    parser.add_argument("--wind", type=float)
    parser.add_argument("--div-game", type=int, default=0)
    parser.add_argument("--spread-line", type=float)
    parser.add_argument("--total-line", type=float)
    return parser.parse_args()


def make_manual_row(matchups: pd.DataFrame, args: argparse.Namespace, feature_cols: list[str]) -> pd.DataFrame:
    required = ["home_team", "away_team", "season", "week"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing required manual matchup arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}")

    row = make_prediction_row(
        matchups=matchups,
        feature_cols=feature_cols,
        home_team=args.home_team,
        away_team=args.away_team,
        season=args.season,
        week=args.week,
        roof=args.roof,
        surface=args.surface,
        temp=args.temp,
        wind=args.wind,
        div_game=args.div_game,
        spread_line=args.spread_line,
        total_line=args.total_line,
    )
    return fill_latest_team_state(
        row,
        matchups,
        feature_cols,
        home_team=args.home_team,
        away_team=args.away_team,
        season=args.season,
        week=args.week,
    )


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.model)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    matchups = pd.read_csv(args.input)

    if args.game_id:
        candidates = matchups[matchups["game_id"].eq(args.game_id)]
        if candidates.empty:
            raise ValueError(f"No matchup row found for game_id={args.game_id}")
        row = candidates.iloc[[0]][feature_cols]
        label = args.game_id
        home_team = str(candidates.iloc[0]["home_team"])
        away_team = str(candidates.iloc[0]["away_team"])
    else:
        row = make_manual_row(matchups, args, feature_cols)
        label = f"{args.away_team} at {args.home_team}, {args.season} week {args.week}"
        home_team = str(args.home_team)
        away_team = str(args.away_team)

    probability = float(model.predict_proba(row)[:, 1][0])
    winner = home_team if probability >= 0.5 else away_team

    print(f"Matchup: {label}")
    print(f"Predicted winner: {winner}")
    print(f"{home_team} win probability: {probability:.3f}")
    print(f"{away_team} win probability: {1 - probability:.3f}")


if __name__ == "__main__":
    main()
