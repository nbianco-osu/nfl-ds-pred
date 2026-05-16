from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import nflreadpy as nfl
import pandas as pd
import requests

from prediction_features import fill_latest_team_state, make_prediction_row


ESPN_TO_NFLVERSE_TEAM = {
    "LAR": "LA",
    "WSH": "WAS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict every scheduled NFL game for a season.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--model", type=Path, default=Path("models/home_win_automl.joblib"))
    parser.add_argument("--history", type=Path, default=Path("data/nfl_matchups_1999_2025.csv"))
    parser.add_argument(
        "--source",
        choices=["auto", "nflreadpy", "espn"],
        default="auto",
        help="Schedule source. auto tries nflreadpy first, then ESPN.",
    )
    parser.add_argument("--schedule-csv", type=Path, help="Optional schedule CSV if nflreadpy does not have the season yet.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def normalize_team(team: str) -> str:
    return ESPN_TO_NFLVERSE_TEAM.get(team, team)


def load_espn_schedule(season: int) -> pd.DataFrame:
    rows = []
    for week in range(1, 19):
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            f"?seasontype=2&week={week}&dates={season}"
        )
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        for event in data.get("events", []):
            competition = event["competitions"][0]
            competitors = competition["competitors"]
            home = next(team for team in competitors if team["homeAway"] == "home")
            away = next(team for team in competitors if team["homeAway"] == "away")
            home_team = normalize_team(home["team"]["abbreviation"])
            away_team = normalize_team(away["team"]["abbreviation"])
            rows.append(
                {
                    "game_id": f"{season}_{week:02d}_{away_team}_{home_team}",
                    "season": season,
                    "week": week,
                    "game_type": "REG",
                    "gameday": event.get("date"),
                    "away_team": away_team,
                    "home_team": home_team,
                    "roof": None,
                    "surface": None,
                    "temp": None,
                    "wind": None,
                    "div_game": None,
                    "spread_line": None,
                    "total_line": None,
                }
            )
    return pd.DataFrame(rows)


def load_nflreadpy_schedule(season: int) -> pd.DataFrame:
    schedule = nfl.load_schedules([season]).to_pandas()
    schedule = schedule[schedule["season"].eq(season)].copy()
    if "game_type" in schedule.columns:
        schedule = schedule[schedule["game_type"].eq("REG")].copy()
    return schedule


def load_schedule(season: int, schedule_csv: Path | None, source: str) -> pd.DataFrame:
    if schedule_csv:
        schedule = pd.read_csv(schedule_csv)
    else:
        if source == "nflreadpy":
            schedule = load_nflreadpy_schedule(season)
        elif source == "espn":
            schedule = load_espn_schedule(season)
        else:
            schedule = load_nflreadpy_schedule(season)
            if schedule.empty:
                print(f"nflreadpy has no {season} rows yet; falling back to ESPN schedule API.")
                schedule = load_espn_schedule(season)

    schedule = schedule[schedule["season"].eq(season)].copy()
    if "game_type" in schedule.columns:
        schedule = schedule[schedule["game_type"].eq("REG")].copy()
    return schedule.sort_values(["week", "game_id"]).reset_index(drop=True)


def predict_schedule(schedule: pd.DataFrame, history: pd.DataFrame, artifact: dict[str, object]) -> pd.DataFrame:
    if schedule.empty:
        raise ValueError(
            "No schedule rows found. Try --source espn, or pass --schedule-csv with columns season, week, "
            "game_id, home_team, away_team, and optional gameday, roof, surface, temp, wind, div_game."
        )

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    rows = []
    feature_rows = []
    for _, game in schedule.iterrows():
        feature_row = make_prediction_row(
            matchups=history,
            feature_cols=feature_cols,
            home_team=str(game["home_team"]),
            away_team=str(game["away_team"]),
            season=int(game["season"]),
            week=int(game["week"]),
            game_type=str(game.get("game_type", "REG")),
            roof=game.get("roof"),
            surface=game.get("surface"),
            temp=game.get("temp"),
            wind=game.get("wind"),
            div_game=game.get("div_game"),
            spread_line=game.get("spread_line"),
            total_line=game.get("total_line"),
        )
        feature_row = fill_latest_team_state(
            feature_row,
            history,
            feature_cols,
            home_team=str(game["home_team"]),
            away_team=str(game["away_team"]),
            season=int(game["season"]),
            week=int(game["week"]),
        )
        feature_rows.append(feature_row)
        rows.append(
            {
                "game_id": game.get("game_id"),
                "season": game.get("season"),
                "week": game.get("week"),
                "gameday": game.get("gameday"),
                "away_team": game.get("away_team"),
                "home_team": game.get("home_team"),
            }
        )

    features = pd.concat(feature_rows, ignore_index=True)
    home_probs = model.predict_proba(features)[:, 1]
    predictions = pd.DataFrame(rows)
    predictions["home_win_probability"] = home_probs
    predictions["away_win_probability"] = 1 - predictions["home_win_probability"]
    predictions["predicted_winner"] = predictions["home_team"].where(
        predictions["home_win_probability"].ge(0.5),
        predictions["away_team"],
    )
    return predictions.sort_values(["week", "gameday", "game_id"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.model)
    history = pd.read_csv(args.history)
    schedule = load_schedule(args.season, args.schedule_csv, args.source)
    predictions = predict_schedule(schedule, history, artifact)

    output = args.output or Path("predictions") / f"nfl_{args.season}_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Wrote {len(predictions):,} predictions to {output}")
    print(predictions.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
