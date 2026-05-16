from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from shap_utils import global_importance, local_top_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP explanation tables for an NFL model.")
    parser.add_argument("--model", type=Path, default=Path("models/home_win_advanced_automl_no_market.joblib"))
    parser.add_argument("--input", type=Path, default=Path("data/nfl_matchups_1999_2025_advanced.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("explanations"))
    parser.add_argument("--holdout-season", type=int)
    parser.add_argument("--max-rows", type=int, default=750)
    parser.add_argument("--game-id", help="Also write a local top-feature explanation for one historical game.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.model)
    data = pd.read_csv(args.input)
    season = args.holdout_season or int(data["season"].max())
    rows = data[data["season"].eq(season)].copy()
    if rows.empty:
        raise ValueError(f"No rows found for holdout season {season}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_path = args.output_dir / f"global_shap_{season}.csv"
    global_importance(artifact, rows, max_rows=args.max_rows).to_csv(global_path, index=False)
    print(f"Wrote global SHAP importance: {global_path}")

    if args.game_id:
        candidates = data[data["game_id"].eq(args.game_id)]
        if candidates.empty:
            raise ValueError(f"No row found for game_id={args.game_id}")
        local_path = args.output_dir / f"local_shap_{args.game_id}.csv"
        local_top_features(artifact, candidates.iloc[[0]], top_n=25).to_csv(local_path, index=False)
        print(f"Wrote local SHAP explanation: {local_path}")


if __name__ == "__main__":
    main()
