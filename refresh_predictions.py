"""Refresh score-based form and public forecasts without retraining the saved model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import nflreadpy as nfl
import pandas as pd

from predict_schedule import predict_schedule
from active_learning import grade_predictions, update_learning
from current_features import build_current
from simulate_season import export_simulations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=int, default=datetime.now().year)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    build_current(args.season)
    snapshots = pd.read_csv(root / f'data/current_features_{args.season}.csv').set_index('game_id')
    model = root / 'models/home_win_advanced_automl_no_market_deep.joblib'
    output = root / f'predictions/nfl_{args.season}_predictions_advanced_no_market_deep.csv'
    history = pd.read_csv(root / 'data/nfl_matchups_1999_2025_advanced.csv')
    schedule = pd.DataFrame(nfl.load_schedules([args.season]).to_dicts())
    schedule = schedule.loc[schedule.game_type.eq('REG')].copy()
    if schedule.empty or schedule.game_id.duplicated().any():
        raise ValueError('Schedule missing or contains duplicate games; existing forecasts preserved.')
    finished = schedule.home_score.notna() & schedule.away_score.notna()
    completed = schedule.loc[finished].copy()
    # Only completed scores enter team form. Advanced inputs remain the saved historical snapshot.
    history = pd.concat([history.loc[history.season.lt(args.season)], completed], ignore_index=True)
    artifact = joblib.load(model)
    upcoming = schedule.loc[~finished]
    predictions = predict_schedule(upcoming, history, artifact, snapshots=snapshots) if len(upcoming) else pd.DataFrame()
    generation = artifact.get('model_version')
    predictions['model_version'] = generation or 'legacy'
    old = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=['game_id'])
    preserved = old.loc[old.game_id.isin(completed.game_id)].copy()
    if 'model_version' not in preserved:
        preserved['model_version'] = 'legacy'
    predictions = pd.concat([predictions, preserved], ignore_index=True)
    if len(predictions) != len(schedule):
        raise ValueError('Completed games lack saved pregame forecasts; refusing retrospective picks.')
    predictions = predictions.drop(columns=['home_score', 'away_score', 'status'], errors='ignore')
    predictions = predictions.merge(schedule[['game_id', 'home_score', 'away_score']], on='game_id', validate='one_to_one')
    predictions['status'] = predictions.home_score.notna().map({True: 'Final', False: 'Scheduled'})
    predictions = predictions.sort_values(['week', 'game_id'])
    predictions = grade_predictions(predictions)
    public = predictions.copy()
    teams = pd.read_csv(root / 'data/team_metadata.csv').set_index('team_abbr')
    for side in ['home', 'away']:
        public[f'{side}_team_name'] = public[f'{side}_team'].map(teams.team_name)
        public[f'{side}_logo'] = public[f'{side}_team'].map(teams.team_logo_espn)
    public['predicted_winner_name'] = public.predicted_winner.map(teams.team_name)
    public['winner_logo'] = public.predicted_winner.map(teams.team_logo_espn)
    public['confidence'] = public[['home_win_probability', 'away_win_probability']].max(axis=1)
    public['updated_at'] = datetime.now(timezone.utc).isoformat()
    public['feature_note'] = (
        f"Model retrained through {artifact['metrics']['final_training_through']}; "
        'Available 2026 EPA, QB and roster inputs refreshed; unavailable inputs use historical fallback. Simulation noise is an assumption, not measured uncertainty.'
        if generation else 'Scores and team form updated; EPA, QB, injury and roster inputs retain historical values. Model not retrained.'
    )
    if public[['home_team_name', 'away_team_name']].isna().any().any():
        raise ValueError('Unknown team in schedule')
    if not public.home_win_probability.between(0, 1).all():
        raise ValueError('Invalid probabilities')
    predictions.to_csv(output, index=False)
    completed.to_csv(root / f'data/nfl_{args.season}_completed.csv', index=False)
    public.to_json(root / 'public_site/data/predictions.json', orient='records', indent=2)
    export_simulations(predictions, root)
    learning_rows = predictions.loc[predictions.model_version.eq(generation)] if generation else predictions
    summary = update_learning(learning_rows, root, generation=generation)
    print(f"Active learner: {summary['status']}; {summary['training_games']} training games")
    print(completed[['game_id', 'home_score', 'away_score']].to_string(index=False))
    print(f'Refreshed {len(upcoming)} upcoming forecasts; preserved {len(preserved)} pregame picks.')


if __name__ == '__main__':
    main()
