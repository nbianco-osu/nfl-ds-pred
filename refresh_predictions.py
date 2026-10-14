"""Refresh score-based form and public forecasts without retraining the saved model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import nflreadpy as nfl
import pandas as pd

from predict_schedule import build_schedule_features
from active_learning import grade_predictions, update_learning
from current_features import build_current
from simulate_season import export_simulations


def run(root, season) -> None:
    from audit_io import write_csv, write_json
    from score_markets import pregame
    build_current(season, root=root)
    snapshots = pd.read_csv(root / f'data/current_features_{season}.csv').set_index('game_id')
    model = root / 'models/home_win_advanced_automl_no_market_deep.joblib'
    output = root / f'predictions/nfl_{season}_predictions_advanced_no_market_deep.csv'
    history = pd.read_csv(root / 'data/nfl_matchups_1999_2025_advanced.csv')
    schedule = pd.read_csv(root/f'data/schedule_{season}.csv')
    schedule = schedule.loc[schedule.game_type.eq('REG')].copy()
    if schedule.empty or schedule.game_id.duplicated().any():
        raise ValueError('Schedule missing or contains duplicate games; existing forecasts preserved.')
    finished = schedule.home_score.notna() & schedule.away_score.notna()
    completed = schedule.loc[finished].copy()
    # Only completed scores enter team form. Advanced inputs remain the saved historical snapshot.
    history = pd.concat([history.loc[history.season.lt(season)], completed], ignore_index=True)
    artifact = joblib.load(model)
    now = datetime.now(timezone.utc)
    upcoming = schedule.loc[schedule.apply(lambda row: pregame(row, now),axis=1)]
    old = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=['game_id'])
    predictions = pd.DataFrame()
    if len(upcoming):
        predictions, features = build_schedule_features(upcoming, history, artifact, snapshots=snapshots)
        probabilities = artifact['model'].predict_proba(features)[:,1]
        predictions['home_win_probability'] = probabilities
        predictions['away_win_probability'] = 1-probabilities
        predictions['predicted_winner'] = predictions.home_team.where(predictions.home_win_probability.ge(.5),predictions.away_team)
        from forecast_audit import archive
        stamps=archive(root,schedule,predictions,features,artifact,probabilities,old)
        predictions['winner_forecast_at']=predictions.game_id.map(stamps)
    generation = artifact.get('model_version')
    predictions['model_version'] = generation or 'legacy'
    preserved = old.loc[~old.game_id.isin(upcoming.game_id)].copy()
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
    from score_markets import refresh as refresh_scores
    predictions = refresh_scores(predictions, schedule, history, snapshots, root)
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
    write_csv(output,predictions)
    write_csv(root / f'data/nfl_{season}_completed.csv',completed)
    import json
    write_json(root / 'public_site/data/predictions.json',json.loads(public.to_json(orient='records')))
    from performance import export as export_performance
    export_performance(root)
    export_simulations(predictions, root)
    from season_charts import export as export_charts
    export_charts(root)
    learning_rows = predictions.loc[predictions.model_version.eq(generation)] if generation else predictions
    summary = update_learning(learning_rows, root, generation=generation)
    from challenger_review import report
    report(root)
    from matchup_context import context
    context(root,season)
    from score_distribution import export as export_distributions
    export_distributions(root,schedule)
    from dashboard_insights import export as export_insights
    export_insights(root)
    from pregame_inputs import export as export_pregame_inputs
    export_pregame_inputs(root,season)
    from warehouse_queue import stage as stage_warehouse
    stage_warehouse(root,season)
    print(f"Active learner: {summary['status']}; {summary['training_games']} training games")
    print(completed[['game_id', 'home_score', 'away_score']].to_string(index=False))
    print(f'Refreshed {len(upcoming)} upcoming forecasts; preserved {len(preserved)} pregame picks.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    now=datetime.now()
    parser.add_argument('--season',type=int,default=now.year-(now.month<=2))
    args=parser.parse_args()
    from staged_refresh import refresh
    refresh(Path(__file__).resolve().parent,args.season)


if __name__ == '__main__':
    main()
