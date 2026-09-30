"""Archive expected scores and line snapshots; grade without retrospective picks."""
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from predict_schedule import build_schedule_features

SNAPSHOT_FIELDS = ['game_id', 'predicted_home_score', 'predicted_away_score', 'score_model_version',
                   'score_forecast_at', 'market_home_spread', 'market_total', 'line_observed_at']
OUTPUT_FIELDS = SNAPSHOT_FIELDS[1:] + ['predicted_margin', 'predicted_total', 'model_spread_side', 'model_total_side',
                                      'actual_spread_side', 'actual_total_side', 'spread_grade', 'total_grade', 'line_basis']


def number(value):
    return float(value) if value is not None and pd.notna(value) and np.isfinite(float(value)) else None


def side(value, positive, negative):
    if abs(value) < 1e-9:
        return 'Push'
    return positive if value > 0 else negative


def grade(pick, actual, final):
    if pick is None:
        return 'No archived pick'
    if pick == 'Push':
        return 'No edge'
    if not final:
        return 'Pending'
    if actual is None:
        return 'No line'
    if actual == 'Push':
        return 'Push'
    return 'Correct' if pick == actual else 'Wrong'


def pregame(game, now):
    if pd.notna(game.get('home_score')) or pd.notna(game.get('away_score')):
        return False
    date = game.get('gameday')
    time = game.get('gametime')
    if pd.isna(date):
        return False
    try:
        # nflverse gametime is Eastern. Missing kickoff: stop at local midnight.
        kickoff = pd.Timestamp(f"{date} {time if pd.notna(time) else '00:00'}", tz='America/New_York').tz_convert('UTC')
        return kickoff > pd.Timestamp(now)
    except (ValueError, TypeError):
        return False


def comparisons(game, saved=None):
    saved = saved or {}
    final = pd.notna(game.get('home_score')) and pd.notna(game.get('away_score'))
    raw_spread, raw_total = number(game.get('spread_line')), number(game.get('total_line'))
    current_spread = -raw_spread if raw_spread is not None else None
    archived_spread, archived_total = number(saved.get('market_home_spread')), number(saved.get('market_total'))
    home, away = number(saved.get('predicted_home_score')), number(saved.get('predicted_away_score'))
    display_spread = archived_spread if archived_spread is not None else current_spread
    display_total = archived_total if archived_total is not None else raw_total
    margin = round(home - away, 1) if home is not None and away is not None else None
    total = round(home + away, 1) if margin is not None else None
    sp = side(margin + archived_spread, game['home_team'], game['away_team']) if margin is not None and archived_spread is not None else None
    ou = side(total - archived_total, 'Over', 'Under') if total is not None and archived_total is not None else None
    actual_sp = side(float(game['home_score']) - float(game['away_score']) + display_spread, game['home_team'], game['away_team']) if final and display_spread is not None else None
    actual_ou = side(float(game['home_score']) + float(game['away_score']) - display_total, 'Over', 'Under') if final and display_total is not None else None
    result = {key: saved.get(key) for key in SNAPSHOT_FIELDS[1:]}
    result.update(game_id=game['game_id'], market_home_spread=display_spread, market_total=display_total,
                  predicted_margin=margin, predicted_total=total, model_spread_side=sp, model_total_side=ou,
                  actual_spread_side=actual_sp, actual_total_side=actual_ou,
                  spread_grade=grade(sp, actual_sp, final), total_grade=grade(ou, actual_ou, final),
                  line_basis='Archived pregame snapshot' if archived_spread is not None and archived_total is not None
                  else 'Mixed: archived available line; source fallback is ungraded' if archived_spread is not None or archived_total is not None
                  else 'Historical source line; no pregame archive' if final else 'Source line / unavailable')
    return result


def refresh(predictions, schedule, history, snapshots, root, now=None):
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    season = int(schedule.season.max())
    path = root / f'predictions/nfl_{season}_score_predictions.csv'
    records = pd.read_csv(path).to_dict('records') if path.exists() else []
    if len(records) != len({row['game_id'] for row in records}):
        raise ValueError('Duplicate archived score forecasts')
    archive = {row['game_id']: row for row in records}
    artifact = joblib.load(root / 'models/score_models/score_ensemble.joblib')
    upcoming = schedule.loc[schedule.apply(lambda row: pregame(row, now), axis=1)]
    if not upcoming.empty:
        metadata, features = build_schedule_features(upcoming, history, artifact, snapshots)
        values = np.round(np.maximum(0, np.mean([model.predict(features) for model in artifact['models']], axis=0)), 1)
        if values.shape != (len(upcoming), 2) or not np.isfinite(values).all():
            raise ValueError('Invalid predicted scores')
        for game, scores in zip(upcoming.to_dict('records'), values):
            spread, total = number(game.get('spread_line')), number(game.get('total_line'))
            row = dict(game_id=game['game_id'], predicted_home_score=float(scores[0]), predicted_away_score=float(scores[1]),
                       score_model_version=artifact['model_version'], market_home_spread=-spread if spread is not None else None,
                       market_total=total)
            old = archive.get(game['game_id'], {})
            # Keep snapshot times stable unless values actually change.
            score_changed = any(row[k] != old.get(k) for k in ['predicted_home_score', 'predicted_away_score', 'score_model_version'])
            line_changed = any(row[k] != number(old.get(k)) for k in ['market_home_spread', 'market_total'])
            row['score_forecast_at'] = now.isoformat() if score_changed else old.get('score_forecast_at')
            row['line_observed_at'] = now.isoformat() if line_changed or not old else old.get('line_observed_at')
            archive[game['game_id']] = row
    additions = pd.DataFrame([comparisons(game, archive.get(game['game_id'])) for game in schedule.to_dict('records')])
    output = predictions.drop(columns=OUTPUT_FIELDS, errors='ignore').merge(additions, on='game_id', validate='one_to_one')
    pd.DataFrame(archive.values(), columns=SNAPSHOT_FIELDS).sort_values('game_id').to_csv(path, index=False)
    metrics_path = root / 'public_site/data/score_metrics.json'
    metrics_path.write_text(json.dumps(artifact['metrics'], indent=2), encoding='utf-8')
    return output
