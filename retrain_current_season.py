"""Evaluate saved configurations on 2026, then refit on all completed games."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from model_utils import get_feature_columns
from prediction_features import make_prediction_row, fill_latest_team_state
from current_features import apply_current


def main():
    root = Path(__file__).resolve().parent
    history = pd.read_csv(root / 'data/nfl_matchups_1999_2025_advanced.csv')
    games = pd.read_csv(root / 'data/nfl_2026_completed.csv')
    games = games.loc[games.home_score.notna() & games.away_score.notna()].copy()
    if games.empty or games.game_id.duplicated().any():
        raise ValueError('Need unique completed current-season games')
    feature_cols = get_feature_columns(history)
    context = pd.concat([history, games], ignore_index=True)
    snapshots_path = root / 'data/current_features_2026.csv'
    snapshots = pd.read_csv(snapshots_path).set_index('game_id') if snapshots_path.exists() else None
    rows = []
    for _, game in games.iterrows():
        args = {key: game.get(key) for key in ['home_team', 'away_team', 'season', 'week', 'game_type', 'roof', 'surface', 'temp', 'wind', 'div_game', 'spread_line', 'total_line']}
        row = make_prediction_row(context, feature_cols, **args)
        row = fill_latest_team_state(row, context, feature_cols, str(game.home_team), str(game.away_team), int(game.season), int(game.week))
        row = apply_current(row, game, snapshots)
        for key in ['game_id', 'gameday', 'home_score', 'away_score']:
            row[key] = game[key]
        row['home_win'] = int(game.home_score > game.away_score)
        if game.home_score != game.away_score:
            rows.append(row)
    current = pd.concat(rows, ignore_index=True)
    assert current.loc[current.week.eq(1), 'home_team_games_played'].eq(0).all()
    assert current.loc[current.week.eq(2), 'home_team_games_played'].eq(1).all()
    full = pd.concat([history, current], ignore_index=True)[history.columns]
    dataset = root / 'data/nfl_matchups_1999_2026_advanced.csv'
    full.to_csv(dataset, index=False)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = root / 'models/archive' / stamp
    archive.mkdir(parents=True)
    reports = {}
    for path in sorted((root / 'models').glob('*.joblib')):
        if 'smoke' in path.name:
            continue
        print(f'Retraining {path.name}', flush=True)
        artifact = joblib.load(path)
        cols = artifact['feature_cols']
        if not set(cols).issubset(full.columns):
            raise ValueError(f'Unknown training features for {path.name}')
        model = clone(artifact['model'])
        model.fit(history[cols], history.home_win)
        p = model.predict_proba(current[cols])[:, 1]
        metrics = {
            'holdout_season': 2026, 'evaluation': '2026 before final refit; evaluation estimator trained through 2025',
            'train_rows': len(history), 'test_rows': len(current),
            'accuracy': float(accuracy_score(current.home_win, p >= .5)),
            'log_loss': float(log_loss(current.home_win, p, labels=[0, 1])),
            'roc_auc': float(roc_auc_score(current.home_win, p)) if current.home_win.nunique() == 2 else None,
            'final_training_rows': len(full), 'final_training_2026_rows': len(current),
            'final_training_through': str(games.gameday.max()),
            'configuration': 'Previously tuned hyperparameters retained; no new search',
            'advanced_features': 'Available 2026 pregame features; historical fallback for missing values' if snapshots is not None else 'Historical snapshots carried forward for 2026',
        }
        model.fit(full[cols], full.home_win)
        shutil.copy2(path, archive / path.name)
        if path.with_suffix('.metrics.json').exists():
            shutil.copy2(path.with_suffix('.metrics.json'), archive / path.with_suffix('.metrics.json').name)
        artifact.update(model=model, metrics=metrics, trained_at=stamp, model_version=stamp,
                        training_input=str(dataset), training_through_season=2026)
        joblib.dump(artifact, path, compress=3)
        path.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        reports[path.name] = metrics
        print(f"Saved: {len(full)} training rows, pre-refit log loss {metrics['log_loss']:.4f}", flush=True)
    (root / 'models/retraining_2026.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
