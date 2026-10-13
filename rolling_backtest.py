"""Weekly expanding-window research baselines; never changes production artifacts."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['home_for', 'home_against', 'away_for', 'away_against']


def prepare(games):
    games = games.copy()
    if games.game_id.duplicated().any():
        raise ValueError('Duplicate game IDs')
    if 'game_type' in games:
        games = games.loc[games.game_type.eq('REG')].copy()
    games['gameday'] = pd.to_datetime(games.gameday)
    games = games.dropna(subset=['home_score', 'away_score']).sort_values(['season', 'week', 'gameday', 'game_id'])
    history = defaultdict(list)
    parts = []
    # Freeze every feature at the first kickoff date of its week, including Monday games.
    for (_, _), week in games.groupby(['season', 'week'], sort=True):
        cutoff = week.gameday.min()
        week = week.copy()
        for side in ['home', 'away']:
            values = []
            for team in week[f'{side}_team']:
                previous = [(pf, pa) for date, pf, pa in history[team] if date < cutoff][-8:]
                values.append(np.mean(previous, axis=0) if previous else [21., 21.])
            week[[f'{side}_for', f'{side}_against']] = values
        week['feature_cutoff'] = cutoff
        parts.append(week)
        for row in week.itertuples():
            history[row.home_team].append((row.gameday, row.home_score, row.away_score))
            history[row.away_team].append((row.gameday, row.away_score, row.home_score))
    if not parts:
        raise ValueError('No completed source games')
    return pd.concat(parts, ignore_index=True)


def run(games, start, end):
    data = prepare(games)
    outputs, folds = [], []
    target = data.loc[data.season.between(start, end)]
    for (season, week), test in target.groupby(['season', 'week'], sort=True):
        cutoff = test.feature_cutoff.min()
        train = data.loc[data.gameday.lt(cutoff)]
        decisive = train.loc[train.home_score.ne(train.away_score)]
        if len(decisive) < 256:
            raise ValueError('At least 256 earlier decisive games are required per fold')
        y = decisive.home_score.gt(decisive.away_score).astype(int)
        winner = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000))
        winner.fit(decisive[FEATURES], y)
        score = make_pipeline(StandardScaler(), Ridge(alpha=10))
        score.fit(train[FEATURES], train[['home_score', 'away_score']])
        predicted_scores = np.maximum(0, score.predict(test[FEATURES]))
        fold = {'season': int(season), 'week': int(week), 'cutoff': cutoff.date().isoformat(),
                'training_through': train.gameday.max().date().isoformat(), 'training_rows': len(train), 'test_rows': len(test)}
        folds.append(fold)
        out = test[['game_id', 'season', 'week', 'home_score', 'away_score']].copy()
        out['form_probability'] = winner.predict_proba(test[FEATURES])[:, 1]
        out['home_baseline_probability'] = float(y.mean())
        out['predicted_home_score'] = predicted_scores[:, 0]
        out['predicted_away_score'] = predicted_scores[:, 1]
        out['cutoff'] = fold['cutoff']
        outputs.append(out)
    if not outputs:
        raise ValueError('No games in requested seasons')
    return pd.concat(outputs, ignore_index=True), folds


def metrics(rows):
    decisive = rows.loc[rows.home_score.ne(rows.away_score)]
    y = decisive.home_score.gt(decisive.away_score).to_numpy()
    report = {'games': len(rows), 'winner_n': len(decisive), 'ties': len(rows)-len(decisive)}
    for model in ['form', 'home_baseline']:
        p = np.clip(decisive[f'{model}_probability'].to_numpy(), 1e-15, 1-1e-15)
        report[model] = {'accuracy': float(((p >= .5) == y).mean()),
                         'log_loss': float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()),
                         'brier': float(((p-y)**2).mean())}
    h = rows.predicted_home_score-rows.home_score
    a = rows.predicted_away_score-rows.away_score
    report['score'] = {'team_mae': float((h.abs().sum()+a.abs().sum())/(2*len(rows))),
                       'margin_mae': float((h-a).abs().mean()), 'total_mae': float((h+a).abs().mean())}
    return report


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-season', type=int, default=2023)
    parser.add_argument('--end-season', type=int, default=2025)
    parser.add_argument('--data', type=Path, default=root/'data/nfl_matchups_1999_2026_advanced.csv')
    args = parser.parse_args()
    predictions, folds = run(pd.read_csv(args.data), args.start_season, args.end_season)
    output = root/'backtests'
    output.mkdir(exist_ok=True)
    predictions.to_csv(output/'weekly_predictions.csv', index=False)
    report = {'schema_version': 1, 'source': args.data.name,
              'source_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'start_season': args.start_season, 'end_season': args.end_season,
              'method': 'Expanding training window; refit before each week. Fixed logistic C=1 and Ridge alpha=10; no tuning on evaluation seasons. Eight previous team games, using scores only. Home baseline is the training home-win rate. No production model is loaded or changed.',
              'limitations': 'Revised historical source, not original-release snapshots. Regular-season rows only; the source dataset excludes historical ties. Results cover available rows, not a complete schedule audit. These simple baselines do not evaluate the advanced production Random Forest.',
              'features': FEATURES, 'summary': metrics(predictions), 'folds': folds,
              'seasons': [{'season': int(season), **metrics(part)} for season, part in predictions.groupby('season')]}
    (root/'public_site/data/backtest.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
