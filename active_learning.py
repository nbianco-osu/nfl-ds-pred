"""Uncertainty sampling and a prospective, shadow probability-calibration learner."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss


def grade_predictions(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    home = pd.to_numeric(rows.get('home_score', pd.Series(index=rows.index, dtype=float)), errors='coerce')
    away = pd.to_numeric(rows.get('away_score', pd.Series(index=rows.index, dtype=float)), errors='coerce')
    final = rows.get('status', pd.Series('Scheduled', index=rows.index)).eq('Final') & home.notna() & away.notna()
    tied = final & home.eq(away)
    winner = rows.home_team.where(home.gt(away), rows.away_team)
    rows['actual_winner'] = winner.where(final & ~tied)
    rows['prediction_correct'] = pd.Series(pd.NA, index=rows.index, dtype='boolean')
    decisive = final & ~tied
    rows.loc[decisive, 'prediction_correct'] = rows.loc[decisive, 'predicted_winner'].eq(winner.loc[decisive])
    rows['prediction_result'] = 'Pending'
    rows.loc[tied, 'prediction_result'] = 'Tie (no winner)'
    rows.loc[decisive, 'prediction_result'] = rows.loc[decisive, 'prediction_correct'].map({True: 'Yes', False: 'No'})
    return rows


def inputs(probabilities) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def query_games(pool: pd.DataFrame, budget: int = 8) -> list[str]:
    """Choose close calls plus random exploration, without using outcome labels."""
    if pool.empty:
        return []
    pool = pool.sort_values('game_id').copy()
    pool['uncertainty'] = 1 - 2 * (pool.home_win_probability - 0.5).abs()
    count = min(budget, len(pool))
    uncertain = pool.sort_values(['uncertainty', 'game_id'], ascending=[False, True]).head(max(1, count - 2))
    remaining = pool.loc[~pool.game_id.isin(uncertain.game_id)]
    explore = remaining.sample(n=count - len(uncertain), random_state=42)
    return uncertain.game_id.tolist() + explore.game_id.tolist()


def update_learning(rows: pd.DataFrame, root: Path) -> dict:
    rows = grade_predictions(rows)
    folder = root / 'learning'
    folder.mkdir(exist_ok=True)
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'seed_ids': rows.loc[rows.prediction_correct.notna(), 'game_id'].tolist(),
        'queries': {}, 'forecasts': {},
    }
    final = rows.loc[rows.prediction_correct.notna()].copy()
    eligible = set(state['seed_ids']) | {g for batch in state['queries'].values() for g in batch}
    train = final.loc[final.game_id.isin(eligible)].sort_values('game_id')
    # Forecasts were persisted BEFORE outcomes; score all available cases, not just queries.
    evaluation = []
    for game in final.itertuples():
        saved = state['forecasts'].get(game.game_id)
        if saved:
            evaluation.append({'game_id': game.game_id, 'label': int(game.actual_winner == game.home_team), **saved})
    metrics = {'evaluated_games': len(evaluation), 'minimum_evaluation_games': 64}
    if evaluation:
        frame = pd.DataFrame(evaluation)
        for name in ['baseline', 'candidate']:
            metrics[f'{name}_log_loss'] = float(log_loss(frame.label, frame[name], labels=[0, 1]))
            metrics[f'{name}_brier'] = float(brier_score_loss(frame.label, frame[name]))
    ready = len(evaluation) >= 64 and metrics['candidate_log_loss'] < metrics['baseline_log_loss'] and metrics['candidate_brier'] < metrics['baseline_brier']
    today = pd.Timestamp(datetime.now(timezone.utc).date(), tz='UTC')
    dates = pd.to_datetime(rows.gameday, errors='coerce', utc=True, format='mixed')
    pending = rows.loc[rows.prediction_result.eq('Pending') & dates.ge(today + pd.Timedelta(days=1))].copy()
    # A query batch is fixed before its results arrive and never selected using errors.
    if not pending.empty:
        season = int(pending.season.min())
        week = int(pending.loc[pending.season.eq(season), 'week'].min())
        key = f'{season}_{week:02d}'
        if key not in state['queries']:
            state['queries'][key] = query_games(pending.loc[pending.season.eq(season) & pending.week.eq(week)])
    if len(train) >= 2 and train.actual_winner.eq(train.home_team).nunique() == 2:
        candidate = LogisticRegression(C=0.1, random_state=42)
        candidate.fit(inputs(train.home_win_probability), train.actual_winner.eq(train.home_team).astype(int))
        artifact = {'model': candidate, 'input': 'logit of archived baseline home-win probability',
                    'training_game_ids': train.game_id.tolist(), 'status': 'shadow_only', 'metrics': metrics}
        joblib.dump(artifact, folder / 'candidate.joblib')
        if not pending.empty:
            probabilities = candidate.predict_proba(inputs(pending.home_win_probability))[:, 1]
            for (_, game), probability in zip(pending.iterrows(), probabilities):
                state['forecasts'][game.game_id] = {'baseline': float(game.home_win_probability), 'candidate': float(probability)}
    summary = {'status': 'Ready for review' if ready else 'Learning in shadow mode',
               'training_games': len(train), 'selected_game_ids': state['queries'], **metrics,
               'note': 'Candidate recalibrates baseline probabilities; no production replacement or proven improvement yet. Ties excluded. Initial completed games are training-only.'}
    state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    pd.DataFrame(evaluation, columns=['game_id', 'label', 'baseline', 'candidate']).to_csv(folder / 'evaluation.csv', index=False)
    return summary


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    frames = sorted((root / 'predictions').glob('nfl_*_predictions_advanced_no_market_deep.csv'))
    print(json.dumps(update_learning(pd.concat([pd.read_csv(p) for p in frames], ignore_index=True), root), indent=2))
