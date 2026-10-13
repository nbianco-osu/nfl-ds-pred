"""Export observed archived-forecast results, independently of research backtests."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_io import write_json


def interval(successes, n):
    if not n:
        return [None,None]
    p,z=successes/n,1.96
    center=(p+z*z/(2*n))/(1+z*z/n)
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-radius),float(center+radius)]


def summarize(rows):
    final = rows.loc[rows.status.eq('Final') & rows.home_score.notna() & rows.away_score.notna()]
    decisive = final.loc[final.home_score.ne(final.away_score)]
    valid = decisive.loc[decisive.home_win_probability.notna()]
    p = valid.home_win_probability.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid archived probabilities')
    y = valid.home_score.gt(valid.away_score).to_numpy(dtype=int)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    correct = valid.predicted_winner.eq(np.where(y, valid.home_team, valid.away_team))
    bins = []
    for i in range(10):
        mask = (p >= i / 10) & ((p < (i + 1) / 10) if i < 9 else (p <= 1))
        bins.append({'lower': i / 10, 'upper': (i + 1) / 10, 'n': int(mask.sum()),
                     'predicted': float(p[mask].mean()) if mask.any() else None,
                     'observed': float(y[mask].mean()) if mask.any() else None,
                     'observed_ci95':interval(int(y[mask].sum()),int(mask.sum()))})
    scores = final.dropna(subset=['predicted_home_score', 'predicted_away_score', 'score_forecast_at'])
    errors = scores.predicted_home_score - scores.home_score
    away_errors = scores.predicted_away_score - scores.away_score
    result = {
        'completed': len(final), 'ties': int(final.home_score.eq(final.away_score).sum()),
        'winner_n': len(valid), 'correct': int(correct.sum()),
        'accuracy': float(correct.mean()) if len(valid) else None,
        'accuracy_ci95':interval(int(correct.sum()),len(valid)),
        'timestamped_winner_n':int(valid.get('winner_forecast_at',pd.Series(dtype=str)).notna().sum()),
        'log_loss': float(-(y * np.log(clipped) + (1-y) * np.log(1-clipped)).mean()) if len(y) else None,
        'brier': float(((p-y)**2).mean()) if len(y) else None,
        'calibration': bins, 'score_n': len(scores),
        'team_score_mae': float((errors.abs().sum() + away_errors.abs().sum()) / (2*len(scores))) if len(scores) else None,
        'margin_mae': float((errors-away_errors).abs().mean()) if len(scores) else None,
        'total_mae': float((errors+away_errors).abs().mean()) if len(scores) else None,
    }
    for kind in ['spread', 'total']:
        counts = final[f'{kind}_grade'].value_counts().to_dict()
        n = counts.get('Correct', 0) + counts.get('Wrong', 0)
        result[kind] = {'correct': counts.get('Correct', 0), 'n': n,
                        'pushes': counts.get('Push', 0), 'excluded': len(final)-n-counts.get('Push', 0)}
    return result


def export(root):
    rows = pd.read_json(root / 'public_site/data/predictions.json', convert_dates=False)
    if rows.game_id.duplicated().any():
        raise ValueError('Duplicate archived games')
    versions = sorted(rows.model_version.fillna('legacy').unique())
    groups = []
    for version in ['all', *versions]:
        selected = rows if version == 'all' else rows.loc[rows.model_version.eq(version)]
        groups.append({'version': version, 'summary': summarize(selected),
                       'weeks': [{'week': int(week), **summarize(part)} for week, part in selected.groupby('week') if part.status.eq('Final').any()]})
    score_groups=[]
    families={'ensemble':rows}
    model_path=root/'public_site/data/model_score_predictions.json'
    if model_path.exists():
        for family,records in json.loads(model_path.read_text()).items():
            model=pd.DataFrame(records)
            base=rows.drop(columns=[c for c in model if c!='game_id'],errors='ignore')
            families[family]=base.merge(model,on='game_id',validate='one_to_one')
    for family,frame in families.items():
        for version in ['all',*sorted(frame.score_model_version.dropna().unique())]:
            selected=frame if version=='all' else frame.loc[frame.score_model_version.eq(version)]
            score_groups.append({'family':family,'version':version,'summary':summarize(selected),
                'weeks':[{'week':int(week),**summarize(part)} for week,part in selected.groupby('week') if part.status.eq('Final').any()]})
    report = {'schema_version': 2, 'source_updated_at': rows.updated_at.iloc[0],
              'default_version': versions[-1] if versions[-1] != 'legacy' else next((v for v in reversed(versions) if v != 'legacy'), 'legacy'),
              'note': 'Observed saved forecasts, not backtests. Older winner archives lack creation timestamps; legacy provenance is incomplete. New forecasts carry timestamps from the October 13 audit onward. All versions combines different models. Score errors require an archived score timestamp. Top summary score/line statistics use the ensemble within the selected winner cohort; the independent score comparison below uses score-model versions. Ties are excluded from winner metrics; pushes and ungraded picks are excluded from line hit rates. Wilson 95% intervals assume independent games and can understate dependence.',
              'groups': groups,'score_groups':score_groups}
    write_json(root / 'public_site/data/performance.json',report)
    return report


if __name__ == '__main__':
    export(Path(__file__).resolve().parent)
