"""Dependency-free publication gate for exported forecast data."""
import json
import math
from datetime import datetime
from pathlib import Path


def validate(root):
    folder=Path(root)/'public_site/data'
    for path in folder.glob('*.json'):
        json.loads(path.read_text(encoding='utf-8'),parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Invalid JSON number: {value}')))
    rows=json.loads((folder/'predictions.json').read_text())
    if not rows or len({r['game_id'] for r in rows})!=len(rows):
        raise ValueError('Missing or duplicate forecasts')
    for r in rows:
        p,q=r['home_win_probability'],r['away_win_probability']
        if not all(isinstance(v,(float,int)) and math.isfinite(v) and 0<=v<=1 for v in [p,q]) or abs(p+q-1)>1e-8:
            raise ValueError('Invalid win probability')
        if r['predicted_winner'] not in [r['home_team'],r['away_team']]:
            raise ValueError('Invalid winner team')
        if r['status']=='Final':
            h,a=r['home_score'],r['away_score']
            expected='Tie (no winner)' if h==a else 'Yes' if r['predicted_winner']==(r['home_team'] if h>a else r['away_team']) else 'No'
            if r['prediction_result']!=expected:
                raise ValueError('Incorrect outcome grading')
        for field in ['predicted_home_score','predicted_away_score']:
            if r.get(field) is not None and (not math.isfinite(r[field]) or r[field]<0):
                raise ValueError('Invalid expected score')
    audit_path=folder/'game_audit.json'
    if audit_path.exists():
        audits=json.loads(audit_path.read_text())
        for r in rows:
            audit=audits.get(r['game_id'])
            if audit and (audit['model_version']!=r['model_version'] or abs(audit['home_probability']-r['home_win_probability'])>1e-8):
                raise ValueError('Explanation is not for the displayed forecast')
    dist_path=folder/'score_distributions.json'
    if dist_path.exists():
        for d in json.loads(dist_path.read_text()).values():
            if abs(d['home_win']+d['away_win']+d['tie']-1)>1e-9:
                raise ValueError('Incoherent score outcome distribution')
            for field in ['spread','total']:
                if d[field] and abs(sum(d[field].values())-1)>1e-9:
                    raise ValueError('Incoherent line probabilities')
    history_path=folder/'forecast_history.json'
    if history_path.exists():
        for game_id, records in json.loads(history_path.read_text())['games'].items():
            if game_id not in {r['game_id'] for r in rows}:
                raise ValueError('Unknown game in forecast history')
            if len({r['id'] for r in records}) != len(records):
                raise ValueError('Duplicate history observations')
            for r in records:
                if not 0 <= r['probability'] <= 1:
                    raise ValueError('Invalid history probability')
                if datetime.fromisoformat(r['forecast_at'].replace('Z','+00:00')).tzinfo is None:
                    raise ValueError('History timestamp lacks timezone')
    return {'games':len(rows),'completed':sum(r['status']=='Final' for r in rows)}


if __name__=='__main__':
    print(validate(Path(__file__).resolve().parent))
