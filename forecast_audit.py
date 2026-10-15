"""Freeze exact future-game inputs and explanations; never backfill final games."""
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

from audit_io import write_json
from score_markets import pregame


def archive(root, schedule, metadata, features, artifact, probabilities, old):
    now = datetime.now(timezone.utc)
    old = old.set_index('game_id') if len(old) else old
    explanation = {}
    path = root/'public_site/data/game_audit.json'
    if path.exists():
        explanation = json.loads(path.read_text())
    coverage = json.loads((root/f'data/current_features_{int(schedule.season.max())}.json').read_text())
    current = pd.read_csv(root/f'data/current_features_{int(schedule.season.max())}.csv').set_index('game_id')
    version = artifact['model_version']
    with (root/'models/home_win_advanced_automl_no_market_deep.joblib').open('rb') as stream:
        model_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
    pipeline = artifact['model']
    # Explain only inputs used for current pregame forecasts, never reconstruct old finals.
    import shap
    matrix = pipeline.named_steps['preprocess'].transform(features)
    explainer = shap.TreeExplainer(pipeline.named_steps['model'])
    values = np.asarray(explainer.shap_values(matrix))
    if values.ndim == 3:
        values = values[:, :, 1]
    names = pipeline.named_steps['preprocess'].get_feature_names_out()
    base = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
    times = {}
    for i, row in metadata.reset_index(drop=True).iterrows():
        game = schedule.loc[schedule.game_id.eq(row.game_id)].iloc[0]
        if not pregame(game, now):
            raise ValueError('Cannot create provenance after kickoff')
        raw = {key: None if pd.isna(value) else value.item() if hasattr(value,'item') else value
               for key,value in features.iloc[i].items()}
        payload = {'game_id': row.game_id, 'model_version': version, 'model_sha256':model_sha256, 'features': raw,
                   'probability': float(probabilities[i]), 'source_hashes': {k:v['sha256'] for k,v in coverage.get('receipts', {}).items()}}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        record_path = root/'audit/forecasts'/row.game_id/(digest+'.json')
        if record_path.exists():
            record = json.loads(record_path.read_text())
        else:
            record = {**payload, 'receipts':coverage.get('receipts', {}), 'forecast_at': now.isoformat(), 'input_sha256': digest}
            write_json(record_path, record)
        times[row.game_id] = record['forecast_at']
        source_status = {}
        fresh = current.loc[row.game_id] if row.game_id in current.index else pd.Series(dtype=float)
        for label, prefix in [('EPA', 'pbp'), ('QB performance', 'qb'), ('Injuries', 'injury'), ('Roster', 'roster'), ('Rest', 'rest')]:
            cols = [c for c in features if f'team_{prefix}' in c and not c.startswith('diff_') and 'prev_season' not in c]
            count = sum(c in fresh and pd.notna(fresh[c]) for c in cols)
            source_status[label] = {'current_fields': count, 'model_fields': len(cols),
                                    'status': 'Current snapshot' if cols and count == len(cols) else 'Partial / historical fallback' if count else 'Historical fallback / unavailable'}
        order = np.argsort(-np.abs(values[i]))[:10]
        top = [{'feature': str(names[j]), 'impact': float(values[i,j]), 'value': float(matrix[i,j])} for j in order]
        explained = base + float(values[i].sum())
        if abs(explained-float(probabilities[i])) > 1e-5:
            raise ValueError('SHAP reconstruction differs from archived probability')
        explanation[row.game_id] = {'forecast_at': record['forecast_at'], 'input_sha256': digest,
            'model_version': version, 'home_probability': float(probabilities[i]), 'base_probability': base,
            'top_features': top, 'other_impact': float(values[i].sum()-sum(t['impact'] for t in top)),
            'sources': source_status, 'source_receipts': coverage.get('receipts', {}),
            'note': 'Tree SHAP in home-win probability units; associations, not causal effects. Transformed numeric values are standardized. Original source publication times are unknown.'}
    write_json(path, explanation)
    return times
