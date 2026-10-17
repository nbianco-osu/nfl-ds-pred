"""Verified candidate inference; never writes production picks or claims a saved forecast."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits


def validate_games(frame, training_through, now):
    if 'game_id' not in frame or 'kickoff_at' not in frame or frame.empty:
        raise ValueError('Nonempty game_id/kickoff_at input required')
    if frame.game_id.isna().any() or frame.game_id.duplicated().any():
        raise ValueError('Missing or duplicate game IDs')
    times = [pd.Timestamp(v) for v in frame.kickoff_at]
    if any(pd.isna(t) or t.tzinfo is None for t in times):
        raise ValueError('Explicit timezone-aware kickoffs required')
    now = pd.Timestamp(now)
    if now.tzinfo is None: raise ValueError('Timezone-aware observation time required')
    # Date-only training cutoffs conservatively cover the entire stated UTC day.
    cutoff = pd.Timestamp(training_through, tz='UTC')+pd.Timedelta(days=1)
    if any(t <= now or t < cutoff for t in times):
        raise ValueError('Only future games after the training cutoff may be forecast')


def predict(root, generation, frame, now=None):
    if Path(generation).name != generation: raise ValueError('Generation name required')
    folder = Path(root)/'models/retrained'/generation
    report = json.loads((folder/'report.json').read_text())
    path = folder/'candidate.joblib'
    if report['status'] != 'completed_shadow_only' or hashlib.sha256(path.read_bytes()).hexdigest() != report['candidate_sha256']:
        raise ValueError('Candidate artifact status or integrity failure')
    observed = now or datetime.now(timezone.utc)
    validate_games(frame, report['training_through'], observed)
    artifact = joblib.load(path)
    if artifact['source_hashes'] != report['source_hashes'] or artifact['feature_cols'] != report['feature_cols']:
        raise ValueError('Artifact/report contract mismatch')
    details = artifact['model'].predict_details(frame)
    output = []
    for i, game in enumerate(frame.to_dict('records')):
        row = {'game_id':str(game['game_id']), 'kickoff_at':str(game['kickoff_at']),
               'computed_at':pd.Timestamp(observed).isoformat(), 'status':'unpublished_candidate',
               'model_version':artifact['model_version']}
        row.update({key:value[i].tolist() for key, value in details.items() if hasattr(value, 'shape')})
        output.append(row)
    return {'predictions':output, 'note':details['note']+' These outputs are not immutable archived forecasts or production picks.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', required=True)
    parser.add_argument('--features', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        print(json.dumps(predict(Path(__file__).resolve().parent, args.generation, pd.read_csv(args.features)), indent=2, allow_nan=False))
