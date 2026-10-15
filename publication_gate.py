"""Read-only R56 publication checks; no model loading or forecast regeneration.

validate_transition(stage, previous) runs dashboard validation and baseline checks,
returns counts, and raises ValueError on rejection. Both paths are repository roots.
An optional keyword-only now (aware datetime) makes lock checks reproducible.

Coverage exceptions belong in public_site/data/coverage_explanations.json:
{"drops": [{"key": "historical/2026/pbp_games", "previous": 79,
            "current": 78, "reason": "Provider removed duplicate game"}]}.
Keys and exact values bind explanations to this transition, not future drops.
Retrieval dates of revised historical sources are NOT performance cutoffs.
"""
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile


PICK_FIELDS = (
    'home_team', 'away_team', 'home_win_probability', 'away_win_probability',
    'predicted_winner', 'model_version', 'winner_forecast_at',
    'predicted_home_score', 'predicted_away_score', 'score_model_version',
    'score_forecast_at', 'predicted_margin', 'predicted_total',
    'model_spread_side', 'model_total_side',
)


def read(path):
    def invalid(value):
        raise ValueError(f'Invalid JSON number in {path}: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def timestamp(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError('timezone required')
        return result
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f'Invalid provenance timestamp: {value!r}') from exc


def before(value, boundary, label):
    # Date-only cutoffs represent inclusive performance days, not retrieval dates.
    try:
        if isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            valid = date.fromisoformat(value) < boundary.date()
        else:
            valid = timestamp(value) < boundary
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Invalid {label}') from exc
    if not valid:
        raise ValueError(f'{label} overlaps forecast or kickoff')


def validate_shadow_provenance(root):
    """Verify exact candidate snapshots, not mutable current-source receipts."""
    root = Path(root)
    path = root/'public_site/data/shadow_next.json'
    if not path.exists():
        return
    snapshots = {}
    reports = {}
    for record in read(path)['forecasts']:
        forecast = timestamp(record.get('forecast_at'))
        kickoff = timestamp(record.get('kickoff_at_observation'))
        if forecast >= kickoff:
            raise ValueError('Late shadow forecast')
        for boundary in (forecast, kickoff):
            before(record.get('training_through'), boundary, 'Shadow training cutoff')
        if record['model'].startswith('production_'):
            original = timestamp(record.get('original_forecast_at'))
            if original > forecast or original >= kickoff:
                raise ValueError('Invalid copied production forecast provenance')
            continue
        generation = record['generation']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', generation):
            raise ValueError('Invalid shadow generation')
        digest = record.get('input_sha256', '')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Missing exact-input digest')
        key = (generation, digest)
        if key not in snapshots:
            source = root/'audit/shadow_next'/generation/'inputs'/(digest+'.json')
            if not source.is_file():
                raise ValueError(f'Missing exact-input snapshot: {digest}')
            snapshot = read(source)
            values = snapshot.get('features')
            if not isinstance(values, dict) or not values or any(
                v is not None and (type(v) not in (int, float) or not math.isfinite(v))
                for v in values.values()
            ):
                raise ValueError('Invalid exact-input features')
            actual = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
            if actual != digest:
                raise ValueError('Exact-input hash mismatch')
            snapshots[key] = snapshot
        snapshot = snapshots[key]
        observed = timestamp(snapshot.get('observed_at'))
        if observed > forecast or observed >= kickoff:
            raise ValueError('Exact-input observation is after forecast or kickoff')
        # Optional explicit performance provenance is distinct from retrieval metadata.
        for field in ('performance_through', 'performance_cutoff'):
            if snapshot.get(field) is not None:
                before(snapshot[field], forecast, 'Input performance cutoff')
                before(snapshot[field], kickoff, 'Input performance cutoff')
        if generation not in reports:
            report_path = root/'models/retrained'/generation/'report.json'
            if not report_path.is_file():
                report_path = root/'backtests/retraining'/generation/'report.json'
            if not report_path.is_file():
                raise ValueError('Missing shadow training report')
            reports[generation] = read(report_path)
        if record.get('training_through') != reports[generation].get('training_through'):
            raise ValueError('Shadow training cutoff differs from training report')


def coverage(root):
    """Comparable counts from the existing source-coverage schemas."""
    root = Path(root)
    folder = root/'public_site/data'
    result = {}
    path = folder/'historical_coverage.json'
    if path.exists():
        for season in read(path)['seasons']:
            for field in ('scheduled_games', 'pbp_games', 'plays'):
                result[f'historical/{season["season"]}/{field}'] = season[field]
    path = folder/'expanded_coverage.json'
    if path.exists():
        report = read(path)
        for season, count in report['pbp'].items():
            result[f'expanded/pbp/{season}'] = count
        for kind, counts in report['ngs'].items():
            for field in ('current_team_weeks', 'latest_week'):
                result[f'expanded/{report["through_season"]}/ngs/{kind}/{field}'] = counts[field]
        result['expanded/dataset_rows'] = report['dataset_rows']
    for path in (root/'data').glob('current_features_*.json'):
        report = read(path)
        for source, count in report.get('sources', {}).items():
            result[f'current/{report["season"]}/{source}'] = count
    path = folder/'data_quality.json'
    if path.exists():
        for row in read(path).get('missingness', []):
            if not 0 <= row['missing'] <= row['rows']:
                raise ValueError('Invalid source missingness')
            result[f'features/{row["feature"]}/available'] = row['rows'] - row['missing']
    for key, value in result.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or int(value) != value:
            raise ValueError(f'Invalid coverage count: {key}')
    return result


def validate_transition(root, previous_root, *, now=None):
    """Validate a staged repository against the last published repository.

    Raises ValueError before publication on bad data, changed/deleted locked picks,
    invalid shadow provenance, or unexplained coverage loss. Never writes files.
    """
    from validate_dashboard import validate
    root, previous_root = Path(root), Path(previous_root)
    result = validate(root)
    now = datetime.now(timezone.utc) if now is None else timestamp(now.isoformat())
    folder = root/'public_site/data'
    prior = previous_root/'public_site/data'
    old_rows = read(prior/'predictions.json')
    current = {r['game_id']: r for r in read(folder/'predictions.json')}
    kickoffs = read(prior/'kickoffs.json') if (prior/'kickoffs.json').exists() else {}
    locked = 0
    for old in old_rows:
        game_id = old['game_id']
        kickoff = kickoffs.get(game_id) or old.get('kickoff_at_observation') or old.get('kickoff_at')
        if kickoff:
            started = timestamp(kickoff) <= now
        else:
            # No precise kickoff: lock conservatively at the start of game day.
            try:
                started = date.fromisoformat(old['gameday']) <= now.date()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Missing previous kickoff: {game_id}') from exc
        if not (started or old.get('status') in ('Final', 'In Progress', 'Live')):
            continue
        locked += 1
        if game_id not in current:
            raise ValueError(f'Locked forecast removed: {game_id}')
        fields = list(PICK_FIELDS)
        if old.get('line_observed_at') is not None:
            fields += ['market_home_spread', 'market_total', 'line_observed_at']
        for field in fields:
            if old.get(field) != current[game_id].get(field):
                raise ValueError(f'Locked forecast changed: {game_id} {field}')
    baseline, candidate = coverage(previous_root), coverage(root)
    path = folder/'coverage_explanations.json'
    explanations = read(path).get('drops', []) if path.exists() else []
    explained = 0
    for key, previous in baseline.items():
        value = candidate.get(key)
        if value is not None and value >= previous:
            continue
        if not any(e.get('key') == key and e.get('previous') == previous
                   and 'current' in e and e['current'] == value
                   and isinstance(e.get('reason'), str) and e['reason'].strip()
                   for e in explanations):
            raise ValueError(f'Unexplained source coverage drop: {key}: {previous} -> {value}')
        explained += 1
    return dict(result, locked_forecasts=locked, coverage_series=len(baseline),
                explained_coverage_drops=explained)


def against_git(root, reference='HEAD'):
    """Read only the baseline files needed for transition checks from Git."""
    root = Path(root)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', reference,
                                    '--', 'public_site/data', 'data'], cwd=root, text=True).splitlines()
    wanted = {'predictions.json', 'kickoffs.json', 'historical_coverage.json',
              'expanded_coverage.json', 'data_quality.json'}
    with tempfile.TemporaryDirectory(prefix='nfl-publication-baseline-') as temporary:
        previous = Path(temporary)
        for name in names:
            path = Path(name)
            if not ((name.startswith('public_site/data/') and path.name in wanted)
                    or re.fullmatch(r'data/current_features_\d{4}\.json', name)):
                continue
            target = previous/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(['git', 'show', f'{reference}:{name}'], cwd=root))
        return validate_transition(root, previous)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--against', default='HEAD')
    args = parser.parse_args()
    print(json.dumps(against_git(Path(__file__).resolve().parent, args.against)))
