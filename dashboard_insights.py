"""Prospective forecast history and version-isolated monitoring exports."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from audit_io import write_json
from score_markets import pregame


def stamp(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return result if result.tzinfo else None
    except (ValueError, TypeError, AttributeError):
        return None


def kickoff(row):
    return pd.Timestamp(f"{row['gameday']} {row.get('gametime') or '00:00'}",
                        tz='America/New_York').to_pydatetime().astimezone(timezone.utc)


def metrics(rows):
    decisive = [r for r in rows if r['home_score'] != r['away_score']]
    losses, briers, gaps = [], [], []
    for row in decisive:
        p = max(1e-9, min(1-1e-9, row['probability']))
        y = int(row['home_score'] > row['away_score'])
        losses.append(-math.log(p if y else 1-p))
        briers.append((p-y)**2)
        gaps.append(p-y)
    scored = [r for r in rows if r.get('predicted_home_score') is not None and r.get('predicted_away_score') is not None]
    mean = lambda values: sum(values)/len(values) if values else None
    return {'n': len(decisive), 'ties': len(rows)-len(decisive),
            'log_loss': mean(losses), 'brier': mean(briers), 'calibration_bias': mean(gaps),
            'score_n': len(scored), 'score_mae': mean([(abs(r['home_score']-r['predicted_home_score'])+
                abs(r['away_score']-r['predicted_away_score']))/2 for r in scored])}


def monitor(history, games):
    grouped = defaultdict(list)
    for game in games:
        if game.get('status') != 'Final':
            continue
        start = kickoff(game)
        by_version = defaultdict(list)
        for forecast in history.get(game['game_id'], []):
            at = stamp(forecast.get('forecast_at'))
            if at and at < start:
                by_version[(forecast['model_version'], forecast.get('score_model_version'))].append(forecast)
        for version, forecasts in by_version.items():
            # One forecast per game/version/horizon, not one observation per refresh.
            horizons = defaultdict(list)
            for f in forecasts:
                hours = (start-stamp(f['forecast_at'])).total_seconds()/3600
                horizons['<24h' if hours < 24 else '1-7 days' if hours <= 168 else '>7 days'].append(f)
            for horizon, values in [('Latest pregame', forecasts), *horizons.items()]:
                selected = max(values, key=lambda f: stamp(f['forecast_at']))
                grouped[(*version, horizon)].append({**selected, 'week':game['week'],
                    'game_id':game['game_id'], 'gameday':game['gameday'],
                    'home_score':game['home_score'], 'away_score':game['away_score']})
    result = []
    for (version, score_version, horizon), rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        rows.sort(key=lambda r: (r['gameday'], r['game_id']))
        recent, baseline = metrics(rows[-32:]), metrics(rows[:-32])
        ready = recent['n'] >= 32 and baseline['n'] >= 32
        warning = ready and (recent['log_loss']-baseline['log_loss'] > .10 or recent['brier']-baseline['brier'] > .03)
        result.append({'model_version':version, 'score_model_version':score_version, 'horizon':horizon,
            'overall':metrics(rows), 'recent':recent, 'baseline':baseline,
            'status':'Review deterioration' if warning else 'No threshold exceeded' if ready else 'Insufficient prospective sample',
            'stages':{label:metrics([r for r in rows if low <= r['week'] <= high])
                      for label, low, high in [('Early (1-6)',1,6),('Middle (7-12)',7,12),('Late (13-18)',13,18)]}})
    return {'cohorts':result, 'note':'Only timestamped, pre-kickoff archives. Winner and score versions stay separate; ties excluded from binary metrics. Latest forecast per game/version/horizon. Compare the last 32 games with earlier games (at least 32 decisive games per window). Review thresholds: log loss +0.10 or Brier +0.03. Descriptive alerts, not significance tests or automatic promotion. Calibration bias is mean home probability minus home-win frequency. Horizon cohorts can contain the same games and are not independent.'}


def export(root, now=None):
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    data = root/'public_site/data'
    read = lambda path, default: json.loads(path.read_text()) if path.exists() else default
    games = read(data/'predictions.json', [])
    for season in {g['season'] for g in games}:
        schedule_path = root/f'data/schedule_{season}.csv'
        if schedule_path.exists():
            schedule = pd.read_csv(schedule_path).set_index('game_id')
            for game in games:
                if game['game_id'] in schedule.index:
                    time = schedule.loc[game['game_id']].get('gametime')
                    game['gametime'] = time if pd.notna(time) else None
    audits = read(data/'game_audit.json', {})
    path = root/'audit/dashboard_history.json'
    history = read(path, {})
    for game in games:
        records = history.setdefault(game['game_id'], [])
        # Import only authentic winner records; earlier score forecasts are not reconstructed.
        for archive in sorted((root/'audit/forecasts'/game['game_id']).glob('*.json')):
            f = read(archive, {})
            at = stamp(f.get('forecast_at'))
            if not at or at >= kickoff(game) or at > now:
                continue
            identifier = 'winner:'+f['input_sha256']
            if any(r['id'] == identifier for r in records):
                continue
            records.append({'id':identifier, 'forecast_at':f['forecast_at'], 'model_version':f['model_version'],
                'score_model_version':None, 'probability':f['probability'], 'predicted_home_score':None,
                'predicted_away_score':None, 'features':f.get('features', {}), 'source_hashes':f.get('source_hashes', {})})
        if pregame(game, now):
            audit = audits.get(game['game_id'], {})
            exact = audit.get('model_version') == game['model_version'] and abs(audit.get('home_probability', -1)-game['home_win_probability']) < 1e-8
            features = next((r['features'] for r in records if r['id'] == 'winner:'+audit.get('input_sha256','')), {}) if exact else {}
            payload = {'model_version':game['model_version'], 'score_model_version':game.get('score_model_version'),
                'probability':game['home_win_probability'], 'predicted_home_score':game.get('predicted_home_score'),
                'predicted_away_score':game.get('predicted_away_score'), 'features':features,
                'source_hashes':{k:v['sha256'] for k,v in audit.get('source_receipts',{}).items()} if exact else {}}
            identifier = 'combined:'+hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            combined = [r for r in records if r['id'].startswith('combined:')]
            # Preserve a return to an older forecast too, but skip unchanged refreshes.
            if not combined or max(combined,key=lambda r:r['forecast_at'])['id'].split('@')[0] != identifier:
                records.append({**payload, 'id':identifier+'@'+now.isoformat(), 'forecast_at':now.isoformat()})
        records.sort(key=lambda r: (r['forecast_at'], r['id']))
    write_json(path, history)
    public = {}
    for game_id, records in history.items():
        previous = {}
        public[game_id] = []
        for row in records:
            kind = (row['model_version'], row.get('score_model_version'))
            old = previous.get(kind)
            changed = [{'feature':key,'before':old['features'].get(key),'after':row['features'].get(key)}
                       for key in sorted(set(old['features']) | set(row['features']))
                       if old['features'].get(key) != row['features'].get(key)] if old else []
            public[game_id].append({k:v for k,v in row.items() if k not in ['features','source_hashes']} | {
                'probability_change':row['probability']-old['probability'] if old else None,
                'home_score_change':row['predicted_home_score']-old['predicted_home_score'] if old and row['predicted_home_score'] is not None and old['predicted_home_score'] is not None else None,
                'away_score_change':row['predicted_away_score']-old['predicted_away_score'] if old and row['predicted_away_score'] is not None and old['predicted_away_score'] is not None else None,
                'changed_inputs':changed, 'changed_sources':[k for k,v in row['source_hashes'].items() if old and old['source_hashes'].get(k) != v]})
            previous[kind] = row
    write_json(data/'forecast_history.json', {'games':public, 'note':'History begins with authentic stored observations, not reconstructed forecasts. Winner-only and combined score snapshots have separate comparisons. Changed inputs are associations, not proven causes of forecast movement.'})
    write_json(data/'drift_monitor.json', monitor(history, games))
    return history


if __name__ == '__main__':
    export(Path(__file__).resolve().parent)
