"""R20 immutable, observation-time-gated starter and weather imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from audit_io import write_json


def timestamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError('Explicit timezone required')
    return result.tz_convert('UTC')


def kickoff(game):
    if pd.isna(game.get('gametime')) or not game.get('gametime'):
        raise ValueError('Unknown kickoff time')
    return pd.Timestamp(f"{game['gameday']} {game['gametime']}",tz='America/New_York').tz_convert('UTC')


def validate(row, game, now):
    kind = row.get('kind')
    if kind not in ['starter','weather']:
        raise ValueError('kind must be starter or weather')
    if now >= kickoff(game) or pd.notna(game.get('home_score')) or pd.notna(game.get('away_score')):
        raise ValueError('Late observations cannot be imported as pregame evidence')
    published = timestamp(row['published_at'])
    if published > now or published >= kickoff(game):
        raise ValueError('Invalid publication timestamp')
    url = urlparse(row['source_url'])
    if url.scheme != 'https' or not url.hostname or url.username or url.password:
        raise ValueError('Public HTTPS source URL required')
    common = {'kind':kind, 'game_id':game['game_id'], 'source_url':row['source_url'],
              'provider':str(row['provider']), 'published_at':published.isoformat()}
    if pd.isna(row['provider']) or not str(row['provider']).strip():
        raise ValueError('Provider name required')
    if kind == 'starter':
        if row.get('status')!='confirmed' or row.get('team') not in [game['home_team'],game['away_team']]:
            raise ValueError('Explicit confirmed status and participating team required')
        if any(pd.isna(row.get(key)) or not str(row.get(key,'')).strip() for key in ['player_id','player_name']):
            raise ValueError('Player identity required')
        return common | {k:row[k] for k in ['team','player_id','player_name','status']}
    valid = timestamp(row['valid_at'])
    if abs((valid-kickoff(game)).total_seconds())>10800:
        raise ValueError('Forecast validity must be within three hours of kickoff')
    temperature, wind = float(row['temperature_f']), float(row['wind_mph'])
    if not np.isfinite([temperature,wind]).all() or not -100<=temperature<=150 or not 0<=wind<=200:
        raise ValueError('Invalid weather values')
    if row.get('stadium_id') != game.get('stadium_id') or not row.get('stadium_id'):
        raise ValueError('Forecast must identify the scheduled stadium')
    return common | {'valid_at':valid.isoformat(),'temperature_f':temperature,'wind_mph':wind,'stadium_id':row['stadium_id']}


def import_rows(root, rows, season, now=None):
    now = timestamp(now or datetime.now(timezone.utc))
    schedule = pd.read_csv(root/f'data/schedule_{season}.csv').set_index('game_id',drop=False)
    path = root/'audit/pregame_inputs.json'
    saved = json.loads(path.read_text()) if path.exists() else {'schema_version':1,'observations':{}}
    added = 0
    # Validate every row before writing anything; observed_at is never supplied by the importer.
    for row in rows:
        if row.get('game_id') not in schedule.index:
            raise ValueError('Unknown game ID')
        game = schedule.loc[row['game_id']].to_dict()
        value = validate(row,game,now)
        identity = hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if identity not in saved['observations']:
            saved['observations'][identity] = value | {'id':identity,'observed_at':now.isoformat(),
                'kickoff_at_observation':kickoff(game).isoformat(), 'season':season,
                'verification':'Supplied confirmation/forecast; source claim not independently verified'}
            added += 1
    write_json(path,saved)
    export(root,season,now)
    return added


def select(records, game, as_of):
    limit = min(timestamp(as_of),kickoff(game))
    candidates = [r for r in records if r['game_id']==game['game_id'] and
                  timestamp(r['observed_at'])<kickoff(game) and timestamp(r['observed_at'])<=limit and
                  timestamp(r['published_at'])<=limit and timestamp(r['published_at'])<kickoff(game)]
    selected = {}
    for row in sorted(candidates,key=lambda r:(r['published_at'],r['observed_at'],r['id'])):
        key = row['kind'] + (':'+row['team'] if row['kind']=='starter' else '')
        if row['kind']=='weather' and (row['stadium_id']!=game.get('stadium_id') or abs((timestamp(row['valid_at'])-kickoff(game)).total_seconds())>10800):
            continue
        selected[key] = row
    return selected


def export(root, season=2026, now=None):
    now = timestamp(now or datetime.now(timezone.utc))
    path = root/'audit/pregame_inputs.json'
    records = list(json.loads(path.read_text())['observations'].values()) if path.exists() else []
    games = []
    for game in pd.read_csv(root/f'data/schedule_{season}.csv').to_dict('records'):
        try:
            chosen = select(records,game,now)
        except ValueError:
            chosen = {}
        games.append({'game_id':game['game_id'],'week':int(game['week']), 'home_team':game['home_team'],
                      'away_team':game['away_team'], 'observations':chosen})
    output = {'season':season,'observations':len(records),'games':games,
              'note':'Timestamped supplied inputs only. observed_at is recorded locally at import, not backfilled. Provider publication times and confirmed-starter claims are supplied assertions, not independently verified. Missing means unavailable, never inferred from past QB usage. No configured automatic provider, no historical backfill, no production-model input changes. Incremental-value evaluation awaits a prospective cohort.'}
    write_json(root/'public_site/data/pregame_inputs.json',output)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--season',type=int,default=2026)
    args=parser.parse_args(); root=Path(__file__).resolve().parent
    if args.input:
        rows = pd.read_csv(args.input).to_dict('records') if args.input.suffix=='.csv' else json.loads(args.input.read_text())
        print('Imported new observations:',import_rows(root,rows,args.season))
    else:
        print('Archived observations:',export(root,args.season)['observations'])
