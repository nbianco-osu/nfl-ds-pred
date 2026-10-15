"""R23: observation-timestamped Open-Meteo forecasts, never historical backfill."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from audit_io import atomic_bytes, write_json
from pregame_inputs import kickoff, timestamp, export

VENUES='https://raw.githubusercontent.com/greerreNFL/Stadiums/main/data/stadiums.csv'
API='https://api.open-meteo.com/v1/forecast'


def read_url(url):
    with urlopen(url,timeout=30) as response:
        return response.read()


def observation(game,payload,observed,url,response_hash):
    observed=timestamp(observed)
    if observed>=kickoff(game) or pd.notna(game.get('home_score')) or pd.notna(game.get('away_score')):
        raise ValueError('Cannot archive post-kickoff weather')
    units=payload['hourly_units']
    if units.get('temperature_2m') not in ['F','°F'] or units.get('wind_speed_10m')!='mp/h':
        raise ValueError('Unexpected weather units')
    hourly=payload['hourly']
    times=pd.to_datetime(hourly['time'],utc=True)
    i=int(np.argmin(abs(times-kickoff(game))))
    if abs((times[i]-kickoff(game)).total_seconds())>3600:
        raise ValueError('Kickoff outside forecast horizon')
    temp,wind=float(hourly['temperature_2m'][i]),float(hourly['wind_speed_10m'][i])
    if not np.isfinite([temp,wind]).all() or not -100<=temp<=150 or not 0<=wind<=200:
        raise ValueError('Invalid weather values')
    value={'kind':'weather','game_id':game['game_id'],'season':int(game['season']),
        'stadium_id':game['stadium_id'],'provider':'Open-Meteo','source_url':url,
        'published_at':None,'timestamp_basis':'observed_at; provider issue time unavailable',
        'observed_at':observed.isoformat(),'kickoff_at_observation':kickoff(game).isoformat(),
        'valid_at':times[i].isoformat(),'temperature_f':temp,'wind_mph':wind,
        'response_sha256':response_hash,'verification':'Retrieved public forecast; outdoor stadium conditions, not indoor playing conditions'}
    # Forecast values plus response identity deduplicate retries without inventing issue times.
    identity={k:v for k,v in value.items() if k!='observed_at'}
    value['id']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return value


def collect(root,season=2026,now=None,fetch=read_url):
    now=timestamp(now or datetime.now(timezone.utc))
    schedule=pd.read_csv(root/f'data/schedule_{season}.csv')
    path=root/'audit/pregame_inputs.json'
    saved=json.loads(path.read_text()) if path.exists() else {'schema_version':1,'observations':{}}
    status={'updated_at':now.isoformat(),'provider':'Open-Meteo','games':[], 'added':0,
        'note':'Non-commercial Open-Meteo forecast API; CC BY 4.0 attribution. Outdoor grid-cell weather only. Provider issue time unavailable: locally observed time is retained, never called publication time. Confirmed QB inputs remain manual imports with source URLs; previous usage is not confirmation.'}
    try:
        content=fetch(VENUES)
        venues=pd.read_csv(StringIO(content.decode())).set_index('stadium_id')
        if not venues.index.is_unique:
            raise ValueError('Duplicate stadium IDs')
        status['venue_source']=VENUES
        status['venue_sha256']=hashlib.sha256(content).hexdigest()
    except Exception as exc:
        status['error']='Venue lookup failed: '+type(exc).__name__
        write_json(root/'public_site/data/input_collection.json',status)
        return status
    cache={}
    for game in schedule.loc[schedule.game_type.eq('REG')].to_dict('records'):
        try:
            start=kickoff(game)
            if start<=now or pd.notna(game['home_score']) or pd.notna(game['away_score']):
                continue
            row={'game_id':game['game_id'],'status':'Outside 16-day horizon'}
            status['games'].append(row)
            if start>now+pd.Timedelta(days=15):
                continue
            if str(game.get('roof','')).lower() in ['dome','closed']:
                row['status']='Indoor; outdoor forecast not used'
                continue
            if game['stadium_id'] not in venues.index:
                row['status']='Stadium coordinates unavailable';continue
            venue=venues.loc[game['stadium_id']]
            lat,lon=float(venue['lat']),float(venue['lon'])
            if not np.isfinite([lat,lon]).all() or not -90<=lat<=90 or not -180<=lon<=180:
                raise ValueError('Invalid stadium coordinates')
            url=API+'?'+urlencode({'latitude':lat,'longitude':lon,'hourly':'temperature_2m,wind_speed_10m',
                'temperature_unit':'fahrenheit','wind_speed_unit':'mph','timezone':'UTC','forecast_days':16})
            if url not in cache:
                body=fetch(url)
                digest=hashlib.sha256(body).hexdigest()
                atomic_bytes(root/'audit/weather_responses'/(digest+'.json'),body)
                cache[url]=(json.loads(body),digest,timestamp(datetime.now(timezone.utc)))
            payload,digest,observed=cache[url]
            value=observation(game,payload,observed,url,digest)
            if value['id'] not in saved['observations']:
                saved['observations'][value['id']]=value;status['added']+=1
            row['status']='Forecast archived'
        except Exception as exc:
            status['games'].append({'game_id':game['game_id'],'status':'Forecast unavailable','error':type(exc).__name__})
    write_json(path,saved)
    write_json(root/'public_site/data/input_collection.json',status)
    export(root,season)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--season',type=int,default=2026)
    args=parser.parse_args();r=collect(Path(__file__).resolve().parent,args.season)
    print('Weather observations added:',r['added']);print('Collection error:',r.get('error'))
