"""Generation-isolated immutable prospective records. Called only by the cloud writer."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from generative_pipeline import digest, features, forecast
from pregame_inputs import kickoff, timestamp


def save_once(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation: no earlier forecast or result may be overwritten.
    payload = json.dumps(record,indent=2,allow_nan=False)
    try:
        with path.open('x',encoding='utf-8') as f:
            f.write(payload)
        return True
    except FileExistsError:
        return False


def eligible(game, artifact, now, horizon):
    try:
        minutes = (kickoff(game)-timestamp(now)).total_seconds()/60
    except ValueError:
        return False
    center,tolerance = {'24h':(1440,30),'90m':(90,15)}[horizon]
    return (pd.isna(game.get('home_score')) and pd.isna(game.get('away_score'))
            and game['season']==artifact['season'] and game['week']>artifact['training_max_week']
            and abs(minutes-center)<=tolerance)


def review(folder, schedule):
    folder = Path(folder)
    schedule = schedule.set_index('game_id')
    matched = [];corrections=[]
    for path in sorted((folder/'forecasts').glob('*.json')):
        record = json.loads(path.read_text())
        if record['model_version'] != folder.name or path.name != record['game_id']+'_'+record['horizon']+'.json':
            raise ValueError('Mixed generation or duplicate forecast identity')
        if record['horizon'] != '90m' or record['game_id'] not in schedule.index:
            continue
        game = schedule.loc[record['game_id']].to_dict()
        if pd.isna(game.get('home_score')) or pd.isna(game.get('away_score')):
            continue
        if timestamp(record['forecast_at']) >= kickoff(game):
            raise ValueError('Post-kickoff record cannot be graded prospectively')
        actual = [float(game['home_score']),float(game['away_score'])]
        label = 0 if actual[0]>actual[1] else 1 if actual[0]==actual[1] else 2
        p = record['prediction']['distributions']['ensemble']
        vector = np.array([p['home_win'],p['tie'],p['away_win']])
        result = dict(game_id=record['game_id'],model_version=record['model_version'],week=int(game['week']),
            forecast_sha256=digest(path),actual=actual,tie=label==1,
            correct=None if label==1 or vector[0]==vector[2] else bool((vector[0]>vector[2]) == (label==0)),
            brier3=float(np.sum((vector-np.eye(3)[label])**2)))
        target = folder/'results'/path.name
        save_once(target,result)
        saved = json.loads(target.read_text())
        if saved['forecast_sha256'] != result['forecast_sha256']:
            raise ValueError('Forecast changed after grading')
        if saved['actual'] != actual:corrections.append(record['game_id'])
        matched.append(saved)
    decisive = [r for r in matched if r['correct'] is not None]
    weeks = len({r['week'] for r in matched})
    return dict(matched_games=len(matched),decisive_games=len(decisive),weeks=weeks,
                accuracy=float(np.mean([r['correct'] for r in decisive])) if decisive else None,
                ready_for_review=len(matched)>=100 and weeks>=6 and not corrections,automatic_promotion=False,result_corrections=corrections,
                note='Candidate-only prospective evidence; not proof of superiority. Review requires 100 games over six weeks and explicit human approval. No mixed generations or replay labels.')


def export(root, season, now=None):
    root = Path(root)
    selector = root/'models/generative/active_shadow.json'
    if not selector.exists():
        return None
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!='nbianco-osu/nfl-ds-pred-runner':
        raise ValueError('Prospective captures belong to the sole private cloud writer')
    config = json.loads(selector.read_text())
    version = config['generation']
    if not version.startswith('generative-') or Path(version).name != version:
        raise ValueError('Invalid generation')
    artifact_path = root/'models/generative'/version/'candidate.joblib'
    if digest(artifact_path) != config['artifact_sha256']:
        raise ValueError('Pinned artifact changed')
    artifact = joblib.load(artifact_path)
    if artifact['model_version'] != version:
        raise ValueError('Generation mismatch')
    schedule_path = root/f'data/schedule_{season}.csv'
    schedule = pd.read_csv(schedule_path)
    folder = root/'audit/generative'/version
    for game in schedule.loc[schedule.game_type.eq('REG')].to_dict('records'):
        for horizon in ['24h','90m']:
            started = datetime.now(timezone.utc) if now is None else timestamp(now)
            if not eligible(game,artifact,started,horizon):continue
            target = folder/'forecasts'/(game['game_id']+'_'+horizon+'.json')
            if target.exists():continue
            history_path = Path(__file__).resolve().parent/'backtests/source_schedule.csv'
            historical = pd.read_csv(history_path)
            history = pd.concat([historical.loc[historical.season.lt(season)],schedule],ignore_index=True)
            if digest(history_path) != artifact['source_hashes']['historical_schedule']:
                raise ValueError('Historical feature source changed')
            inp = features(history,pd.DataFrame([game]))
            prediction,_ = forecast(artifact,game,inp,int(config.get('trials',1000)))
            finished = datetime.now(timezone.utc) if now is None else timestamp(now)
            if not eligible(game,artifact,finished,horizon):continue
            record = dict(model_version=version,game_id=game['game_id'],horizon=horizon,
                forecast_at=finished.isoformat(),artifact_sha256=config['artifact_sha256'],
                schedule_sha256=digest(schedule_path),training_source_hashes=artifact['source_hashes'],
                features=inp[['home_for','home_against','away_for','away_against']].iloc[0].to_dict(),
                game={k:game[k] if pd.notna(game.get(k)) else None for k in ['gameday','gametime','home_team','away_team','week','spread_line','total_line']},
                prediction=prediction)
            save_once(target,record)
    result = review(folder,schedule)
    result['model_version'] = version
    from audit_io import write_json
    write_json(root/'public_site/data/generative_readiness.json',result)
    return result
