"""R27 actionable, deterministic monitoring; no secret or raw exception export."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
from audit_io import write_json
from pregame_inputs import kickoff, timestamp


def read(root,path,default=None):
    file=root/path
    return json.loads(file.read_text()) if file.exists() else default


def export(root,season=2026,now=None):
    now=timestamp(now or datetime.now(timezone.utc))
    issues=[]
    def add(key,severity,message,action):
        issues.append({'id':key,'severity':severity,'message':message,'action':action})
    health=read(root,'public_site/data/refresh_health.json',{})
    last=health.get('last_successful_refresh')
    if not last or now-timestamp(last)>pd.Timedelta(days=8):
        add('stale_refresh','error','No successful refresh within eight days.','Run refresh_predictions.py and inspect validation failures.')
    attempt=read(root,'audit/refresh_attempt.json',{})
    if attempt.get('status')=='failed':
        add('refresh_failed','error','Latest refresh failed ('+attempt.get('error_type','unknown')+').','Inspect local run logs; last validated forecasts are retained.')
    coverage=read(root,f'data/current_features_{season}.json',{})
    input_attempt=read(root,'audit/input_attempt.json',{})
    if input_attempt.get('status')=='failed':
        add('input_import_failed','warning','Latest supplied-input import was rejected.','Correct the input schema/timestamps and rerun; prior archive is preserved.')
    for key,value in coverage.get('sources',{}).items():
        if key.endswith('_error'):
            add('source_'+key,'warning',key.removesuffix('_error')+' source unavailable.','Review upstream availability; do not replace missing reports with healthy status.')
    preds=read(root,'public_site/data/predictions.json',[])
    lookup={r['game_id']:r for r in preds}
    schedule=pd.read_csv(root/f'data/schedule_{season}.csv')
    upcoming=[]
    for g in schedule.loc[schedule.game_type.eq('REG')].to_dict('records'):
        if pd.isna(g['home_score']) and pd.isna(g['away_score']) and now<kickoff(g)<=now+pd.Timedelta(days=7):
            upcoming.append(g)
    missing=[g['game_id'] for g in upcoming if g['game_id'] not in lookup or lookup[g['game_id']].get('home_win_probability') is None or lookup[g['game_id']].get('predicted_home_score') is None]
    if missing:
        add('missing_predictions','error',f'{len(missing)} upcoming games lack winner or score predictions.','Regenerate and validate forecasts before publishing.')
    inputs=read(root,'public_site/data/pregame_inputs.json',{'games':[]})
    archived={g['game_id']:g['observations'] for g in inputs['games']}
    missing_qb=sum('starter:'+g[s+'_team'] not in archived.get(g['game_id'],{}) for g in upcoming for s in ['home','away'])
    if missing_qb:
        add('missing_starters','warning',f'{missing_qb} starting-QB confirmations unavailable for the next seven days.','Import timestamped confirmations; do not infer starters from previous usage.')
    outdoor=[g for g in upcoming if str(g.get('roof','')).lower() not in ['dome','closed']]
    missing_weather=sum('weather' not in archived.get(g['game_id'],{}) for g in outdoor)
    if missing_weather:
        add('missing_weather','warning',f'{missing_weather} upcoming outdoor forecasts unavailable.','Run weather_inputs.py; inspect coordinate and provider errors.')
    stale_weather=sum('weather' in archived.get(g['game_id'],{}) and now-timestamp(archived[g['game_id']]['weather']['observed_at'])>pd.Timedelta(hours=48) for g in outdoor)
    if stale_weather:
        add('stale_weather','warning',f'{stale_weather} weather forecasts are older than 48 hours.','Refresh weather before using it for matchup analysis.')
    collection=read(root,'public_site/data/input_collection.json',{})
    if collection.get('error') or any(r.get('error') for r in collection.get('games',[])):
        add('weather_collection','warning','Latest weather collection has failures.','Inspect input_collection.json; archived observations are preserved.')
    queue=read(root,'data/warehouse_queue/latest.json',{})
    receipt=read(root,'database/normalized_verification.json',{})
    # Inspect actual manifest key names; equality alone is not a remote verification.
    queue_hash=queue.get('queue_sha256') or queue.get('sha256')
    receipt_hash=receipt.get('queue_sha256')
    if queue_hash and queue_hash!=receipt_hash:
        add('warehouse_pending','warning','Database queue is not verified as synced.','Execute and verify pending connector batches, then record the remote receipt.')
    live=read(root,'public_site/data/live_challengers.json',{'forecasts':[]})
    replay=read(root,'public_site/data/reproducibility.json',{'sample':[]})
    mismatches=sum(r.get('status')=='Mismatch' for r in replay['sample'])
    if mismatches:
        add('replay_mismatch','warning',f'{mismatches} sampled archived forecasts differ on replay.','Inspect reproducibility.json; preserve old rounded-input records and use full precision for future archives.')
    live_ids={r['game_id'] for r in live['forecasts']}
    if any(g['game_id'] not in live_ids for g in upcoming):
        add('missing_challengers','warning','Some upcoming games have no saved shadow forecast.','Run live_challengers.py using the pinned artifacts; do not retrain automatically.')
    report={'updated_at':now.isoformat(),'status':'attention' if issues else 'healthy','issues':issues,
        'last_successful_refresh':last,'upcoming_games':len(upcoming),'missing_prediction_ids':missing,
        'note':'Operational checks, not model-performance guarantees. Warnings can indicate intentionally unavailable inputs. Notifications should fire only when the issue fingerprint changes or a substantive deployment completes.'}
    report['issue_fingerprint']=hashlib.sha256(json.dumps(issues,sort_keys=True).encode()).hexdigest()
    write_json(root/'public_site/data/reliability.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--season',type=int,default=2026)
    args=parser.parse_args();print(json.dumps(export(Path(__file__).resolve().parent,args.season),indent=2))
