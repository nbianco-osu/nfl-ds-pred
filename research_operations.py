"""R30-R39 operational research exports; never modifies serving models or picks."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_io import write_json
from pregame_inputs import kickoff,timestamp
from live_challengers import sample


def read(root,path,default=None):
    p=root/path
    return json.loads(p.read_text()) if p.exists() else default


def checkpoints(root,season=2026,now=None):
    now=timestamp(now or datetime.now(timezone.utc))
    schedule=pd.read_csv(root/f'data/schedule_{season}.csv')
    predictions={r['game_id']:r for r in read(root,'public_site/data/predictions.json',[])}
    health=read(root,'public_site/data/refresh_health.json',{})
    refreshed=timestamp(health['last_successful_refresh']) if health.get('last_successful_refresh') else None
    fresh=refreshed is not None and pd.Timedelta(0)<=now-refreshed<=pd.Timedelta(minutes=30)
    path=root/'audit/checkpoints.json';archive=read(root,'audit/checkpoints.json',{})
    status=[]
    for g in schedule.loc[schedule.game_type.eq('REG')].to_dict('records'):
        start=kickoff(g)
        for label,hours,tolerance in [('24h',24,.5),('90m',1.5,.25)]:
            remaining=(start-now).total_seconds()/3600
            key=g['game_id']+'|'+label
            value='Captured' if key in archive else 'Missed window' if remaining<hours-tolerance else 'Not yet due'
            if key not in archive and abs(remaining-hours)<=tolerance and now<start:
                p=predictions.get(g['game_id'])
                value='Needs fresh validated forecast'
                if fresh and p and p['status']!='Final' and p.get('home_win_probability') is not None:
                    archive[key]={'game_id':g['game_id'],'horizon':label,'observed_at':now.isoformat(),
                        'kickoff_at_observation':start.isoformat(),'minutes_before_kickoff':remaining*60,
                        'validated_refresh_at':refreshed.isoformat(),'prediction':p}
                    value='Captured'
            status.append({'game_id':g['game_id'],'horizon':label,'status':value})
    write_json(path,archive)
    report={'observations':list(archive.values()),'windows':status,
        'note':'Actual captures only: 24h +/-30 minutes or 90m +/-15 minutes before kickoff, after a validated refresh within 30 minutes. No synthetic/backfilled snapshots. First capture is immutable. The pregame scheduler checks every 10 minutes; execution requires the local host and scheduler to be available. Refresh duration can miss a window; capture is not guaranteed.'}
    write_json(root/'public_site/data/checkpoints.json',report)


def error_analysis(root):
    predictions=read(root,'public_site/data/predictions.json',[])
    final=[r for r in predictions if r['status']=='Final' and r['home_score']!=r['away_score']]
    groups={}
    for r in final:
        version=r.get('model_version','legacy')
        p=float(r['home_win_probability']);confidence=max(p,1-p)
        band='50-59%' if confidence<.6 else '60-69%' if confidence<.7 else '70%+'
        provenance='Timestamped archive' if r.get('winner_forecast_at') else 'Legacy; timestamp unknown'
        for dimension,value in [('confidence',band),('team',r['home_team']),('team',r['away_team']),
                                ('stage','Weeks 1-6' if r['week']<=6 else 'Weeks 7-12' if r['week']<=12 else 'Weeks 13+'),('provenance',provenance)]:
            groups.setdefault((version,dimension,value),[]).append(r)
    rows=[]
    for (v,d,value),g in sorted(groups.items()):
        p=np.clip([r['home_win_probability'] for r in g],1e-8,1-1e-8)
        y=np.array([r['home_score']>r['away_score'] for r in g])
        rows.append({'version':v,'dimension':d,'group':value,'games':len(g),
            'accuracy':float(((p>=.5)==y).mean()),'log_loss':float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()),
            'brier':float(((p-y)**2).mean()),'status':'Small sample; descriptive only' if len(g)<10 else 'Descriptive; overlapping groups'})
    write_json(root/'public_site/data/error_analysis.json',{'groups':rows,
        'note':'Saved production picks only, separated by model version. Ties excluded. Team groups overlap and must not be summed. Historical input availability cannot be reconstructed; provenance grouping reports archive evidence, not injury status. Under 10 games is flagged; no subgroup significance claims.'})


def quality_registry(root,season):
    schedule=pd.read_csv(root/f'data/schedule_{season}.csv')
    current_path=root/f'data/current_features_{season}.csv'
    current=pd.read_csv(current_path) if current_path.exists() else pd.DataFrame()
    checks=[{'check':'Unique schedule keys','passed':not schedule.game_id.duplicated().any()},
            {'check':'Unique feature keys','passed':not current.empty and not current.game_id.duplicated().any()},
            {'check':'Complete feature game coverage','passed':not current.empty and set(schedule.game_id)<=set(current.game_id)}]
    numeric=current.select_dtypes(include='number')
    checks.append({'check':'No infinite numeric features','passed':not np.isinf(numeric.to_numpy()).any()})
    missing=[{'feature':c,'missing':int(current[c].isna().sum()),'rows':len(current)} for c in current if c!='game_id']
    write_json(root/'public_site/data/data_quality.json',{'checks':checks,'missingness':sorted(missing,key=lambda r:r['missing'],reverse=True),
        'note':'Schema/key/finite-value checks and observed missingness, not proof of provider correctness. Missing values are not replaced here. No arbitrary healthy-player or neutral-team fill.'})
    models=[]
    for path in sorted((root/'models').rglob('*.joblib')):
        models.append({'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
            'role':'Production classifier' if path.name=='home_win_advanced_automl_no_market_deep.joblib' else 'Separate artifact; role defined by training metadata'})
    sources=[]
    for path in sorted((root/'audit/sources').glob('*/latest.json')):
        meta=json.loads(path.read_text());sources.append({'source':meta.get('source'),'sha256':meta.get('sha256'),
            'rows':meta.get('rows'),'retrieved_at':meta.get('retrieved_at'),'provider_published_at':meta.get('provider_published_at')})
    write_json(root/'public_site/data/model_registry.json',{'models':models,'sources':sources,
        'note':'Artifact inventory with SHA-256 fingerprints, not a matched performance ranking. Retrieval timestamps are not original publication times. Models are not loaded by the browser; no credentials or raw private sources are exposed.'})
    write_json(root/'public_site/data/kickoffs.json',{g['game_id']:kickoff(g).isoformat() for g in schedule.to_dict('records')})


def scenario_distributions(root):
    config=read(root,'models/research/live_challengers.json')
    if not config:return
    artifact=joblib.load(root/config['artifact'])
    live=read(root,'public_site/data/live_challengers.json',{'forecasts':[]})
    scenarios=[]
    for r in live['forecasts']:
        if r['kind']!='correlated_negative_binomial' or r['model_version']!=artifact['model_version']:continue
        x=pd.DataFrame([r['features']],columns=artifact['feature_cols'])
        mean=artifact['model'].predict(x)[0]
        draws=sample(artifact,mean,r['kind'])
        values,counts=np.unique(draws[:,0]-draws[:,1],return_counts=True)
        scenarios.append({'game_id':r['game_id'],'model_version':r['model_version'],'forecast_at':r['forecast_at'],
            'expected_home':float(draws[:,0].mean()),'expected_away':float(draws[:,1].mean()),
            'margin_histogram':[[int(v),int(c)] for v,c in zip(values,counts)]})
    write_json(root/'public_site/data/qb_scenarios.json',{'games':scenarios,
        'note':'Hypothetical QB/availability sensitivity, not a player valuation model. User-specified point adjustments shift archived shadow score margins; baseline uncertainty is held fixed. No confirmed-starter inference, causal claim, production update or automatic bet. Extreme adjustments may be unrealistic.'})


def replay_audit(root):
    path=root/'models/home_win_advanced_automl_no_market_deep.joblib'
    digest=hashlib.sha256(path.read_bytes()).hexdigest();artifact=joblib.load(path)
    latest=read(root,'public_site/data/game_audit.json',{})
    records=[]
    for game,meta in sorted(latest.items())[:10]:
        record_path=root/'audit/forecasts'/game/(meta['input_sha256']+'.json')
        if not record_path.exists():
            records.append({'game_id':game,'status':'Unavailable original feature record'});continue
        record=json.loads(record_path.read_text())
        if record['model_sha256']!=digest:
            records.append({'game_id':game,'status':'Different model artifact; not replayed'});continue
        x=pd.DataFrame([record['features']],columns=artifact['feature_cols'])
        p=float(artifact['model'].predict_proba(x)[0,1]);difference=abs(p-record['probability'])
        records.append({'game_id':game,'status':'Reproduced' if difference<1e-8 else 'Mismatch','absolute_difference':difference})
    write_json(root/'public_site/data/reproducibility.json',{'model_sha256':digest,'sample':records,
        'note':'Deterministic replay of up to ten original archived input records with the matching local production artifact. Older archives used 10-decimal JSON precision, which can affect tree thresholds; mismatches remain visible and records are not rewritten. Future archives preserve full numeric precision. No reconstruction of missing legacy inputs. Sampling is not an exhaustive audit; passing does not establish predictive skill.'})


def export(root,season=2026,audit=False,scenarios=False):
    checkpoints(root,season);error_analysis(root);quality_registry(root,season)
    if audit:replay_audit(root)
    if scenarios:scenario_distributions(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--season',type=int,default=2026)
    parser.add_argument('--audit',action='store_true');parser.add_argument('--scenarios',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=2):export(Path(__file__).resolve().parent,args.season,args.audit,args.scenarios)
