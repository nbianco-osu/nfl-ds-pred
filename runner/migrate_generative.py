"""Explicit shadow-only migration serialized with the existing private refresh writer."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

from audit_io import write_json
from runner.common import GitHub,credential,PRIVATE,pack,unpack,inventory,digest


def allowed(files,version):
    return set(files)=={f'models/generative/{version}/candidate.joblib',f'models/generative/{version}/report.json',f'models/generative/{version}/historical_schedule.csv',
                        'public_site/data/generative.json'}


def prepare(root,run):
    report=json.loads((run/'report.json').read_text());version=report['model_version']
    if report['status']!='shadow_only' or digest(run/'candidate.joblib')!=report['artifact_sha256']:
        raise ValueError('Invalid research artifact')
    destination=root/'data/generative_migrations'/version
    destination.mkdir(parents=True,exist_ok=False)
    payload=destination/'payload'
    model=payload/'models/generative'/version;model.mkdir(parents=True)
    for name in ['candidate.joblib','report.json']:shutil.copyfile(run/name,model/name)
    historical=root/'backtests/source_schedule.csv'
    if digest(historical)!=report['source_hashes']['historical_schedule']:raise ValueError('Frozen historical source changed')
    shutil.copyfile(historical,model/'historical_schedule.csv')
    public=payload/'public_site/data';public.mkdir(parents=True)
    shutil.copyfile(root/'public_site/data/generative.json',public/'generative.json')
    files=pack(payload,destination/'payload.tar.gz')
    if not allowed(files,version):raise ValueError('Unexpected payload')
    gh=GitHub(credential());gh.assert_private()
    uploaded=gh.upload(destination/'payload.tar.gz')
    manifest=dict(uploaded,files=files,generation=version)
    # A tiny manifest is stored in the private repository; no credentials in it.
    head=gh.head()
    gh.commit(head,{f'generative-migrations/{version}.json':json.dumps(manifest,indent=2)},'Stage approved generative shadow payload')
    print(json.dumps({'generation':version,'manifest':f'generative-migrations/{version}.json'}))


def migrate():
    from runner.run import guard_live
    from publication_gate import validate_transition
    from dashboard_snapshot import export as snapshot
    from generative_tracking import export as track
    gh=GitHub(os.environ['GH_TOKEN']);gh.assert_private();parent=gh.head()
    config=json.loads(gh.read('config.json',parent));guard_live(config)
    state=json.loads(gh.read('state.json',parent));version=os.environ['GENERATION']
    if not version.startswith('generative-') or Path(version).name!=version:raise ValueError('Invalid generation')
    manifest=json.loads(gh.read(f'generative-migrations/{version}.json',parent))
    if manifest['generation']!=version or not allowed(manifest['files'],version):raise ValueError('Invalid migration scope')
    if any(name in state['files'] for name in manifest['files'] if name.startswith('models/')):raise ValueError('Generation already installed')
    with tempfile.TemporaryDirectory(prefix='nfl-generative-') as temp:
        base=Path(temp);current=base/'current';previous=base/'previous'
        gh.download(state,base/'old.tar.gz');unpack(base/'old.tar.gz',current,state['files']);unpack(base/'old.tar.gz',previous,state['files'])
        gh.download(manifest,base/'payload.tar.gz');unpack(base/'payload.tar.gz',current,manifest['files'])
        report=json.loads((current/f'models/generative/{version}/report.json').read_text())
        if digest(current/f'models/generative/{version}/candidate.joblib')!=report['artifact_sha256']:raise ValueError('Artifact mismatch')
        selector=dict(generation=version,artifact_sha256=report['artifact_sha256'],trials=1000,status='shadow_only')
        write_json(current/'models/generative/active_shadow.json',selector)
        if digest(current/f'models/generative/{version}/historical_schedule.csv')!=report['source_hashes']['historical_schedule']:
            raise ValueError('Frozen historical feature source mismatch')
        season=report['season'];track(current,season);snapshot(current)
        after=inventory(current)
        for name,sha in state['files'].items():
            if after.get(name)!=sha and not name.startswith('public_site/data/') and name!='models/generative/active_shadow.json':
                raise ValueError('Migration changed existing private state: '+name)
        validate_transition(current,previous)
        files=pack(current,base/'new.tar.gz');new=gh.upload(base/'new.tar.gz')
        new.update(files=files,generation=state['generation']+1,parent_state=parent)
        config['public_exports']=sorted(set(config['public_exports'])|{'public_site/data/generative.json','public_site/data/generative_readiness.json'})
        receipt=dict(generation=version,production_changed=False,completed_at=datetime.now(timezone.utc).isoformat())
        commit=gh.commit(parent,{'state.json':json.dumps(new,indent=2),'config.json':json.dumps(config,indent=2),
            'generative-migration.json':json.dumps(receipt,indent=2)},'Install generative shadow without changing production')
        print(json.dumps(dict(receipt,commit=commit)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','migrate']);p.add_argument('--run',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(Path(__file__).resolve().parents[1],a.run)
    else:migrate()
