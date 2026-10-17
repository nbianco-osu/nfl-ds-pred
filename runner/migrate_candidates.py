"""Explicit, serialized shadow migration; never replaces production artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from audit_io import write_json
from runner.common import GitHub, credential, inventory, pack, unpack, digest, PRIVATE


def prepare(root, generation, ensemble):
    for value in [generation, ensemble]:
        if Path(value).name != value or value in {'.', '..'}: raise ValueError('Generation required')
    parent = root/'models/retrained'/generation
    report = json.loads((parent/'report.json').read_text())
    eroot = root/'models/retrained'/ensemble
    evidence = json.loads((eroot/'report.json').read_text())
    if report['status'] != 'completed_shadow_only' or len(report['models']) != 240:
        raise ValueError('Completed 240-model library required')
    if report['source_hashes'] != evidence['source_hashes'] or evidence['parent_generation'] != generation:
        raise ValueError('Ensemble/library mismatch')
    for model in report['models']:
        if digest(parent/(model['name']+'.joblib')) != model['artifact_sha256']:
            raise ValueError('Candidate hash mismatch')
    if digest(eroot/'candidate.joblib') != evidence['candidate_sha256']:
        raise ValueError('Ensemble hash mismatch')
    destination = root/'data/candidate_migrations'/generation
    destination.mkdir(parents=True, exist_ok=False)
    payload = destination/'payload'
    for version in [generation, ensemble]:
        for prefix in ['models/retrained', 'backtests/retraining']:
            shutil.copytree(root/prefix/version, payload/prefix/version)
    folder = payload/'models/retrained'/generation
    shutil.copyfile(eroot/'candidate.joblib', folder/'temporal_ensemble.joblib')
    report.update(feature_adapter='expanded_differentials_v1', ensemble_kind='robust_r60',
                  ensemble_generation=ensemble, ensemble_sha256=evidence['candidate_sha256'])
    for prefix in ['models/retrained', 'backtests/retraining']:
        write_json(payload/prefix/generation/'report.json', report)
    policy = dict(generation=generation, declared_at=datetime.now(timezone.utc).isoformat(),
        candidate='ensemble_winner', horizon='90m', review_weeks=[10,14,18],
        minimum_games=[50,100,200], minimum_weeks=6, familywise_alpha=.05,
        looks=3, selection='One predeclared winner ensemble versus same-horizon production.',
        automatic_promotion=False, note='Fixed looks; Bonferroni-adjusted week-bootstrap intervals are approximate, not an anytime-valid guarantee.')
    write_json(folder/'review_policy.json', policy)
    files = pack(payload, destination/'runtime.tar.gz')
    gh = GitHub(credential()); gh.assert_private()
    uploaded = gh.upload(destination/'runtime.tar.gz')
    manifest = dict(uploaded, files=files, generation=generation, ensemble=ensemble,
                    approved_scope='Install shadow candidates only; preserve production and all prior cohorts.')
    content = json.dumps(manifest, sort_keys=True).encode()
    release = gh.api('GET', f'/repos/{PRIVATE}/releases/tags/{uploaded["tag"]}')
    response = gh.session.post(release['upload_url'].split('{')[0], params={'name':'migration.json'},
        data=content, headers={'Content-Type':'application/json'}, timeout=120)
    if response.status_code != 201: raise RuntimeError(f'Manifest upload HTTP {response.status_code}')
    receipt = {'manifest_asset_id':response.json()['id'], 'manifest_sha256':hashlib.sha256(content).hexdigest(),
               'generation':generation, 'ensemble':ensemble}
    write_json(destination/'upload.json', receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def allowed_additions(files, generation, ensemble):
    prefixes = [f'{prefix}/{version}/' for prefix in ['models/retrained','backtests/retraining']
                for version in [generation,ensemble]]
    return bool(files) and all(any(name.startswith(prefix) for prefix in prefixes) for name in files)


def migrate():
    from runner.run import guard_live
    from publication_gate import validate_transition
    from shadow_next import export as shadow_export
    from export_retraining import export as retraining_export
    from dashboard_snapshot import export as snapshot_export
    from iteration_r61 import export as iteration_export
    gh = GitHub(os.environ['GH_TOKEN']); gh.assert_private()
    parent = gh.head(); config = json.loads(gh.read('config.json',parent))
    guard_live(config); state = json.loads(gh.read('state.json',parent))
    with tempfile.TemporaryDirectory(prefix='nfl-candidate-migration-') as temp:
        base=Path(temp); current=base/'current'; previous=base/'previous'
        gh.download({'asset_id':int(os.environ['MANIFEST_ASSET_ID']), 'sha256':os.environ['MANIFEST_SHA256']},base/'manifest.json')
        manifest=json.loads((base/'manifest.json').read_text())
        version=manifest['generation']; ensemble=manifest['ensemble']
        for value in [version,ensemble]:
            if Path(value).name != value or value in {'.','..'}:raise ValueError('Unsafe generation')
        if not allowed_additions(manifest['files'],version,ensemble):raise ValueError('Migration scope violation')
        if any(name in state['files'] for name in manifest['files']):raise ValueError('Generation already installed; never overwrite')
        gh.download(state,base/'old.tar.gz')
        unpack(base/'old.tar.gz',current,state['files']);unpack(base/'old.tar.gz',previous,state['files'])
        gh.download(manifest,base/'new.tar.gz');unpack(base/'new.tar.gz',current,manifest['files'])
        report=json.loads((current/'models/retrained'/version/'report.json').read_text())
        for name,value in report['source_hashes'].items():
            if digest(current/'data'/name)!=value:raise ValueError('Training source differs from authoritative cloud snapshot')
        for model in report['models']:
            if digest(current/'models/retrained'/version/(model['name']+'.joblib'))!=model['artifact_sha256']:raise ValueError('Model hash mismatch')
        if digest(current/'models/retrained'/version/'temporal_ensemble.joblib')!=report['ensemble_sha256']:raise ValueError('Ensemble hash mismatch')
        selector=json.loads((current/'models/retrained/active_shadow.json').read_text())
        selector.update(generation=version,status='shadow_only')
        write_json(current/'models/retrained/active_shadow.json',selector)
        season=datetime.now(timezone.utc).year-(datetime.now(timezone.utc).month<=2)
        shadow_export(current,season);retraining_export(current,version)
        iteration_export(current,season,run_research=True)
        snapshot_export(current)
        changed=[]
        after=inventory(current)
        for name,sha in state['files'].items():
            if after.get(name)!=sha:
                changed.append(name)
                if name!='models/retrained/active_shadow.json' and not name.startswith('public_site/data/'):
                    raise ValueError('Migration modified existing private state: '+name)
        validation=validate_transition(current,previous)
        new_files=pack(current,base/'complete.tar.gz');new_state=gh.upload(base/'complete.tar.gz')
        new_state.update(files=new_files,generation=state['generation']+1,parent_state=parent)
        config['shadow_generation']=version
        exports=[name for name in new_files if name.startswith((f'backtests/retraining/{version}/',f'backtests/retraining/{ensemble}/',f'audit/shadow_next/{version}/')) and name.endswith(('.json','.csv'))]
        exports+=['public_site/data/iteration_r61.json']
        config['public_exports']=sorted(set(config['public_exports'])|set(exports))
        receipt=dict(generation=version,ensemble=ensemble,previous_generation=json.loads((previous/'models/retrained/active_shadow.json').read_text())['generation'],
            validation=validation,changed_existing_files=changed,production_changed=False,completed_at=datetime.now(timezone.utc).isoformat())
        commit=gh.commit(parent,{'state.json':json.dumps(new_state,indent=2),'config.json':json.dumps(config,indent=2),
            'candidate-migration.json':json.dumps(receipt,indent=2)},'Install corrected 240-model shadow cohort; preserve all prior state')
        print(json.dumps(dict(receipt,commit=commit)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','migrate'])
    parser.add_argument('--generation');parser.add_argument('--ensemble');args=parser.parse_args()
    if args.action=='prepare':prepare(Path(__file__).resolve().parents[1],args.generation,args.ensemble)
    else:migrate()
