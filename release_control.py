"""R33: immutable model bundle manifests, human approvals and transactional local rollback."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import joblib
from sklearn.utils.validation import check_is_fitted
from audit_io import atomic_bytes,write_json

FILES=['models/home_win_advanced_automl_no_market_deep.joblib',
       'models/home_win_advanced_automl_no_market_deep.metrics.json'] + [
       'models/score_models/'+name for name in ['ExtraTreesRegressor.joblib','HistGradientBoostingRegressor.joblib',
       'RandomForestRegressor.joblib','Ridge.joblib','score_ensemble.joblib','score_ensemble.metrics.json']]


def digest(data):return hashlib.sha256(data).hexdigest()


def snapshot(root,source=None):
    source=Path(source or root)
    files={name:digest((source/name).read_bytes()) for name in FILES}
    version=digest(json.dumps(files,sort_keys=True).encode())[:20]
    folder=root/'releases'/version
    for name,sha in files.items():
        blob=root/'.release_blobs'/sha
        if not blob.exists():atomic_bytes(blob,(source/name).read_bytes())
    manifest={'release_id':version,'files':files,'created_at':datetime.now(timezone.utc).isoformat(),
              'status':'unapproved','note':'Local model bundle only. Data, archived picks and hosted deployments are never rolled back by this tool.'}
    if not (folder/'manifest.json').exists():write_json(folder/'manifest.json',manifest)
    return version


def manifest(root,version):
    if len(version)!=20 or any(c not in '0123456789abcdef' for c in version):raise ValueError('Invalid release ID')
    value=json.loads((root/'releases'/version/'manifest.json').read_text())
    if set(value['files'])!=set(FILES) or digest(json.dumps(value['files'],sort_keys=True).encode())[:20]!=version:
        raise ValueError('Release manifest integrity failure')
    for name,sha in value['files'].items():
        if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('Invalid blob hash')
        if digest((root/'.release_blobs'/sha).read_bytes())!=sha:raise ValueError('Missing/corrupt local release blob')
    return value


def approve(root,version,reviewer,reason):
    manifest(root,version)
    if not reviewer.strip() or not reason.strip():raise ValueError('Named reviewer and reason required')
    path=root/'releases'/version/'approval.json'
    if path.exists():raise ValueError('Approval already recorded; immutable')
    write_json(path,{'release_id':version,'reviewer':reviewer,'reason':reason,
                     'approved_at':datetime.now(timezone.utc).isoformat()})


def plan(root,version):
    value=manifest(root,version)
    current=joblib.load(root/FILES[0])
    candidate=joblib.load(root/'.release_blobs'/value['files'][FILES[0]])
    if candidate.get('feature_cols')!=current.get('feature_cols'):
        raise ValueError('Feature contract changed; integrate explicitly before release')
    if not hasattr(candidate.get('model'),'predict_proba'):raise ValueError('Invalid classifier artifact')
    check_is_fitted(candidate['model'])
    for name in FILES[2:]:
        if not name.endswith('.joblib'):continue
        old=joblib.load(root/name);new=joblib.load(root/'.release_blobs'/value['files'][name])
        if new.get('feature_cols')!=old.get('feature_cols'):raise ValueError('Score feature contract changed')
        models=new.get('models')
        if models is None:
            check_is_fitted(new['model'])
        else:
            if not models:raise ValueError('Empty ensemble')
            for model in (models.values() if isinstance(models,dict) else models):
                check_is_fitted(model)
    approval_path=root/'releases'/version/'approval.json'
    approved=False
    if approval_path.exists():
        approval=json.loads(approval_path.read_text())
        approved=approval.get('release_id')==version and bool(approval.get('reviewer')) and bool(approval.get('reason'))
    return {'release_id':version,'changes':[name for name,sha in value['files'].items() if digest((root/name).read_bytes())!=sha],
            'approved':approved,'production_changed':False}


def activate(root,version):
    proposal=plan(root,version)
    if not proposal['approved']:raise ValueError('Explicit human approval required')
    previous=snapshot(root)
    value=manifest(root,version)
    original={name:(root/name).read_bytes() for name in FILES}
    try:
        for name,sha in value['files'].items():atomic_bytes(root/name,(root/'.release_blobs'/sha).read_bytes())
        write_json(root/'releases/active.json',{'release_id':version,'previous_release_id':previous,
                    'activated_at':datetime.now(timezone.utc).isoformat()})
    except Exception:
        for name,data in original.items():atomic_bytes(root/name,data)
        raise
    return {'release_id':version,'previous_release_id':previous,'production_changed':bool(proposal['changes'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['snapshot','plan','approve','activate','rollback'])
    parser.add_argument('--release');parser.add_argument('--source',type=Path)
    parser.add_argument('--reviewer',default='');parser.add_argument('--reason',default='')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    if args.action=='snapshot':print(snapshot(root,args.source))
    elif args.action=='plan':print(json.dumps(plan(root,args.release),indent=2))
    elif args.action=='approve':approve(root,args.release,args.reviewer,args.reason)
    else:print(json.dumps(activate(root,args.release),indent=2))
