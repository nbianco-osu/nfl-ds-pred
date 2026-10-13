"""Produce version-isolated review evidence; record human decisions, never promote."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_io import write_json


def report(root):
    cohorts=[]
    for folder in sorted((root/'learning/generations').glob('*')):
        if not (folder/'summary.json').exists():
            continue
        summary=json.loads((folder/'summary.json').read_text())
        data=pd.read_csv(folder/'evaluation.csv')
        if data.game_id.duplicated().any():
            raise ValueError('Repeated challenger evaluation labels')
        interval=None
        if len(data):
            p=np.clip(data[['candidate','baseline']].to_numpy(),1e-12,1-1e-12)
            y=data.label.to_numpy()[:,None]
            loss=-(y*np.log(p)+(1-y)*np.log(1-p))
            delta=loss[:,0]-loss[:,1]
            rng=np.random.default_rng(42)
            interval=np.quantile(delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(axis=1),[.025,.975]).tolist()
        ready=len(data)>=64 and interval is not None and interval[1]<0 and summary.get('candidate_brier',1)>=0 and summary.get('candidate_brier',1)<summary.get('baseline_brier',0)
        cohorts.append({'model_version':folder.name,'evaluated_games':len(data),'minimum_games':64,
            'log_loss_difference_ci95':interval,'ready_for_review':ready,'summary':summary,
            'decision':json.loads((folder/'review.json').read_text()) if (folder/'review.json').exists() else None})
    result={'cohorts':cohorts,'note':'Shadow calibration only. Negative candidate-minus-baseline log loss favors the candidate. Bootstrap intervals assume independent games; small samples and selection can be misleading. At least 64 prospective games plus favorable log-loss interval and Brier score are required for review, not automatic promotion. Versions are never pooled.'}
    write_json(root/'public_site/data/challenger_review.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version')
    parser.add_argument('--decision',choices=['hold','reject','approve'])
    parser.add_argument('--reviewer')
    parser.add_argument('--reason')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    result=report(root)
    if args.decision:
        if not args.reviewer or not args.reason:
            parser.error('A named human reviewer and reason are required')
        cohort=next((c for c in result['cohorts'] if c['model_version']==args.version),None)
        if not cohort or (args.decision=='approve' and not cohort['ready_for_review']):
            parser.error('Unknown cohort or insufficient evidence for approval')
        record={'decision':args.decision,'reviewer':args.reviewer,'reason':args.reason,
            'recorded_at':datetime.now(timezone.utc).isoformat(),'evaluated_games':cohort['evaluated_games'],
            'production_changed':False}
        folder=root/'learning/generations'/args.version
        write_json(folder/'review.json',record)
        write_json(folder/'reviews'/(record['recorded_at'].replace(':','-')+'.json'),record)
        result=report(root)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
