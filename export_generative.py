"""Publish research evidence only; does not capture or rewrite forecasts."""
import argparse
import json
from pathlib import Path
from audit_io import write_json
from generative_evaluation import evidence


def export(root, run):
    report=json.loads((Path(run)/'report.json').read_text())
    report['comparison_evidence']=evidence(report)
    write_json(Path(root)/'public_site/data/generative.json',report)
    target=Path(root)/'public_site/data/generative_readiness.json'
    if not target.exists() or json.loads(target.read_text()).get('model_version')!=report['model_version']:
        write_json(target,dict(model_version=report['model_version'],matched_games=0,ready_for_review=False,
            automatic_promotion=False,note='No prospective results captured for this generation yet. Research replay is not counted.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    export(Path(__file__).parent,a.run)
