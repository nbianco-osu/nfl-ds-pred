"""Publish completed candidate evidence without modifying serving forecasts."""
import argparse
import json
from pathlib import Path

from audit_io import atomic_bytes, write_json


def export(root, version):
    if Path(version).name != version or version in {'.', '..'}:
        raise ValueError('Expected a generation name')
    folder = root / 'backtests/retraining' / version
    report = json.loads((folder / 'report.json').read_text())
    if report['status'] != 'completed_shadow_only' or report['production_changed']:
        raise ValueError('Only completed shadow evidence is supported')
    models = report['models']
    if len(models) not in {60,120,240} or len({m['name'] for m in models}) != len(models):
        raise ValueError('Incomplete or duplicate models')
    report['validation_selected'] = {
        'winner': min((m for m in models if m['task'] == 'winner'), key=lambda m:m['validation_log_loss'])['name'],
        'score': min((m for m in models if m['task'] == 'score'), key=lambda m:m['validation_team_mae'])['name']}
    write_json(root / 'public_site/data/retraining.json', report)
    atomic_bytes(root / 'public_site/data/retraining_holdout.csv', (folder / 'holdout_predictions.csv').read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    export(Path(__file__).resolve().parent, parser.parse_args().version)
