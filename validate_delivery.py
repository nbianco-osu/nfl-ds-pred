"""Verify the pinned public data manifest without loading private model artifacts."""
import hashlib
import json
from pathlib import Path
import re


def validate(site):
    site = Path(site)
    manifest = json.loads((site/'data/dashboard_manifest.json').read_text())
    if manifest.get('schema_version') != 2 or not {'predictions.json', 'metrics.json'} <= manifest['files'].keys():
        raise ValueError('Invalid dashboard manifest')
    for name, entry in manifest['files'].items():
        if not re.fullmatch(r'[-a-zA-Z0-9_.]+\.(json|csv)', name):
            raise ValueError('Invalid dataset name')
        expected = 'revisions/' + entry['sha256'] + Path(name).suffix
        if not re.fullmatch(r'revisions/[0-9a-f]{64}\.(json|csv)', entry['path']) or entry['path'] != expected:
            raise ValueError('Invalid revision path')
        payload = (site/'data'/entry['path']).read_bytes()
        if len(payload) != entry['bytes'] or hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise ValueError('Dashboard artifact digest mismatch: ' + name)
        if name.endswith('.json'):
            json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    digest = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    if digest != manifest['revision']:
        raise ValueError('Manifest revision mismatch')
    return {'revision': digest, 'datasets': len(manifest['files']), 'manifest_bytes': (site/'data/dashboard_manifest.json').stat().st_size}


if __name__ == '__main__':
    print(validate(Path(__file__).resolve().parent/'public_site'))
