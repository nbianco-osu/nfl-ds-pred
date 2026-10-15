"""One consistent public data revision for the hosted dashboard's cloud reader."""
import hashlib
import json
from pathlib import Path
from audit_io import write_json


def export(root):
    folder = Path(root)/'public_site/data'
    files = {p.name: p.read_text(encoding='utf-8') for p in sorted(folder.iterdir())
             if p.is_file() and p.suffix in ('.json', '.csv') and p.name != 'live_snapshot.json'}
    for name, content in files.items():
        if name.endswith('.json'):
            json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    report = {'schema_version': 1, 'content_sha256': digest, 'files': files}
    write_json(folder/'live_snapshot.json', report)
    return digest


if __name__ == '__main__':
    print(export(Path(__file__).resolve().parent))
