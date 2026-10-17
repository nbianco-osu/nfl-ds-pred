"""One consistent public data revision for the hosted dashboard's cloud reader."""
import hashlib
import json
from pathlib import Path
from audit_io import write_json, atomic_bytes


def export(root):
    folder = Path(root)/'public_site/data'
    files = {p.name: p.read_text(encoding='utf-8') for p in sorted(folder.iterdir())
             if p.is_file() and p.suffix in ('.json', '.csv') and p.name not in {'live_snapshot.json', 'dashboard_manifest.json'}}
    for name, content in files.items():
        if name.endswith('.json'):
            json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    report = {'schema_version': 1, 'content_sha256': digest, 'files': files}
    write_json(folder/'live_snapshot.json', report)
    entries = {}
    for name, content in files.items():
        payload = (json.dumps(json.loads(content), ensure_ascii=True, separators=(',', ':'), allow_nan=False)
                   if name.endswith('.json') else content).encode('utf-8')
        sha = hashlib.sha256(payload).hexdigest()
        relative = 'revisions/' + sha + Path(name).suffix
        target = folder/relative
        if target.exists() and target.read_bytes() != payload:
            raise ValueError('Immutable dashboard revision collision')
        if not target.exists():
            atomic_bytes(target, payload)
        entries[name] = {'path': relative, 'sha256': sha, 'bytes': len(payload)}
    revision = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    write_json(folder/'dashboard_manifest.json', {'schema_version': 2, 'revision': revision, 'files': entries})
    return digest


if __name__ == '__main__':
    print(export(Path(__file__).resolve().parent))
