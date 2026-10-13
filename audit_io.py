"""Atomic exports and content-addressed source receipts. Retrieval is not publication."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False, default=str)+'\n').encode())


def write_csv(path, frame):
    atomic_bytes(path, frame.to_csv(index=False).encode())


def receipt(root, name, frame):
    from io import BytesIO
    buffer = BytesIO()
    frame.write_parquet(buffer)
    content = buffer.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    folder = Path(root)/'audit/sources'/name
    path = folder/(digest+'.parquet')
    if not path.exists():
        atomic_bytes(path, content)
    meta = {'source': 'nflreadpy.'+name, 'sha256': digest, 'rows': frame.height,
            'retrieved_at': datetime.now(timezone.utc).isoformat(),
            'provider_published_at': None, 'cache_may_be_used': True,
            'note': 'Local observation time, not proof of original publication or freshness.'}
    write_json(folder/'latest.json', meta)
    return meta
