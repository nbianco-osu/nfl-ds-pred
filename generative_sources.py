"""Fetch immutable research inputs without changing the production source pointers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nflreadpy as nfl


def collect(root, season):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name, loader in [('pbp', nfl.load_pbp), ('schedule', nfl.load_schedules)]:
        frame = loader(season).to_pandas()
        path = root/(name+'.parquet')
        frame.to_parquet(path, index=False)
        sources[name] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             rows=len(frame), observed_at=datetime.now(timezone.utc).isoformat())
    meta = dict(season=season, sources=sources, provider='nflreadpy / nflverse',
                note='Retrieval observation, not original publication time; revised historical source. Provider cache may apply.')
    (root/'manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--season', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(collect(a.output, a.season)))
