"""Download documented nflverse sources with receipts; no web-page scraping."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

from audit_io import write_json, write_csv


def collect(root):
    folder=root/'data/research_sources'
    folder.mkdir(parents=True,exist_ok=True)
    receipts=[]
    for year in range(2022,2027):
        for name, loader in [('ftn',nfl.load_ftn_charting),('snap_counts',nfl.load_snap_counts)]:
            path=folder/f'{name}_{year}.parquet'
            try:
                if not path.exists():
                    frame=loader([year])
                    frame.write_parquet(path)
                frame=pd.read_parquet(path)
                receipts.append({'source':name,'season':year,'rows':len(frame),
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'observed_at':datetime.now(timezone.utc).isoformat(),
                    'attribution':'FTN Data via nflverse (CC-BY-SA 4.0)' if name=='ftn' else 'Pro Football Reference via nflverse; redistribution terms require review',
                    'url':f'https://nflreadr.nflverse.com/reference/load_{"ftn_charting" if name=="ftn" else "snap_counts"}.html',
                    'columns':list(frame.columns)})
                print(f'{name} {year}: {len(frame)} rows; columns {list(frame.columns)}',flush=True)
            except Exception as error:
                receipts.append({'source':name,'season':year,'error':type(error).__name__,'rows':0})
                print(f'{name} {year}: unavailable ({type(error).__name__})',flush=True)
    write_json(folder/'manifest.json',receipts)
    return receipts


if __name__=='__main__':
    collect(Path(__file__).resolve().parent)
