"""Private, idempotent warehouse sync. Credentials only from the environment."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from prepare_database import records


def sync(root):
    url=os.environ.get('SUPABASE_URL','').rstrip('/')
    key=os.environ.get('SUPABASE_SECRET_KEY','')
    parsed=urlparse(url)
    if parsed.scheme!='https' or not parsed.hostname or not parsed.hostname.endswith('.supabase.co') or parsed.path or not key:
        raise ValueError('Set SUPABASE_URL and SUPABASE_SECRET_KEY privately; no secrets in arguments or browser code.')
    headers={'apikey':key,'Content-Type':'application/json','Prefer':'resolution=ignore-duplicates,return=representation'}
    if not key.startswith('sb_secret_'): headers['Authorization']='Bearer '+key
    values=list(records(root));verified=0
    for start in range(0,len(values),50):
        batch=values[start:start+50]
        endpoint=url+'/rest/v1/nfl_dataset_records'
        with urlopen(Request(endpoint+'?on_conflict=id',data=json.dumps(batch,allow_nan=False).encode(),headers=headers,method='POST'),timeout=120) as response:
            response.read()
        ids=','.join(r['id'] for r in batch)
        query=urlencode({'id':f'in.({ids})','select':'id,snapshot_sha256,record_key'})
        with urlopen(Request(endpoint+'?'+query,headers={'apikey':key,**({'Authorization':'Bearer '+key} if not key.startswith('sb_secret_') else {})}),timeout=120) as response:
            received={r['id']:r for r in json.load(response)}
        if any(received.get(r['id'],{}).get('snapshot_sha256')!=r['snapshot_sha256'] or received.get(r['id'],{}).get('record_key')!=r['record_key'] for r in batch):
            raise ValueError('Read-back verification failed; existing records preserved.')
        verified+=len(batch)
        print(f'Verified {verified}/{len(values)} immutable records',flush=True)
    return verified


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--push',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent
    if not args.push:
        print(f'Dry run: {sum(1 for _ in records(root))} records. Use --push after applying database/schema.sql.')
    else:
        try: sync(root)
        except HTTPError as error: raise SystemExit(f'Supabase HTTP {error.code}; import stopped. No response body logged to protect credentials.') from None
