"""Stage bounded, verified incremental snapshots for the connected SQL tool.

No credentials, remote writes, or acknowledgement inferred from local staging.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd
from audit_io import write_json


def snapshots(root,season):
    sources = [('forecast_current',root/'public_site/data/predictions.json','game_id'),
               (f'schedule_{season}',root/f'data/schedule_{season}.csv','game_id'),
               ('teams',root/'data/team_metadata.csv','team_abbr')]
    for dataset,path,key in sources:
        rows = json.loads(pd.read_csv(path).to_json(orient='records')) if path.suffix=='.csv' else json.loads(path.read_text())
        for value in rows:
            # Operational clock changes must not generate warehouse revisions.
            payload = {k:v for k,v in value.items() if v is not None and k!='updated_at'}
            digest = hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
            record_key = str(payload[key])
            yield {'id':hashlib.sha256(f'{dataset}:{digest}:{record_key}'.encode()).hexdigest(),
                   'dataset':dataset,'snapshot_sha256':digest,'record_key':record_key,
                   'season':payload.get('season'),'game_id':payload.get('game_id'),'payload':payload}


def query(batch):
    literal = "'"+json.dumps(batch,separators=(',',':'),allow_nan=False).replace("'","''")+"'::jsonb"
    # Reuse an identical earlier snapshot, including the original file-hash IDs.
    return f"""with incoming as (
      select * from jsonb_to_recordset({literal}) as x(id text,dataset text,snapshot_sha256 text,record_key text,season bigint,game_id text,payload jsonb)
    ), added as (
      insert into public.nfl_dataset_records(id,dataset,snapshot_sha256,record_key,season,game_id,payload)
      select i.* from incoming i where not exists (
        select 1 from public.nfl_dataset_records d where d.dataset=i.dataset and d.record_key=i.record_key
          and d.season is not distinct from i.season and d.game_id is not distinct from i.game_id
          and (d.payload-'updated_at')=i.payload)
      on conflict(id) do nothing returning *
    ), stored as (select * from added union all select * from public.nfl_dataset_records)
    select (select count(*) from incoming) as expected,(select count(*) from added) as inserted,
      count(*) filter(where exists(select 1 from stored d where d.dataset=i.dataset and d.record_key=i.record_key
        and d.season is not distinct from i.season and d.game_id is not distinct from i.game_id
        and (d.payload-'updated_at')=i.payload)) as matched from incoming i;"""


def stage(root,season):
    rows = list(snapshots(root,season))
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate queue identities')
    digest = hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    folder = root/'data/warehouse_queue'/digest[:16]
    folder.mkdir(parents=True,exist_ok=True)
    files = []
    for start in range(0,len(rows),10):
        name = f'batch_{start//10:03}.json'
        write_json(folder/name,{'query':query(rows[start:start+10]),'rows':len(rows[start:start+10])})
        files.append(name)
    manifest = {'status':'pending_remote_verification','project_id':'unzfybzuynezjhcjyptu',
                'sha256':digest,'season':season,'rows':len(rows),'files':files,
                'folder':str(folder.relative_to(root)),'note':'Connected SQL tool required. Verify matched=expected in every batch before recording success. Never expose credentials. Insert-only; old snapshots preserved.'}
    write_json(root/'data/warehouse_queue/latest.json',manifest)
    return manifest


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--season',type=int,default=2026)
    print(json.dumps(stage(Path(__file__).resolve().parent,parser.parse_args().season),indent=2))
