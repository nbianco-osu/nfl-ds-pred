"""Stage bounded, insert-only SQL requests for the connected Supabase tool."""
import csv
import hashlib
import json
from pathlib import Path


def stage(root):
    folder = root / 'data/database_import'
    source = folder / 'resume_missing_part_01.csv'
    if hashlib.sha256(source.read_bytes()).hexdigest() != '7ad4626c50e39e7f3b39c64716be35cab17eeef112470f29f23e8a7cffc05662':
        raise ValueError('Recovery checkpoint changed; reconcile a fresh export first')
    csv.field_size_limit(10_000_000)
    with source.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1217 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Unexpected recovery row count or duplicates')
    output = folder / 'connector_recovery'
    output.mkdir(exist_ok=True)
    files = []
    for offset in range(0, len(rows), 25):
        batch = rows[offset:offset+25]
        for row in batch:
            row['payload'] = json.loads(row['payload'])
            row['season'] = int(row['season']) if row['season'] else None
            row['game_id'] = row['game_id'] or None
        keys = sorted({key for row in batch for key in row['payload']})
        packed = {'keys': keys, 'rows': [[row[key] for key in ['id','dataset','snapshot_sha256','record_key','season','game_id']] + [[row['payload'].get(key) for key in keys]] for row in batch]}
        literal = "'" + json.dumps(packed, separators=(',', ':'), allow_nan=False).replace("'", "''") + "'::jsonb"
        incoming = f"select r->>0 as id,r->>1 as dataset,r->>2 as snapshot_sha256,r->>3 as record_key,(r->>4)::bigint as season,r->>5 as game_id,(select jsonb_object_agg(k.key,v.value) from jsonb_array_elements_text(j->'keys') with ordinality k(key,n) join jsonb_array_elements(r->6) with ordinality v(value,n) using(n) where v.value <> 'null'::jsonb) as payload from (select {literal} as j) source cross join lateral jsonb_array_elements(j->'rows') r"
        match = 'd.dataset=r.dataset and d.snapshot_sha256=r.snapshot_sha256 and d.record_key=r.record_key and d.season is not distinct from r.season and d.game_id is not distinct from r.game_id and d.payload=r.payload'
        insert = f'with incoming as ({incoming}), added as (insert into public.nfl_dataset_records(id,dataset,snapshot_sha256,record_key,season,game_id,payload) select * from incoming on conflict(id) do nothing returning id) select count(*) as inserted from added;'
        verify = f'with incoming as ({incoming}) select count(*) as expected, count(d.id) as found, count(*) filter(where {match}) as matched from incoming r left join public.nfl_dataset_records d using(id);'
        combined = f'with incoming as ({incoming}), added as (insert into public.nfl_dataset_records(id,dataset,snapshot_sha256,record_key,season,game_id,payload) select * from incoming on conflict(id) do nothing returning *), stored as (select * from added union all select * from public.nfl_dataset_records where id in(select id from incoming)) select count(*) as expected,count(d.id) as found,count(*) filter(where {match}) as matched,(select count(*) from added) as inserted from incoming r left join stored d using(id);'
        target = output / f'batch_{offset//25:03}.json'
        target.write_text(json.dumps({'insert': insert, 'verify': verify, 'combined': combined, 'rows': len(batch)}), encoding='utf-8')
        files.append(target.name)
    (output/'manifest.json').write_text(json.dumps({'files': files, 'rows': len(rows)}), encoding='utf-8')
    print(json.dumps({'batches': len(files), 'rows': len(rows)}))


if __name__ == '__main__':
    stage(Path(__file__).resolve().parent)
