"""Prepare idempotent private warehouse snapshots and browser-importable CSV."""
import csv
import hashlib
import json
from pathlib import Path
import pandas as pd
from audit_io import write_json


def records(root):
    sources=[('research_matchups','data/research_matchups.csv','game_id'),
        ('team_games','data/research_team_games.csv',None),
        ('teams','data/team_metadata.csv','team_abbr'),
        ('schedule_2026','data/schedule_2026.csv','game_id'),
        ('forecast_current','public_site/data/predictions.json','game_id'),
        ('source_receipts','data/research_sources/manifest.json',None)]
    for dataset,relative,key in sources:
        path=root/relative
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        values=json.loads(pd.read_csv(path).to_json(orient='records')) if path.suffix=='.csv' else json.loads(path.read_text())
        for index,payload in enumerate(values):
            record_key=str(payload[key]) if key else f"{payload.get('game_id','')}_{payload.get('team','')}_{index}"
            identity=hashlib.sha256(f'{dataset}:{digest}:{record_key}'.encode()).hexdigest()
            yield {'id':identity,'dataset':dataset,'snapshot_sha256':digest,'record_key':record_key,
                   'season':payload.get('season'),'game_id':payload.get('game_id'),'payload':{k:v for k,v in payload.items() if v is not None}}
    report=json.loads((root/'models/research32_report.json').read_text())
    for model in report['models']:
        digest=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()
        yield {'id':hashlib.sha256((model['name']+digest).encode()).hexdigest(),'dataset':'model_research32',
               'snapshot_sha256':digest,'record_key':model['name'],'season':2026,'game_id':None,'payload':model}


def prepare(root):
    folder=root/'data/database_import';folder.mkdir(parents=True,exist_ok=True)
    counts={};parts=[];stream=None;writer=None;size=0
    try:
        for row in records(root):
            counts[row['dataset']]=counts.get(row['dataset'],0)+1
            row['payload']=json.dumps(row['payload'],separators=(',',':'),allow_nan=False)
            if stream is None or size>70_000_000:
                if stream: stream.close()
                path=folder/f'part_{len(parts)+1:02}.csv';parts.append(path)
                stream=path.open('w',encoding='utf-8',newline='');writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();size=0
            writer.writerow(row);size+=len(row['payload'])
    finally:
        if stream: stream.close()
    manifest={'counts':counts,'total_rows':sum(counts.values()),'files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in parts],
        'note':'Private curated snapshots only; raw FTN/snap rows remain local. FTN-derived fields attributed to FTN Data via nflverse under CC-BY-SA 4.0. Dashboard CSV import is initial-load only; repeatable sync uses immutable IDs.'}
    write_json(folder/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': prepare(Path(__file__).resolve().parent)
