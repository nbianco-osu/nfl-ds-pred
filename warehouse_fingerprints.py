"""Read-only manifest fingerprints using JSONB ordering and normalized numbers.

Compare with database/verify_snapshot.sql for the current flat-payload manifest.
"""
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def jsonb_text(value):
    if isinstance(value, dict):
        keys=sorted(value,key=lambda k:(len(k.encode('utf-8')),k.encode('utf-8')))
        return '{'+', '.join(json.dumps(k,ensure_ascii=False)+': '+jsonb_text(value[k]) for k in keys)+'}'
    if isinstance(value,list): return '['+', '.join(map(jsonb_text,value))+']'
    if isinstance(value,Decimal):
        if value.is_zero(): return '0'
        number = format(value, 'f')
        return number.rstrip('0').rstrip('.') if '.' in number else number
    return json.dumps(value,ensure_ascii=False,allow_nan=False)


def fingerprints(folder):
    csv.field_size_limit(10_000_000)
    manifest=json.loads((folder/'manifest.json').read_text())
    groups={};seen=set()
    for part in manifest['files']:
        path=folder/part['name']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=part['sha256']: raise ValueError('Source CSV changed')
        with path.open(newline='',encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if row['id'] in seen: raise ValueError('Duplicate source ID')
                seen.add(row['id'])
                payload=jsonb_text(json.loads(row['payload'],parse_float=Decimal))
                md5=hashlib.md5(payload.encode('utf-8')).hexdigest()
                fields=[row[k] for k in ['id','dataset','snapshot_sha256','record_key','season','game_id']]+[md5]
                if any('|' in v or '\n' in v for v in fields): raise ValueError('Ambiguous fingerprint field')
                groups.setdefault(row['dataset'],[]).append('|'.join(fields))
    return {name:{'rows':len(values),'sha256':hashlib.sha256('\n'.join(sorted(values)).encode('utf-8')).hexdigest()} for name,values in groups.items()}


if __name__=='__main__':
    print(json.dumps(fingerprints(Path(__file__).resolve().parent/'data/database_import'),indent=2))
