"""Compare dashboard data to Git HEAD, excluding operational timestamp-only churn."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

TIMESTAMPS={'updated_at','source_updated_at','retrieved_at','last_successful_refresh'}


def canonical(value):
    if isinstance(value,dict):
        return {k:canonical(v) for k,v in value.items() if k not in TIMESTAMPS}
    if isinstance(value,list):
        return [canonical(v) for v in value]
    return value


def changes(root,reference='HEAD'):
    git=shutil.which('git') or r'C:\Program Files\Git\cmd\git.exe'
    tracked=subprocess.check_output([git,'ls-tree','-r','--name-only',reference,'--','public_site/data'],cwd=root,text=True).splitlines()
    paths=set(tracked)|{p.relative_to(root).as_posix() for p in (root/'public_site/data').glob('*.json')}
    changed=[]
    for name in sorted(paths):
        # Bundle duplicates the individual datasets; its digest includes receipt times.
        if name in {'public_site/data/live_snapshot.json', 'public_site/data/dashboard_manifest.json'} or name.startswith('public_site/data/revisions/'):
            continue
        if not name.endswith('.json'):
            continue
        path=root/name
        old=subprocess.run([git,'show',f'{reference}:{name}'],cwd=root,capture_output=True,text=True)
        if not path.exists() or old.returncode:
            changed.append(name)
        elif canonical(json.loads(old.stdout))!=canonical(json.loads(path.read_text())):
            changed.append(name)
    return {'publication_needed':bool(changed),'substantive_files':changed,
            'note':'Forecast timestamps and input hashes remain meaningful. Only operational observation/refresh timestamps are ignored. Code changes require separate publication.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--against',default='HEAD')
    args=parser.parse_args()
    print(json.dumps(changes(Path(__file__).resolve().parent,args.against),indent=2))
