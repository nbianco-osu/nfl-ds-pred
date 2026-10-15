"""Validate in an isolated staging directory before atomically replacing output files."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from audit_io import atomic_bytes, write_json
from validate_dashboard import validate


def refresh(root,season):
    from refresh_lock import refresh_lock
    with refresh_lock(Path(root)):
        _locked_refresh(root,season)


def _locked_refresh(root,season):
    root=Path(root)
    try:
        _refresh(root,season)
    except Exception as exc:
        write_json(root/'audit/refresh_attempt.json',{'status':'failed','error_type':type(exc).__name__,
            'attempted_at':datetime.now(timezone.utc).isoformat()})
        from reliability import export
        try:
            export(root,season)
        except Exception:
            pass
        raise
    write_json(root/'audit/refresh_attempt.json',{'status':'succeeded','attempted_at':datetime.now(timezone.utc).isoformat()})
    from reliability import export
    export(root,season)
    from research_operations import checkpoints
    checkpoints(root,season)


def _refresh(root,season):
    root=Path(root)
    before=json.loads((root/'public_site/data/predictions.json').read_text())
    with tempfile.TemporaryDirectory(prefix='nfl-refresh-') as temporary:
        stage=Path(temporary)
        for folder in ['data','models','predictions','learning','backtests','audit','public_site/data']:
            if (root/folder).exists():
                shutil.copytree(root/folder,stage/folder,ignore=shutil.ignore_patterns('refresh.lock'))
        from refresh_predictions import run
        run(stage,season)
        checked=validate(stage)
        after={r['game_id']:r for r in json.loads((stage/'public_site/data/predictions.json').read_text())}
        preserved=['home_win_probability','away_win_probability','predicted_winner','model_version',
                   'predicted_home_score','predicted_away_score','score_model_version','score_forecast_at']
        for old in before:
            if old['status']=='Final':
                for key in preserved:
                    if old.get(key)!=after[old['game_id']].get(key):
                        raise ValueError(f'Completed forecast changed: {old["game_id"]} {key}')
        # Files are atomic individually; publish only after the completion manifest is written.
        changes=[]
        for folder in ['data','predictions','learning','audit','public_site/data']:
            for source in (stage/folder).rglob('*'):
                if source.is_file():
                    destination=root/source.relative_to(stage)
                    content=source.read_bytes()
                    if not destination.exists() or destination.read_bytes()!=content:
                        changes.append((destination,content,destination.read_bytes() if destination.exists() else None))
        applied=[]
        try:
            for destination,content,old in changes:
                atomic_bytes(destination,content)
                applied.append((destination,old))
            write_json(root/'public_site/data/refresh_health.json',{
                'last_successful_refresh':datetime.now(timezone.utc).isoformat(),
                'validation':'passed','season':season,**checked,
                'note':'Validated staged refresh; individual files replaced atomically. Production publishing uses a single built archive.'})
        except Exception:
            for destination,old in reversed(applied):
                if old is None:
                    destination.unlink(missing_ok=True)
                else:
                    atomic_bytes(destination,old)
            raise
