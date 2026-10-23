"""Fresh-machine rehearsal, due-window refresh, CAS persistence, restricted publishing."""
import argparse
import hashlib
from datetime import datetime, timedelta, timezone
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from zoneinfo import ZoneInfo

from runner.common import GitHub, PRIVATE, PUBLIC, inventory, pack, unpack


@contextmanager
def portable_model_pointer(root):
    """Normalize the legacy Windows pointer only inside the disposable runtime."""
    pointer = Path(root)/'models/research/live_challengers.json'
    original = pointer.read_bytes() if pointer.exists() else None
    try:
        if original is not None:
            config = json.loads(original)
            artifact = config['artifact'].replace('\\', '/')
            if not re.fullmatch(r'models/research/[A-Za-z0-9_-]+/models\.joblib', artifact):
                raise ValueError('Unexpected challenger artifact path')
            path = Path(root)/artifact
            if not path.is_file() or not path.resolve().is_relative_to(Path(root).resolve()):
                raise ValueError('Missing or unsafe challenger artifact')
            config['artifact'] = artifact
            pointer.write_text(json.dumps(config), encoding='utf-8')
        yield
    finally:
        if original is not None:
            pointer.write_bytes(original)


def production_input_exports(root, previous=None):
    """Only content-addressed production snapshots, never arbitrary audit sources."""
    result = []
    required = {'game_id', 'model_version', 'features', 'probability', 'source_hashes',
                'receipts', 'forecast_at', 'input_sha256'}
    for path in sorted((Path(root)/'audit/forecasts').glob('*/*.json')):
        if not re.fullmatch(r'\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}', path.parent.name):
            continue
        if not re.fullmatch(r'[0-9a-f]{64}\.json', path.name):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError('Unsafe production input snapshot')
        name = path.relative_to(root).as_posix()
        baseline = Path(previous)/name if previous is not None else None
        if baseline is not None and baseline.is_file():
            if path.read_bytes() != baseline.read_bytes():
                raise ValueError('Existing public production snapshot changed')
            # Historical snapshots are grandfathered only when byte-identical.
            result.append(name)
            continue
        record = json.loads(path.read_text(encoding='utf-8'))
        if not required <= set(record) or set(record) - required - {'model_sha256'}:
            raise ValueError('Unexpected production snapshot schema')
        if record['game_id'] != path.parent.name or record['input_sha256'] != path.stem:
            raise ValueError('Production snapshot identity mismatch')
        features = record['features']
        categorical = {'game_type', 'home_team', 'away_team', 'surface', 'roof'}
        if not isinstance(features, dict) or not features:
            raise ValueError('Invalid production inputs')
        for key, value in features.items():
            if value is None:
                continue
            if key in categorical and isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9 _/-]{1,40}', value):
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Invalid production input feature')
        payload = {k: record[k] for k in ('game_id', 'model_version', 'features', 'probability', 'source_hashes')}
        if 'model_sha256' in record:
            payload['model_sha256'] = record['model_sha256']
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if digest != path.stem:
            raise ValueError('Production snapshot content hash mismatch')
        result.append(path.relative_to(root).as_posix())
    return result


def retained(before, after):
    if isinstance(before, dict):
        return isinstance(after, dict) and all(k in after and retained(v, after[k]) for k, v in before.items())
    if isinstance(before, list):
        return isinstance(after, list) and all(value in after for value in before)
    return before == after


@contextmanager
def no_training():
    """The legacy refresh refits a shadow calibrator; keep that learner frozen."""
    import refresh_predictions
    original = refresh_predictions.update_learning

    def frozen(rows, root, generation=None):
        folder = Path(root)/'learning'
        if generation:
            if not generation.replace('T', '').replace('Z', '').isdigit():
                raise ValueError('Invalid learning generation')
            folder = folder/'generations'/generation
        path = folder/'summary.json'
        summary = json.loads(path.read_text()) if path.exists() else {'training_games': 0}
        return dict(summary, status='Frozen shadow learner: R57 never retrains')

    refresh_predictions.update_learning = frozen
    try:
        yield
    finally:
        refresh_predictions.update_learning = original


def preserve(previous, current):
    """Never lose private files, immutable model bytes, or prior checkpoint captures."""
    old, new = inventory(previous), inventory(current)
    if not set(old) <= set(new):
        raise ValueError('Private state files removed')
    for name in old:
        if (name.startswith('models/') or name.endswith('.joblib')) and old[name] != new[name]:
            raise ValueError('No retraining or model replacement allowed')
        if name.startswith(('audit/forecasts/', 'audit/shadow_next/')) and old[name] != new[name]:
            if not name.endswith('.json') or not retained(json.loads((Path(previous)/name).read_text()),
                                                         json.loads((Path(current)/name).read_text())):
                raise ValueError('Immutable audit snapshot changed')
    if any(name.endswith('.joblib') for name in set(new) - set(old)):
        raise ValueError('New fitted artifacts forbidden')
    path = 'audit/checkpoints.json'
    before = json.loads((Path(previous)/path).read_text()) if (Path(previous)/path).exists() else {}
    after = json.loads((Path(current)/path).read_text()) if (Path(current)/path).exists() else {}
    if any(after.get(k) != v for k, v in before.items()):
        raise ValueError('Checkpoint regression')


def schedule_plan(rows, now):
    """Full-season weekly UTC union, including DST shifts of known kickoffs."""
    from pregame_inputs import kickoff
    slots = set()
    windows = []
    for row in rows:
        if row.get('game_type') != 'REG' or not isinstance(row.get('gametime'), str):
            continue
        start = kickoff(row).to_pydatetime()
        for label, minutes, offsets in [('24h', 1440, (-20, 0, 20)), ('90m', 90, (-10, 0, 10))]:
            target = start - timedelta(minutes=minutes)
            for offset in offsets:
                check = target + timedelta(minutes=offset)
                slots.add(((check.weekday()+1) % 7, check.hour, check.minute))
            windows.append({'game_id': row['game_id'], 'horizon': label, 'target': target.isoformat()})
    groups = {}
    for day, hour, minute in slots:
        groups.setdefault((day, hour), set()).add(minute)
    crons = [f'{",".join(map(str, sorted(minutes)))} {hour} * * {day}'
             for (day, hour), minutes in sorted(groups.items())]
    crons += ['17 11 * * *', '0 13,14 * * 2']
    return {'generated_at': now.isoformat(), 'cron_proposal': crons, 'windows': windows,
            'weekly_checks_upper_bound': len(slots)+9,
            'monthly_checks_estimate': round((len(slots)+9)*365.25/12/7),
            'automatic_enabled': False, 'note': 'Static full-season union; daily flex drift proposals only. Tuesday 09:00 America/New_York forced refresh.'}


def weekly_due(now):
    local = now.astimezone(ZoneInfo('America/New_York'))
    return local.weekday() == 1 and local.hour == 9


def guard_live(config):
    if not all(config.get(k) is True for k in ('automatic_enabled', 'parent_approved', 'dry_run_verified')):
        raise ValueError('Live run blocked pending fresh dry-run verification and parent switch approval')


def export_cloud_schedule(root, config):
    """Describe cloud authority only for an already-gated live execution."""
    guard_live(config)
    from audit_io import write_json
    write_json(Path(root)/'public_site/data/checkpoint_schedule.json', {
        'status': 'ACTIVE', 'authority': 'private GitHub Actions',
        'repository': PRIVATE, 'workflow': 'runner.yml',
        'schedule': 'Static full-season bounded kickoff triggers plus daily schedule drift checks',
        'weekly_refresh': 'Tuesday 09:00 America/New_York (DST-aware)',
        'shadow_calibrator': 'frozen',
        'note': 'Cloud authority: bounded checks around known 24-hour and 90-minute pregame windows, '
                'daily future-schedule drift checks, and Tuesday 9 a.m. Eastern results refresh. '
                'GitHub schedule latency or dropped runs and refresh duration can cause missed windows; '
                'captures are never backfilled. The legacy shadow calibrator is frozen; no cloud retraining.',
    })


def valid_public_export(name):
    research = re.fullmatch(r'backtests/retraining/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+\.(?:csv|json)', name)
    return bool((research or any(name.startswith(p + '/') for p in ('data', 'audit', 'learning', 'public_site/data')))
                and Path(name).suffix in {'.csv', '.json'} and '..' not in Path(name).parts
                and '\\' not in name and ':' not in name)


def publish(root, destination, config):
    from publication_gate import validate_transition
    from publication_changes import changes
    validate_transition(root, destination)
    exports = list(config['public_exports'])
    manifest_path = root/'public_site/data/dashboard_manifest.json'
    if manifest_path.exists():
        from validate_delivery import validate as validate_delivery
        validate_delivery(root/'public_site')
        manifest = json.loads(manifest_path.read_text())
        exports += ['public_site/data/dashboard_manifest.json']
        exports += ['public_site/data/' + entry['path'] for entry in manifest['files'].values()]
    exports += production_input_exports(root, destination)
    generation = config.get('shadow_generation')
    if generation and re.fullmatch(r'[A-Za-z0-9_-]+', generation):
        folder = root/'audit/shadow_next'/generation/'inputs'
        exports += [p.relative_to(root).as_posix() for p in folder.glob('*.json')
                    if re.fullmatch(r'[0-9a-f]{64}\.json', p.name)]
    exports = sorted(set(exports))
    for name in exports:
        if not valid_public_export(name):
            raise ValueError('Invalid public export allowlist')
        source, target = root/name, destination/name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    substantive = changes(destination)['publication_needed']
    if not substantive:
        print('No semantic public changes; timestamp-only snapshot not published.')
        return
    if not (destination/'public_site/data/live_snapshot.json').is_file():
        raise ValueError('Single public dashboard snapshot missing')
    existing = [name for name in exports if (destination/name).is_file()]
    subprocess.run(['git', 'add', '--', *existing], cwd=destination, check=True)
    changed = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=destination).decode().splitlines()
    if not set(changed) <= set(exports):
        raise ValueError('Unapproved staged public paths')
    if changed:
        subprocess.run(['git', '-c', 'user.name=Private NFL runner', '-c', 'user.email=runner@users.noreply.github.com',
                        'commit', '-m', 'Publish validated forecast exports'], cwd=destination, check=True)
        subprocess.run(['git', 'push', 'origin', 'HEAD:main'], cwd=destination, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['duecheck', 'dry-run', 'run', 'force-run', 'schedule-check', 'publish'], default='duecheck')
    parser.add_argument('--public-root', type=Path)
    args = parser.parse_args()
    if args.mode == 'force-run' and os.environ.get('SCHEDULE_CRON') == '0 13,14 * * 2':
        if not weekly_due(datetime.now(timezone.utc)):
            print('Weekly DST guard: not Tuesday 09:00 America/New_York; skipped.')
            return
    gh = GitHub(os.environ['GH_TOKEN'])
    gh.assert_private()
    parent = gh.head()
    config = json.loads(gh.read('config.json', parent))
    state = json.loads(gh.read('state.json', parent))
    if config['source_repo'] != PUBLIC or config['source_ref'] != 'main':
        raise ValueError('Unexpected source checkout configuration')
    if args.mode in ('run', 'force-run', 'publish'):
        guard_live(config)
    source = Path(__file__).resolve().parents[1]
    now = datetime.now(timezone.utc)
    season = now.year - (now.month <= 2)
    with tempfile.TemporaryDirectory(prefix='r57-run-') as directory:
        base = Path(directory)
        archive, root, previous = base/'runtime.tar.gz', base/'work', base/'previous'
        gh.download(state, archive)
        unpack(archive, root, state['files'])
        unpack(archive, previous, state['files'])
        if inventory(root) != state['files']:
            raise ValueError('Restored inventory mismatch')
        from pregame_refresh import due
        if args.mode == 'duecheck':
            from publication_gate import validate_transition
            validation = validate_transition(root, previous)
            print(json.dumps({'duecheck': 'passed', 'due': due(root, season, now),
                              'validation': validation,
                              'restored_files': len(state['files']), 'state_commit': parent,
                              'bundle_sha256': state['sha256'], 'refreshed': False, 'published': False}))
            return
        from publication_gate import validate_transition
        validate_transition(root, previous)
        if args.mode == 'publish':
            if args.public_root is None:
                raise ValueError('Public checkout required')
            publish(root, args.public_root.resolve(), config)
            return
        import pandas as pd
        from staged_refresh import refresh
        # Lightweight future schedule discovery precedes the expensive staged refresh.
        import nflreadpy as nfl
        schedule = nfl.load_schedules([season]).to_pandas()
        if schedule.empty or schedule.game_id.duplicated().any():
            raise ValueError('Invalid fresh schedule')
        schedule.to_csv(root/f'data/schedule_{season}.csv', index=False)
        plan = schedule_plan(schedule.to_dict('records'), now)
        windows = due(root, season, now)
        print(json.dumps({'mode': args.mode, 'due': windows, 'cron_proposal': plan['cron_proposal']}))
        if args.mode == 'schedule-check':
            gh.commit(parent, {'schedule-plan.json': json.dumps(plan, indent=2)}, 'Review future kickoff schedule')
            return
        prior_schedule = pd.read_csv(previous/f'data/schedule_{season}.csv')
        score_columns = ['game_id', 'home_score', 'away_score']
        results_changed = not schedule[score_columns].sort_values('game_id').reset_index(drop=True).equals(
            prior_schedule[score_columns].sort_values('game_id').reset_index(drop=True))
        if args.mode == 'run' and not windows and not results_changed:
            if os.environ.get('SCHEDULE_CRON') == '17 11 * * *':
                installed = set(config.get('schedule_crons', []))
                plan['uncovered_crons'] = sorted(set(plan['cron_proposal']) - installed)
                gh.commit(parent, {'schedule-plan.json': json.dumps(plan, indent=2)}, 'Daily future schedule drift check')
                if plan['uncovered_crons']:
                    print('::warning::Known kickoff coverage changed; parent must review schedule-plan.json')
            print('No due window or changed results; no bundle upload or public publication required.')
            return
        if args.mode in ('dry-run', 'force-run') or windows or results_changed:
            with no_training(), portable_model_pointer(root):
                refresh(root, season)
        from research_operations import checkpoints
        checkpoints(root, season)
        preserve(previous, root)
        from forecast_scorecard import export as export_scorecard
        from iteration_evidence import export as export_iterations
        export_scorecard(root)
        export_iterations(root, season, run_research=False)
        from iteration_r61 import export as export_r61
        export_r61(root, season, run_research=False)
        from generative_tracking import export as export_generative
        export_generative(root, season)
        checked = validate_transition(root, previous)
        if args.mode in ('run', 'force-run'):
            export_cloud_schedule(root, config)
        from dashboard_snapshot import export as export_snapshot
        export_snapshot(root)
        if args.mode == 'dry-run':
            # A rehearsal intentionally cannot approve itself or advance private/public state.
            print(json.dumps({'dry_run': 'passed', 'source_commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=source).decode().strip(),
                'state_commit': parent, 'validation': checked, 'published': False}))
            return
        manifest = pack(root, archive)
        next_state = gh.upload(archive)
        next_state.update({'files': manifest, 'generation': state['generation'] + 1,
                           'parent_state': parent, 'source_commit': subprocess.check_output(
                               ['git', 'rev-parse', 'HEAD'], cwd=source).decode().strip()})
        gh.commit(parent, {'state.json': json.dumps(next_state, indent=2),
                           'schedule-plan.json': json.dumps(plan, indent=2)}, 'Persist validated runner state')


if __name__ == '__main__':
    main()
