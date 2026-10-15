"""Explicit private bootstrap; OAuth stays in memory and never becomes an Actions secret."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import tempfile

from runner.common import GitHub, PRIVATE, PUBLIC, credential, inventory, pack, unpack


def ensure_repo(gh):
    response = gh.session.get('https://api.github.com/repos/' + PRIVATE, timeout=60)
    if response.status_code == 404:
        gh.api('POST', '/user/repos', json={'name': PRIVATE.split('/')[1], 'private': True,
                                          'auto_init': True, 'description': 'Private NFL runtime state; manual activation only'})
    elif response.status_code != 200:
        raise RuntimeError(f'Repository lookup HTTP {response.status_code}')
    gh.assert_private()


def setup_key(gh):
    """Repairable installation: existing key+secret is a no-op, partial setup stops."""
    title = 'R57 private runner public exports'
    keys = gh.api('GET', f'/repos/{PUBLIC}/keys')
    secrets = gh.api('GET', f'/repos/{PRIVATE}/actions/secrets')['secrets']
    key = next((k for k in keys if k['title'] == title), None)
    secret = any(s['name'] == 'PUBLIC_EXPORT_DEPLOY_KEY' for s in secrets)
    if key and secret and not key['read_only']:
        return
    if key or secret:
        raise RuntimeError('Partial deploy-key setup: inspect and remove only R57 key/secret before retry')
    from nacl.public import PublicKey, SealedBox
    with tempfile.TemporaryDirectory(prefix='r57-key-') as directory:
        path = Path(directory)/'key'
        subprocess.run(['ssh-keygen', '-t', 'ed25519', '-N', '', '-C', title, '-f', str(path)],
                       check=True, capture_output=True)
        public = gh.api('GET', f'/repos/{PRIVATE}/actions/secrets/public-key')
        encrypted = SealedBox(PublicKey(base64.b64decode(public['key']))).encrypt(path.read_bytes())
        gh.api('PUT', f'/repos/{PRIVATE}/actions/secrets/PUBLIC_EXPORT_DEPLOY_KEY', json={
            'key_id': public['key_id'], 'encrypted_value': base64.b64encode(encrypted).decode()})
        gh.api('POST', f'/repos/{PUBLIC}/keys', json={'title': title,
               'key': path.with_suffix('.pub').read_text().strip(), 'read_only': False})


def export_allowlist(root):
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    approved = set(p for p in tracked if p and Path(p).suffix in {'.json', '.csv'}
                  and any(p.startswith(f + '/') for f in ('data', 'audit', 'learning', 'public_site/data'))
                  and not any(x in p.lower() for x in ('secret', 'credential', 'token', 'password')))
    approved.update({'public_site/data/live_snapshot.json', 'public_site/data/iteration_evidence.json',
                     'public_site/data/forecast_scorecard.json'})
    return sorted(approved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--create', action='store_true')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--deploy-key', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Download/restore private bundle and check due windows, no refresh')
    parser.add_argument('--handoff-check', action='store_true', help='Refuse switch if any local runtime state differs')
    args = parser.parse_args()
    gh = GitHub(credential())
    if args.create:
        ensure_repo(gh)
    gh.assert_private()
    if args.handoff_check:
        state = json.loads(gh.read('state.json', gh.head()))
        if inventory(args.root) != state['files']:
            raise ValueError('Local authority advanced since bootstrap; do not activate cloud')
        print('Handoff baseline equal. Keep local automation paused during approval/switch.')
    if args.verify:
        parent = gh.head()
        state = json.loads(gh.read('state.json', parent))
        with tempfile.TemporaryDirectory(prefix='r57-verify-') as directory:
            base = Path(directory)
            gh.download(state, base/'runtime.tar.gz')
            unpack(base/'runtime.tar.gz', base/'restored', state['files'])
            if inventory(base/'restored') != state['files']:
                raise ValueError('Restore verification failed')
            from datetime import datetime, timezone
            from pregame_refresh import due
            now = datetime.now(timezone.utc)
            windows = due(base/'restored', now.year-(now.month <= 2), now)
            print(json.dumps({'external_restore': 'passed', 'state_commit': parent,
                              'files': len(state['files']), 'sha256': state['sha256'],
                              'due': windows, 'refresh': False, 'fresh_actions_run': False}))
    if args.deploy_key:
        setup_key(gh)
    if args.upload:
        parent = gh.head()
        response = gh.session.get(f'https://api.github.com/repos/{PRIVATE}/contents/state.json', timeout=60)
        if response.status_code == 200:
            print('Existing state preserved; bootstrap never resets an initialized runner.')
            return
        if response.status_code != 404:
            raise RuntimeError('Cannot establish whether private state already exists')
        if (args.root/'audit/refresh.lock').exists():
            raise RuntimeError('Local refresh lock exists; bootstrap only from an idle snapshot')
        with tempfile.TemporaryDirectory(prefix='r57-bundle-') as directory:
            archive = Path(directory)/'runtime.tar.gz'
            manifest = pack(args.root, archive)
            state = gh.upload(archive)
            state.update({'files': manifest, 'generation': 0, 'parent_state': None})
        from runner.run import schedule_plan
        from datetime import datetime, timezone
        import pandas as pd
        now = datetime.now(timezone.utc)
        season = now.year-(now.month <= 2)
        plan = schedule_plan(pd.read_csv(args.root/f'data/schedule_{season}.csv').to_dict('records'), now)
        shadow = args.root/'models/retrained/active_shadow.json'
        config = {'source_repo': PUBLIC, 'source_ref': 'main', 'automatic_enabled': False,
                  'parent_approved': False, 'dry_run_verified': False,
                  'shadow_generation': json.loads(shadow.read_text()).get('generation') if shadow.exists() else None,
                  'schedule_crons': plan['cron_proposal'],
                  'public_exports': export_allowlist(args.root)}
        files = {'state.json': json.dumps(state, indent=2), 'config.json': json.dumps(config, indent=2),
                 'schedule-plan.json': json.dumps(plan, indent=2),
                 '.github/workflows/runner.yml': (args.root/'runner/workflow.yml').read_text()}
        sha = gh.commit(parent, files, 'R57 manual-only private runner bootstrap')
        print(json.dumps({'private_repository': PRIVATE, 'commit': sha, 'files': len(manifest),
                          'schedule_enabled': False, 'bundle': state['tag']}))


if __name__ == '__main__':
    main()
