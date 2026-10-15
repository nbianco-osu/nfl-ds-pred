"""Reconcile final public reports only; never activate or overwrite live cloud state."""
import json
from pathlib import Path
import tempfile

from runner.common import GitHub, credential, inventory, pack, unpack
from runner.bootstrap import export_allowlist
from runner.run import preserve
from publication_gate import validate_transition
from refresh_lock import refresh_lock

REPORTS = {'public_site/data/iteration_evidence.json', 'public_site/data/live_snapshot.json',
           'public_site/data/checkpoint_schedule.json'}


def report_changes(before, after):
    if set(before) != set(after):
        raise ValueError('Handoff inventory changed; operator review required')
    changed = {name for name in before if before[name] != after[name]}
    if not changed <= REPORTS:
        raise ValueError('Non-report state changed; operator reconciliation required')
    return sorted(changed)


def reconcile(root):
    root = Path(root)
    gh = GitHub(credential())
    gh.assert_private()
    parent = gh.head()
    config = json.loads(gh.read('config.json', parent))
    if config.get('automatic_enabled') or config.get('parent_approved'):
        raise ValueError('Cannot replace an active cloud state')
    state = json.loads(gh.read('state.json', parent))
    with refresh_lock(root), tempfile.TemporaryDirectory(prefix='nfl-handoff-') as folder:
        folder = Path(folder)
        local = inventory(root)
        changes = report_changes(state['files'], local)
        if not changes:
            return {'status': 'already_equal', 'state_commit': parent}
        gh.download(state, folder/'old.tar.gz')
        unpack(folder/'old.tar.gz', folder/'previous', state['files'])
        preserve(folder/'previous', root)
        checked = validate_transition(root, folder/'previous')
        manifest = pack(root, folder/'new.tar.gz')
        if manifest != local:
            raise ValueError('Local state changed during handoff')
        new_state = gh.upload(folder/'new.tar.gz')
        new_state.update(files=manifest, generation=state['generation']+1, parent_state=parent)
        config['public_exports'] = sorted(set(config['public_exports']) | set(export_allowlist(root)))
        receipt = {'changed_reports': changes, 'validation': checked,
                   'audit_and_model_hashes_unchanged': True, 'scheduler_enabled': False}
        sha = gh.commit(parent, {'state.json': json.dumps(new_state, indent=2),
                                'config.json': json.dumps(config, indent=2),
                                'handoff.json': json.dumps(receipt, indent=2)},
                        'Reconcile final reports; preserve all model and forecast state')
        return {'status': 'reconciled', 'state_commit': sha, **receipt}


if __name__ == '__main__':
    print(json.dumps(reconcile(Path(__file__).resolve().parents[1])))
