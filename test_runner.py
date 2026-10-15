"""R57 safety boundaries, without credentials or network access."""
import io
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from runner.common import GitHub, SafeSession, allowed, inventory, pack, unpack
from runner.run import (guard_live, preserve, retained, schedule_plan, weekly_due,
                        portable_model_pointer, production_input_exports, export_cloud_schedule)


class RunnerTests(unittest.TestCase):
    def test_handoff_only_final_reports(self):
        from runner.handoff import report_changes
        original = {'models/m.joblib': 'm', 'public_site/data/live_snapshot.json': 'a'}
        self.assertEqual(report_changes(original, dict(original, **{'public_site/data/live_snapshot.json': 'b'})),
                         ['public_site/data/live_snapshot.json'])
        with self.assertRaises(ValueError):
            report_changes(original, dict(original, **{'models/m.joblib': 'new'}))
        with self.assertRaises(ValueError):
            report_changes(original, {})
    def test_archive_allowlist(self):
        for name in ['models/current.joblib', 'audit/sources/schedule/receipt.json', 'public_site/data/predictions.json']:
            self.assertTrue(allowed(name), name)
        for name in ['.git/config', 'data/.env', 'models/archive/old.joblib', 'data/credentials.json',
                     'data/a.zip', 'data/__pycache__/a.json', '../data/a.json', '/data/a.json',
                     'data/../a.json', 'data\\a.json', 'data/C:a.json', 'runner/common.py', 'audit/refresh.lock']:
            self.assertFalse(allowed(name), name)

    def test_roundtrip_and_no_git(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base/'source/data').mkdir(parents=True)
            (base/'source/data/a.json').write_text('{"a":1}')
            (base/'source/.git').mkdir()
            (base/'source/.git/config').write_text('private metadata')
            manifest = pack(base/'source', base/'bundle.tgz')
            unpack(base/'bundle.tgz', base/'restored', manifest)
            self.assertEqual(inventory(base/'restored'), manifest)
            self.assertFalse((base/'restored/.git').exists())
            with self.assertRaisesRegex(ValueError, 'digest'):
                unpack(base/'bundle.tgz', base/'bad', {'data/a.json': '0'*64})

    def test_tar_traversal_and_links(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name, kind in [('../escape.json', tarfile.REGTYPE), ('data/link.json', tarfile.SYMTYPE)]:
                with tarfile.open(base/'bad.tgz', 'w:gz') as tar:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = '/etc/passwd'
                    tar.addfile(member, io.BytesIO())
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    unpack(base/'bad.tgz', base/'out', {name: 'bad'})

    def test_activation_requires_every_gate(self):
        with self.assertRaises(ValueError):
            guard_live({'automatic_enabled': True})
        guard_live(dict.fromkeys(['automatic_enabled', 'parent_approved', 'dry_run_verified'], True))

    def test_cloud_schedule_note_requires_live_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'public_site/data/checkpoint_schedule.json'
            with self.assertRaises(ValueError):
                export_cloud_schedule(root, {'automatic_enabled': False})
            self.assertFalse(path.exists())
            export_cloud_schedule(root, dict.fromkeys(
                ['automatic_enabled', 'parent_approved', 'dry_run_verified'], True))
            report = json.loads(path.read_text())
            self.assertEqual(report['repository'], 'nbianco-osu/nfl-ds-pred-runner')
            self.assertEqual(report['weekly_refresh'], 'Tuesday 09:00 America/New_York (DST-aware)')
            self.assertEqual(report['shadow_calibrator'], 'frozen')
            self.assertIn('never backfilled', report['note'])
            self.assertNotIn('every 10 minutes', report['note'])

    def test_cas_race_refuses_state_update(self):
        gh = GitHub('unit-test-placeholder')
        gh.assert_private = Mock()
        gh.head = Mock(return_value='newer')
        gh.api = Mock()
        with self.assertRaisesRegex(ValueError, 'CAS'):
            gh.commit('old', {'state.json': '{}'}, 'test')
        gh.api.assert_not_called()

    def test_commit_never_force_pushes(self):
        gh = GitHub('unit-test-placeholder')
        gh.assert_private = Mock()
        gh.head = Mock(return_value='parent')
        gh.api = Mock(side_effect=[{'tree': {'sha': 'base'}}, {'sha': 'tree'}, {'sha': 'commit'}, {}])
        gh.commit('parent', {'state.json': '{}'}, 'test')
        self.assertEqual(gh.api.call_args.kwargs['json'], {'sha': 'commit', 'force': False})

    def test_checkpoint_and_model_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for label in ['old', 'new']:
                (base/label/'audit').mkdir(parents=True)
                (base/label/'audit/checkpoints.json').write_text('{"game|24h":{"observed_at":"original"}}')
                (base/label/'models').mkdir()
                (base/label/'models/m.joblib').write_bytes(b'model')
            preserve(base/'old', base/'new')
            (base/'new/audit/checkpoints.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Checkpoint'):
                preserve(base/'old', base/'new')
            (base/'new/models/m.joblib').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'model'):
                preserve(base/'old', base/'new')

    def test_workflow_has_no_schedule_or_broad_secret(self):
        text = (Path(__file__).parent/'runner/workflow.yml').read_text()
        self.assertNotIn('  schedule:', text)
        self.assertNotIn('cron:', text)
        self.assertIn('persist-credentials: false', text)
        self.assertIn('PUBLIC_EXPORT_DEPLOY_KEY', text)
        self.assertNotIn('git add -f', text)

    def test_transport_exception_hides_signed_url(self):
        import requests
        with patch.object(requests.Session, 'request', side_effect=requests.ConnectionError('https://secret?signed=token')):
            with self.assertRaises(RuntimeError) as result:
                SafeSession().get('https://api.github.com')
        self.assertNotIn('signed', str(result.exception))
        self.assertTrue(result.exception.__suppress_context__)

    def test_audit_append_only(self):
        self.assertTrue(retained({'a': {'value': 1}}, {'a': {'value': 1}, 'b': 2}))
        self.assertFalse(retained({'a': {'value': 1}}, {'a': {'value': 2}}))

    def test_full_season_union_and_unknown_kickoff(self):
        from datetime import datetime, timezone
        rows = [{'game_id': 'known', 'game_type': 'REG', 'gameday': '2026-12-20', 'gametime': '13:00'},
                {'game_id': 'unknown', 'game_type': 'REG', 'gameday': '2026-12-21', 'gametime': None}]
        plan = schedule_plan(rows, datetime(2026, 10, 15, tzinfo=timezone.utc))
        self.assertEqual(len(plan['windows']), 2)
        self.assertEqual(plan['weekly_checks_upper_bound'], 15)
        self.assertIn('17 11 * * *', plan['cron_proposal'])
        self.assertIn('0 13,14 * * 2', plan['cron_proposal'])
        self.assertTrue(all(len(cron.split()) == 5 for cron in plan['cron_proposal']))

    def test_weekly_eastern_dst_gate(self):
        from datetime import datetime, timezone
        self.assertTrue(weekly_due(datetime(2026, 10, 13, 13, tzinfo=timezone.utc)))
        self.assertFalse(weekly_due(datetime(2026, 10, 13, 14, tzinfo=timezone.utc)))
        self.assertTrue(weekly_due(datetime(2026, 12, 15, 14, tzinfo=timezone.utc)))
        self.assertFalse(weekly_due(datetime(2026, 12, 15, 13, tzinfo=timezone.utc)))

    def test_windows_pointer_restored_even_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root/'models/research/r25-model/models.joblib'
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b'unchanged model')
            pointer = root/'models/research/live_challengers.json'
            original = json.dumps({'artifact': r'models\research\r25-model\models.joblib'}).encode()
            pointer.write_bytes(original)
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with portable_model_pointer(root):
                    self.assertEqual(json.loads(pointer.read_text())['artifact'], 'models/research/r25-model/models.joblib')
                    raise RuntimeError('test failure')
            self.assertEqual(pointer.read_bytes(), original)
            self.assertEqual(artifact.read_bytes(), b'unchanged model')
            pointer.write_text(json.dumps({'artifact': '../private.joblib'}))
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                with portable_model_pointer(root):
                    pass

    def test_production_snapshot_export_identity_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {'game_id': '2026_06_NE_NYJ', 'model_version': 'test',
                       'features': {'x': 1.0, 'home_team': 'NYJ', 'away_team': 'NE',
                                    'game_type': 'REG', 'roof': 'outdoors', 'surface': 'fieldturf'},
                       'probability': 0.6, 'source_hashes': {}}
            sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            record = dict(payload, input_sha256=sha, receipts={}, forecast_at='2026-10-15T12:00:00Z')
            path = root/'audit/forecasts'/payload['game_id']/(sha+'.json')
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(record))
            other = root/'audit/sources/secret.json'
            other.parent.mkdir(parents=True)
            other.write_text('{}')
            self.assertEqual(production_input_exports(root), [path.relative_to(root).as_posix()])
            record['game_id'] = '2026_06_NE_BUF'
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'identity'):
                production_input_exports(root)
            record['game_id'] = payload['game_id']
            record['features'] = {'x': 2.0}
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'hash'):
                production_input_exports(root)
            baseline = root/'previous'
            old = baseline/path.relative_to(root)
            old.parent.mkdir(parents=True)
            old.write_bytes(path.read_bytes())
            self.assertEqual(production_input_exports(root, baseline), [path.relative_to(root).as_posix()])
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Existing public'):
                production_input_exports(root, baseline)


if __name__ == '__main__':
    unittest.main()
