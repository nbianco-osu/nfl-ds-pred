import json
from pathlib import Path
import tempfile
import unittest
from dashboard_snapshot import export
from validate_delivery import validate


class DeliveryTests(unittest.TestCase):
    def fixture(self, root):
        folder = root/'public_site/data'
        folder.mkdir(parents=True)
        for name, data in [('predictions.json', [{'game_id': 'one', 'score': 21}]), ('metrics.json', {'accuracy': .6})]:
            (folder/name).write_text(json.dumps(data, indent=2))
        return folder

    def test_stable_compact_roundtrip_and_immutable_old_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = self.fixture(root)
            export(root); first = (folder/'dashboard_manifest.json').read_bytes()
            validate(root/'public_site'); export(root)
            self.assertEqual(first, (folder/'dashboard_manifest.json').read_bytes())
            entry = json.loads(first)['files']['predictions.json']
            old = (folder/entry['path']).read_bytes()
            self.assertEqual(json.loads(old), json.loads((folder/'predictions.json').read_text()))
            (folder/'predictions.json').write_text('[]'); export(root)
            self.assertEqual(old, (folder/entry['path']).read_bytes())
            self.assertNotEqual(first, (folder/'dashboard_manifest.json').read_bytes())

    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.fixture(root);export(root)
            entry=json.loads((folder/'dashboard_manifest.json').read_text())['files']['metrics.json']
            (folder/entry['path']).write_bytes(b'{}')
            with self.assertRaises(ValueError):validate(root/'public_site')
            with self.assertRaises(ValueError):export(root)

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=self.fixture(root);export(root)
            path=folder/'dashboard_manifest.json';data=json.loads(path.read_text())
            data['files']['metrics.json']['path']='../private.json';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):validate(root/'public_site')
