import csv
import tempfile
import unittest
from pathlib import Path

from reconcile_browser_import import reconcile


class BrowserReconciliationTests(unittest.TestCase):
    def test_missing_only_and_semantic_json_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, exported, output = [root / name for name in ['source.csv', 'export.csv', 'missing.csv']]
            self.write(source, [{'id': 'a', 'payload': '{"x":1,"y":2}'}, {'id': 'b', 'payload': '{}'}])
            self.write(exported, [{'id': 'a', 'payload': '{"y": 2, "x": 1}'}])
            result = reconcile(source, exported, output)
            self.assertEqual((result['matched_rows'], result['missing_rows']), (1, 1))
            with output.open(newline='') as stream:
                self.assertEqual([r['id'] for r in csv.DictReader(stream)], ['b'])
            with self.assertRaises(ValueError):
                reconcile(source, exported, output)

    def test_mismatch_or_duplicate_fails_before_output(self):
        for rows in [[{'id': 'a', 'payload': '{"x":2}'}], [{'id': 'a', 'payload': '{}'}] * 2]:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, exported, output = [root / name for name in ['source.csv', 'export.csv', 'missing.csv']]
                self.write(source, [{'id': 'a', 'payload': '{"x":1}'}])
                self.write(exported, rows)
                with self.assertRaises(ValueError):
                    reconcile(source, exported, output)
                self.assertFalse(output.exists())

    @staticmethod
    def write(path, rows):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['id', 'payload'])
            writer.writeheader()
            writer.writerows(rows)
