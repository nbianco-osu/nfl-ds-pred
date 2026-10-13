"""Prepare missing rows from a Supabase browser export, without remote writes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def reconcile(source, exported, output):
    source, exported, output = map(Path, (source, exported, output))
    if output.resolve() in {source.resolve(), exported.resolve()} or output.exists():
        raise ValueError('Output must be a new file, separate from the inputs')
    csv.field_size_limit(10_000_000)
    with exported.open(encoding='utf-8-sig', newline='') as stream:
        stored = {}
        for row in csv.DictReader(stream):
            if not row.get('id') or row['id'] in stored:
                raise ValueError('Export contains missing or duplicate IDs')
            stored[row['id']] = row
    missing = []
    matched = 0
    seen = set()
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        for row in reader:
            if not row.get('id') or row['id'] in seen:
                raise ValueError('Source contains missing or duplicate IDs')
            seen.add(row['id'])
            existing = stored.get(row['id'])
            if existing is None:
                missing.append(row)
                continue
            for key in fields:
                left, right = row.get(key), existing.get(key)
                if key == 'payload':
                    left, right = json.loads(left), json.loads(right)
                if left != right:
                    raise ValueError(f'Stored row differs from source: {row["id"]}, {key}')
            matched += 1
    with output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(missing)
    receipt = {'source': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'export_sha256': hashlib.sha256(exported.read_bytes()).hexdigest(),
               'export_rows': len(stored), 'source_rows': len(seen),
               'matched_rows': matched, 'missing_rows': len(missing),
               'output': output.name, 'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--export', dest='exported', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.source, args.exported, args.output), indent=2))
