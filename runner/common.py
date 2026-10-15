"""Restricted bundles and private GitHub state commits."""
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

import requests

PRIVATE = 'nbianco-osu/nfl-ds-pred-runner'
PUBLIC = 'nbianco-osu/nfl-ds-pred'
ROOTS = ('models', 'data', 'backtests', 'predictions', 'audit', 'learning', 'public_site/data')
EXTENSIONS = {'.json', '.csv', '.parquet', '.joblib'}
DENIED = {'archive', '__pycache__', '.ipynb_checkpoints', 'database_import'}


def allowed(name):
    p = PurePosixPath(name)
    return (not p.is_absolute() and '\\' not in name and ':' not in name
            and all(x not in DENIED and not x.startswith('.') and x != '..' for x in p.parts)
            and p.suffix in EXTENSIONS
            and not any(x in name.lower() for x in ('credential', 'secret', 'token', 'password'))
            and any(name.startswith(root + '/') for root in ROOTS))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def inventory(root):
    root = Path(root)
    result = {}
    for folder in ROOTS:
        for path in sorted((root/folder).rglob('*')):
            name = path.relative_to(root).as_posix()
            if path.is_file() and allowed(name):
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError('Bundle link outside root')
                result[name] = digest(path)
    return result


def pack(root, target):
    manifest = inventory(root)
    if not manifest:
        raise ValueError('Empty runtime bundle')
    with tarfile.open(target, 'w:gz') as tar:
        for name in manifest:
            tar.add(Path(root)/name, arcname=name, recursive=False)
    if inventory(root) != manifest:
        raise ValueError('State changed during snapshot; retry when local refresh is idle')
    return manifest


def unpack(archive, destination, expected):
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('Bundle inventory mismatch')
        for member in members:
            if not member.isfile() or not allowed(member.name):
                raise ValueError('Unsafe bundle member')
            payload = tar.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != expected[member.name]:
                raise ValueError('Bundle digest mismatch')
            target = Path(destination)/member.name
            if not target.resolve().is_relative_to(Path(destination).resolve()):
                raise ValueError('Unsafe destination')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)


def credential():
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n',
                            text=True, capture_output=True, check=True)
    values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if not values.get('password'):
        raise RuntimeError('Git credential manager did not return a credential')
    return values['password']


class SafeSession(requests.Session):
    def request(self, *args, **kwargs):
        try:
            return super().request(*args, **kwargs)
        except requests.RequestException:
            raise RuntimeError('GitHub transport failed; URL and credentials suppressed') from None


class GitHub:
    def __init__(self, token, repo=PRIVATE):
        if repo != PRIVATE:
            raise ValueError('Private state destination is fixed')
        self.repo = repo
        self.session = SafeSession()
        self.session.headers.update({'Authorization': 'Bearer ' + token,
                                     'Accept': 'application/vnd.github+json',
                                     'X-GitHub-Api-Version': '2022-11-28'})

    def api(self, method, path, **kwargs):
        response = self.session.request(method, 'https://api.github.com' + path, timeout=120, **kwargs)
        if not response.ok:
            # Never include response bodies, headers, or credentials in errors.
            raise RuntimeError(f'GitHub {method} {path}: HTTP {response.status_code}')
        return response.json() if response.content else None

    def assert_private(self):
        repo = self.api('GET', '/repos/' + self.repo)
        if repo.get('private') is not True or repo['full_name'] != PRIVATE:
            raise ValueError('Refusing non-private or unexpected repository')
        return repo

    def head(self):
        return self.api('GET', f'/repos/{self.repo}/git/ref/heads/main')['object']['sha']

    def read(self, name, ref='main'):
        value = self.api('GET', f'/repos/{self.repo}/contents/{name}', params={'ref': ref})
        return base64.b64decode(value['content'])

    def commit(self, parent, files, message):
        self.assert_private()
        if self.head() != parent:
            raise ValueError('CAS conflict; state has advanced')
        base = self.api('GET', f'/repos/{self.repo}/git/commits/{parent}')['tree']['sha']
        tree = self.api('POST', f'/repos/{self.repo}/git/trees', json={'base_tree': base, 'tree': [
            {'path': name, 'mode': '100644', 'type': 'blob', 'content': content}
            for name, content in files.items()]})
        commit = self.api('POST', f'/repos/{self.repo}/git/commits', json={
            'message': message, 'parents': [parent], 'tree': tree['sha']})
        # A concurrently advanced main is not an ancestor of this single-parent commit.
        self.api('PATCH', f'/repos/{self.repo}/git/refs/heads/main',
                 json={'sha': commit['sha'], 'force': False})
        return commit['sha']

    def upload(self, archive):
        self.assert_private()
        sha = digest(archive)
        tag = 'bundle-' + sha
        path = f'/repos/{self.repo}/releases'
        response = self.session.get('https://api.github.com' + path + '/tags/' + tag, timeout=120)
        if response.status_code == 404:
            release = self.api('POST', path, json={'tag_name': tag, 'target_commitish': 'main',
                                                  'name': tag, 'prerelease': True})
        else:
            if not response.ok:
                raise RuntimeError(f'Release lookup HTTP {response.status_code}')
            release = response.json()
        for asset in release['assets']:
            if asset['name'] == 'runtime.tar.gz':
                if asset.get('digest') != 'sha256:' + sha:
                    raise ValueError('Existing release asset digest differs or is unavailable')
                return {'asset_id': asset['id'], 'sha256': sha, 'tag': tag}
        with Path(archive).open('rb') as f:
            response = self.session.post(release['upload_url'].split('{')[0], params={'name': 'runtime.tar.gz'},
                data=f, headers={'Content-Type': 'application/gzip'}, timeout=1800)
        if response.status_code != 201:
            raise RuntimeError(f'Release upload HTTP {response.status_code}')
        asset = response.json()
        if asset.get('digest') != 'sha256:' + sha:
            raise ValueError('Uploaded release digest mismatch')
        return {'asset_id': asset['id'], 'sha256': sha, 'tag': tag}

    def download(self, state, target):
        self.assert_private()
        url = f'https://api.github.com/repos/{self.repo}/releases/assets/{int(state["asset_id"])}'
        with self.session.get(url, headers={'Accept': 'application/octet-stream'}, stream=True, timeout=300) as r:
            if not r.ok:
                raise RuntimeError(f'Private release download HTTP {r.status_code}')
            try:
                with Path(target).open('wb') as f:
                    for chunk in r.iter_content(1024*1024):
                        f.write(chunk)
            except requests.RequestException:
                raise RuntimeError('Private download interrupted; signed URL suppressed') from None
        if digest(target) != state['sha256']:
            raise ValueError('Downloaded release digest mismatch')
