#!/usr/bin/env python3
"""Assemble a disposable, source-pinned paired root for CI (not a sync)."""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path, PurePosixPath


REQUIRED_POLY_KV_DROP = 'fib-quant 0.1.0-alpha.1'


def require_exact_sha(observed: str, expected: str, owner: str) -> None:
    if not re.fullmatch(r'[0-9a-f]{40}', expected) or observed != expected:
        raise ValueError(f'{owner} source SHA differs from pinned exact 40-hex revision')


def require_pr_merge_parents(parents: list[str], pr_head_sha: str) -> None:
    if not re.fullmatch(r'[0-9a-f]{40}', pr_head_sha) or len(parents) != 2 or parents[1] != pr_head_sha:
        raise ValueError('pull_request merge revision does not bind exact PR head as second parent')


def validate_entry(path: str, mode: str) -> None:
    posix = PurePosixPath(path)
    if not path or '\\' in path or posix.is_absolute() or any(part in ('..', '.', '') for part in path.split('/')):
        raise ValueError(f'unsafe source path: {path!r}')
    if mode not in ('100644', '100755'):
        raise ValueError(f'unsupported source mode for {path}: {mode}')


def claim_archive_member(path: str, kind: str, seen: dict[str, str]) -> None:
    validate_entry(path, '100644')
    if kind not in ('directory', 'file', 'symlink') or path in seen:
        raise ValueError(f'duplicate or unsupported archive entry: {path}')
    parts = path.split('/')
    if any(seen.get('/'.join(parts[:index])) not in (None, 'directory') for index in range(1, len(parts))):
        raise ValueError(f'archive path collides with non-directory ancestor: {path}')
    if kind != 'directory' and any(existing.startswith(path + '/') for existing in seen):
        raise ValueError(f'archive path collides with existing descendants: {path}')
    seen[path] = kind


def validate_lock_delta(before: dict, after: dict) -> None:
    expected = copy.deepcopy(before)
    packages = [p for p in expected.get('package', []) if p.get('name') == 'poly-kv' and p.get('version') == '0.1.0-alpha.1']
    if len(packages) != 1 or REQUIRED_POLY_KV_DROP not in packages[0].get('dependencies', []):
        raise ValueError('pinned lock has no unique PolyKV FibQuant dependency to drop')
    packages[0]['dependencies'].remove(REQUIRED_POLY_KV_DROP)
    if expected != after:
        raise ValueError('scratch lock changed outside the exact declared PolyKV dependency drop')


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(repo), *args])


def commit_parents(repo: Path, sha: str) -> list[str]:
    # Git's pretty-format %P hides parents behind a shallow-checkout graft.
    # The raw commit object retains both parent headers even at depth one.
    header = git(repo, 'cat-file', '-p', sha).split(b'\n\n', 1)[0]
    parents = []
    for line in header.splitlines():
        if line.startswith(b'parent '):
            parent = line[len(b'parent '):].decode('ascii', 'strict')
            if not re.fullmatch(r'[0-9a-f]{40}', parent):
                raise ValueError('invalid parent in raw Git commit header')
            parents.append(parent)
    return parents


def tracked_files(repo: Path, sha: str, *, strict: bool) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    for row in git(repo, 'ls-tree', '-r', '-z', sha).split(b'\0'):
        if not row:
            continue
        meta, raw_name = row.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        name = raw_name.decode('utf-8', 'strict')
        if strict:
            validate_entry(name, mode)
            if kind != 'blob':
                raise ValueError(f'unsupported mirror object: {name}')
        records[name] = (mode, oid)
    if not records:
        raise ValueError(f'no tracked files in {repo}')
    return records


def extract_archive(repo: Path, sha: str, destination: Path, *, owner: bool) -> None:
    proc = subprocess.Popen(
        ['git', '-C', str(repo), 'archive', '--format=tar', sha],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdout is not None
        seen: dict[str, str] = {}
        with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
            for member in archive:
                name = member.name.rstrip('/')
                if not name:
                    continue
                if owner and (name == 'semantic-memory' or name.startswith('semantic-memory/')):
                    continue
                if member.isdir():
                    claim_archive_member(name, 'directory', seen)
                    # Git archive directory entries are not durable source objects.
                    validate_entry(name, '100644')
                    (destination / name).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    if owner and member.issym():
                        # Pinned Libraries archive has unrelated salvage symlinks.
                        claim_archive_member(name, 'symlink', seen)
                        continue
                    raise ValueError(f'non-regular archive entry: {name}')
                claim_archive_member(name, 'file', seen)
                mode = '100755' if member.mode & 0o111 else '100644'
                validate_entry(name, mode)
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f'missing archive payload: {name}')
                with source, target.open('wb') as handle:
                    shutil.copyfileobj(source, handle)
                target.chmod(0o755 if mode == '100755' else 0o644)
        stderr = proc.stderr.read() if proc.stderr is not None else b''
        if proc.wait() != 0:
            raise ValueError(f'git archive failed: {stderr.decode(errors="replace")}')
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


def assemble(libraries: Path, mirror: Path, libraries_sha: str, mirror_sha: str, scratch: Path,
             pr_head_sha: str = '') -> dict:
    require_exact_sha(git(libraries, 'rev-parse', 'HEAD').decode().strip(), libraries_sha, 'Libraries')
    require_exact_sha(git(mirror, 'rev-parse', 'HEAD').decode().strip(), mirror_sha, 'mirror')
    if pr_head_sha:
        require_pr_merge_parents(commit_parents(mirror, mirror_sha), pr_head_sha)
    files = tracked_files(mirror, mirror_sha, strict=True)
    # Refuse to overwrite any existing source or artifact, including a symlink.
    scratch.mkdir(parents=True, exist_ok=False)
    paired = scratch / 'Libraries'
    paired.mkdir()
    extract_archive(libraries, libraries_sha, paired, owner=True)
    package = paired / 'semantic-memory'
    package.mkdir()
    extract_archive(mirror, mirror_sha, package, owner=False)
    for path, (mode, oid) in files.items():
        target = package / path
        if not target.is_file() or git(mirror, 'hash-object', str(target)).decode().strip() != oid:
            raise ValueError(f'assembled mirror content differs from Git source: {path}')
        if bool(target.stat().st_mode & 0o111) != (mode == '100755'):
            raise ValueError(f'assembled mirror executable mode differs: {path}')
    shutil.copy2(paired / 'Cargo.lock', scratch / 'original-Cargo.lock')
    result = {'libraries_sha': libraries_sha, 'mirror_sha': mirror_sha, 'pr_head_sha': pr_head_sha or None,
              'mirror_tracked_files': len(files),
              'paired_root': str(paired), 'original_lock': str(scratch / 'original-Cargo.lock')}
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    build = sub.add_parser('assemble')
    for flag in ('libraries', 'mirror', 'libraries-sha', 'mirror-sha', 'scratch'):
        build.add_argument('--' + flag, required=True)
    build.add_argument('--pr-head-sha', default='')
    lock = sub.add_parser('check-lock')
    lock.add_argument('--before', required=True)
    lock.add_argument('--after', required=True)
    args = parser.parse_args()
    if args.action == 'assemble':
        assemble(Path(args.libraries), Path(args.mirror), args.libraries_sha, args.mirror_sha,
                 Path(args.scratch), args.pr_head_sha)
    else:
        validate_lock_delta(tomllib.loads(Path(args.before).read_text()), tomllib.loads(Path(args.after).read_text()))
        print('scratch lock delta is exactly the declared PolyKV FibQuant dependency drop')


if __name__ == '__main__':
    main()
