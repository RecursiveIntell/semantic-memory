"""Fail-closed contract tests for the disposable paired-root CI projection."""
from __future__ import annotations

import copy
import gc
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path

from scripts.ci import paired_root


class PairedRootTests(unittest.TestCase):
    @staticmethod
    def fixture_repo(path: Path, files: dict[str, str]) -> str:
        path.mkdir()
        subprocess.run(['git', 'init', '-q', str(path)], check=True)
        subprocess.run(['git', '-C', str(path), 'config', 'user.name', 'Fixture'], check=True)
        subprocess.run(['git', '-C', str(path), 'config', 'user.email', 'fixture@example.invalid'], check=True)
        for name, content in files.items():
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        subprocess.run(['git', '-C', str(path), 'add', '--', '.'], check=True)
        subprocess.run(['git', '-C', str(path), 'commit', '-qm', 'fixture'], check=True)
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()

    def test_paths_and_modes_cannot_escape_or_change_source_kind(self):
        for path in ('../escape', '/absolute', 'src/../escape', 'src/./file', 'src//file', 'src\\escape', ''):
            with self.subTest(path=path), self.assertRaises(ValueError):
                paired_root.validate_entry(path, '100644')
        for mode in ('120000', '160000', '040000'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                paired_root.validate_entry('src/lib.rs', mode)
        paired_root.validate_entry('src/lib.rs', '100644')
        paired_root.validate_entry('examples/canary.rs', '100755')

    def test_lock_delta_allows_only_known_poly_kv_optional_dependency_removal(self):
        before = {'version': 4, 'package': [
            {'name': 'poly-kv', 'version': '0.1.0-alpha.1', 'dependencies': ['fib-quant 0.1.0-alpha.1', 'serde']},
            {'name': 'semantic-memory', 'version': '0.5.15', 'dependencies': ['poly-kv']},
        ]}
        after = copy.deepcopy(before)
        after['package'][0]['dependencies'].remove('fib-quant 0.1.0-alpha.1')
        paired_root.validate_lock_delta(before, after)
        for mutation in (
            lambda doc: doc['package'][1].update(version='0.5.16'),
            lambda doc: doc['package'][0]['dependencies'].remove('serde'),
            lambda doc: doc['package'].append({'name': 'shadow', 'version': '1'}),
        ):
            with self.subTest(mutation=mutation):
                invalid = copy.deepcopy(after)
                mutation(invalid)
                with self.assertRaises(ValueError):
                    paired_root.validate_lock_delta(before, invalid)

    def test_wrong_source_sha_is_rejected_before_assembly(self):
        with self.assertRaises(ValueError):
            paired_root.require_exact_sha('a' * 40, 'b' * 40, 'Libraries')
        with self.assertRaises(ValueError):
            paired_root.require_exact_sha('short', 'short', 'mirror')
        paired_root.require_exact_sha('a' * 40, 'a' * 40, 'Libraries')

    def test_forwarded_blobs_bind_exact_paths_and_modes_not_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner, mirror = root / 'owner', root / 'mirror'
            forwarded = {path: 'owner:' + path for path in paired_root.FORWARDED_EXACT_PATHS}
            owner_sha = self.fixture_repo(owner, {
                **{'semantic-memory/' + path: value for path, value in forwarded.items()},
                'semantic-memory/src/db.rs': 'partial owner-only change',
            })
            mirror_sha = self.fixture_repo(mirror, {**forwarded, 'src/db.rs': 'intentional partial difference'})
            paired_root.check_forwarded_paths(owner, mirror, owner_sha, mirror_sha)
            with self.assertRaisesRegex(ValueError, 'exact 40-hex'):
                paired_root.check_forwarded_paths(owner, mirror, 'bad', mirror_sha)
            path = next(iter(forwarded))
            (mirror / path).write_text('different')
            subprocess.run(['git', '-C', str(mirror), 'commit', '-qam', 'changed bytes'], check=True)
            changed = subprocess.check_output(['git', '-C', str(mirror), 'rev-parse', 'HEAD'], text=True).strip()
            with self.assertRaisesRegex(ValueError, path):
                paired_root.check_forwarded_paths(owner, mirror, owner_sha, changed)
            (mirror / path).write_text(forwarded[path])
            (mirror / path).chmod(0o755)
            subprocess.run(['git', '-C', str(mirror), 'add', '--', path], check=True)
            subprocess.run(['git', '-C', str(mirror), 'commit', '-qm', 'changed mode'], check=True)
            mode_changed = subprocess.check_output(['git', '-C', str(mirror), 'rev-parse', 'HEAD'], text=True).strip()
            with self.assertRaisesRegex(ValueError, path):
                paired_root.check_forwarded_paths(owner, mirror, owner_sha, mode_changed)
            (mirror / path).chmod(0o644)
            subprocess.run(['git', '-C', str(mirror), 'rm', '-qf', '--', path], check=True)
            subprocess.run(['git', '-C', str(mirror), 'commit', '-qm', 'missing path'], check=True)
            missing = subprocess.check_output(['git', '-C', str(mirror), 'rev-parse', 'HEAD'], text=True).strip()
            with self.assertRaisesRegex(ValueError, path):
                paired_root.check_forwarded_paths(owner, mirror, owner_sha, missing)

    def test_shared_census_fences_new_drift_but_allows_exact_owner_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner, baseline, current = root / 'owner', root / 'baseline', root / 'current'
            owner_sha = self.fixture_repo(owner, {
                'semantic-memory/src/same.rs': 'same',
                'semantic-memory/src/held.rs': 'owner',
                'semantic-memory/src/owner_only.rs': 'future shared',
            })
            baseline_sha = self.fixture_repo(baseline, {
                'src/same.rs': 'same', 'src/held.rs': 'held', 'src/mirror_only.rs': 'ci-only',
            })
            current_sha = self.fixture_repo(current, {
                'src/same.rs': 'same', 'src/held.rs': 'held', 'src/mirror_only.rs': 'updated ci-only',
            })
            census = (2, 1, 1, 1, 1)
            def check(sha):
                paired_root.check_shared_baseline(owner, baseline, current,
                    owner_sha, baseline_sha, sha, expected_counts=census)
            def commit(message):
                subprocess.run(['git', '-C', str(current), 'add', '-A'], check=True)
                subprocess.run(['git', '-C', str(current), 'commit', '-qm', message], check=True)
                return subprocess.check_output(['git', '-C', str(current), 'rev-parse', 'HEAD'], text=True).strip()
            check(current_sha)
            (current / 'src/held.rs').write_text('owner')
            check(commit('exact owner forward'))
            (current / 'src/held.rs').write_text('third value')
            with self.assertRaisesRegex(ValueError, 'src/held.rs'):
                check(commit('unauthorized held edit'))
            (current / 'src/held.rs').write_text('held')
            (current / 'src/same.rs').write_text('drift')
            with self.assertRaisesRegex(ValueError, 'src/same.rs'):
                check(commit('identical path drift'))
            (current / 'src/same.rs').write_text('same')
            (current / 'src/same.rs').chmod(0o755)
            with self.assertRaisesRegex(ValueError, 'src/same.rs'):
                check(commit('mode drift'))
            (current / 'src/same.rs').unlink()
            with self.assertRaisesRegex(ValueError, 'shared owner/mirror path set'):
                check(commit('missing shared path'))
            (current / 'src/same.rs').write_text('same')
            (current / 'src/same.rs').chmod(0o644)
            (current / 'src/owner_only.rs').write_text('future shared')
            with self.assertRaisesRegex(ValueError, 'shared owner/mirror path set'):
                check(commit('new shared path'))
            with self.assertRaisesRegex(ValueError, '40-hex'):
                paired_root.check_shared_baseline(owner, baseline, current, owner_sha,
                    'invalid', current_sha, expected_counts=census)
            with self.assertRaisesRegex(ValueError, 'baseline census changed'):
                paired_root.check_shared_baseline(owner, baseline, current, owner_sha,
                    baseline_sha, current_sha, expected_counts=(3, 1, 1, 1, 1))

    def test_archive_member_rejects_duplicates_and_file_directory_collisions(self):
        seen = {}
        paired_root.claim_archive_member('src', 'directory', seen)
        paired_root.claim_archive_member('src/lib.rs', 'file', seen)
        with self.assertRaises(ValueError):
            paired_root.claim_archive_member('src/lib.rs', 'file', seen)
        with self.assertRaises(ValueError):
            paired_root.claim_archive_member('src', 'file', seen)
        paired_root.claim_archive_member('top', 'file', seen)
        with self.assertRaises(ValueError):
            paired_root.claim_archive_member('top/child', 'file', seen)
        with self.assertRaises(ValueError):
            paired_root.claim_archive_member('src/lib.rs/child', 'directory', seen)

    def test_pr_merge_must_name_exact_head_as_second_parent(self):
        base = 'a' * 40
        head = 'b' * 40
        paired_root.require_pr_merge_parents([base, head], head)
        for parents in ([base], [base, 'c' * 40], [head, base]):
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                paired_root.require_pr_merge_parents(parents, head)

    def test_commit_parent_headers_survive_shallow_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            origin = root / 'origin'
            base = self.fixture_repo(origin, {'file.txt': 'base'})
            subprocess.run(['git', '-C', str(origin), 'checkout', '-qb', 'feature'], check=True)
            (origin / 'file.txt').write_text('feature')
            subprocess.run(['git', '-C', str(origin), 'commit', '-qam', 'feature'], check=True)
            head = subprocess.check_output(['git', '-C', str(origin), 'rev-parse', 'HEAD'], text=True).strip()
            subprocess.run(['git', '-C', str(origin), 'checkout', '-q', '--detach', base], check=True)
            subprocess.run(['git', '-C', str(origin), 'merge', '--no-ff', '-qm', 'merge feature', 'feature'], check=True)
            merged = subprocess.check_output(['git', '-C', str(origin), 'rev-parse', 'HEAD'], text=True).strip()
            subprocess.run(['git', '-C', str(origin), 'branch', 'merge-fixture'], check=True)
            shallow = root / 'shallow'
            subprocess.run(['git', 'clone', '-q', '--branch', 'merge-fixture', '--depth', '1',
                            'file://' + str(origin), str(shallow)], check=True)
            self.assertEqual(subprocess.check_output(
                ['git', '-C', str(shallow), 'show', '-s', '--format=%P', 'HEAD'], text=True).strip(), '')
            self.assertEqual(paired_root.commit_parents(shallow, merged), [base, head])

    def test_real_git_assembly_preserves_source_and_wrong_sha_leaves_no_scratch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner, mirror = root / 'owner', root / 'mirror'
            owner_sha = self.fixture_repo(owner, {
                'Cargo.lock': 'version = 4\n',
                'sibling.txt': 'canonical sibling',
                'semantic-memory/old.txt': 'old owner projection',
            })
            mirror_sha = self.fixture_repo(mirror, {'src/lib.rs': 'mirror source'})
            scratch = root / 'pair'
            with self.assertRaises(ValueError):
                paired_root.assemble(owner, mirror, '0' * 40, mirror_sha, scratch)
            self.assertFalse(scratch.exists())
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always', ResourceWarning)
                result = paired_root.assemble(owner, mirror, owner_sha, mirror_sha, scratch)
                gc.collect()
            self.assertFalse([w for w in captured if issubclass(w.category, ResourceWarning)])
            self.assertEqual(result['mirror_tracked_files'], 1)
            self.assertEqual((scratch / 'Libraries/semantic-memory/src/lib.rs').read_text(), 'mirror source')
            self.assertFalse((scratch / 'Libraries/semantic-memory/old.txt').exists())
            self.assertEqual((scratch / 'Libraries/sibling.txt').read_text(), 'canonical sibling')
            self.assertEqual((scratch / 'original-Cargo.lock').read_text(), 'version = 4\n')


if __name__ == '__main__':
    unittest.main()
