# SPDX-FileCopyrightText: 2026 Sankalp Gilda
#
# SPDX-License-Identifier: Apache-2.0

"""Check source mutations that must not enter a native contract build."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location("evidence_gate", Path(__file__).with_name("run.py"))
_GATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GATE)


class TestSourceSelection(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        (self.source / "haystack").mkdir(parents=True)
        for name, body in {
            "README.md": "fixture\n",
            "VERSION.txt": "3.4.0rc0\n",
            "pyproject.toml": "[project]\nname='fixture'\n",
            "haystack/__init__.py": "FIXTURE = True\n",
        }.items():
            (self.source / name).write_text(body)
        for args in [
            ["init", "-q"],
            ["add", "."],
            ["-c", "user.name=Test Author", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
        ]:
            subprocess.run(["git", *args], cwd=self.source, check=True, capture_output=True)

    def test_untracked_source_cannot_enter_selected_stage(self):
        (self.source / "haystack/unselected.py").write_text("UNSELECTED = True\n")
        data = _GATE._framework_selection(self.source, None)
        staged = self.root / "staged"
        _GATE._stage_selected_source(self.source, staged, data)
        self.assertTrue((staged / "haystack/__init__.py").is_file())
        self.assertFalse((staged / "haystack/unselected.py").exists())
        self.assertEqual(len(data["files"]), 4)

    def test_changed_tracked_source_refuses_before_stage_creation(self):
        data = _GATE._framework_selection(self.source, None)
        (self.source / "haystack/__init__.py").write_text("CHANGED = True\n")
        staged = self.root / "staged"
        with self.assertRaisesRegex(ValueError, "source blob differs: haystack/__init__.py"):
            _GATE._stage_selected_source(self.source, staged, data)
        self.assertFalse(staged.exists())

    def test_selected_symlink_refuses_before_stage_creation(self):
        data = _GATE._framework_selection(self.source, None)
        target = self.root / "outside.py"
        target.write_text("OUTSIDE = True\n")
        member = self.source / "haystack/__init__.py"
        member.unlink()
        member.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "source must not be a symlink"):
            _GATE._stage_selected_source(self.source, self.root / "staged", data)
        self.assertFalse((self.root / "staged").exists())

    def test_selected_path_cannot_escape_source(self):
        data = {"files": [{"path": "../outside.py", "sha": "0" * 40}]}
        with self.assertRaisesRegex(ValueError, "source path escapes selection"):
            _GATE._stage_selected_source(self.source, self.root / "staged", data)
        self.assertFalse((self.root / "staged").exists())

    def test_captured_selection_refuses_wrong_repository(self):
        captured = self.root / "selection.json"
        data = _GATE._framework_selection(self.source, None)
        data["repo"] = "other/project"
        captured.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "unexpected host source selection"):
            _GATE._framework_selection(self.source, captured)

    def test_child_environment_drops_credentials_and_python_overrides(self):
        values = {
            name: "not-a-secret"
            for name in [
                "AUTH_TOKEN",
                "PASSWORD",
                "PYTHONPATH",
                "PIP_INDEX_URL",
                "HATCH_ENV",
                "HATCH_ENV_ACTIVE",
                "HATCH_PROJECT",
                "VIRTUAL_ENV",
            ]
        }
        with patch.dict(os.environ, values):
            child = _GATE._clean_environment(self.root)
        self.assertTrue(all(name not in child for name in values))
        self.assertEqual(child["TMPDIR"], str(self.root / "temporary"))
        self.assertEqual(child["HAYSTACK_TELEMETRY_ENABLED"], "false")
        self.assertEqual(child["HATCH_DATA_DIR"], str(self.root / "hatch-data"))
        self.assertEqual(child["HATCH_CACHE_DIR"], str(self.root / "hatch-cache"))

    def _reference_fixture(self):
        profile = self.source / _GATE._PROFILE
        (profile / "probity_haystack").mkdir(parents=True)
        contract = (
            b'SDK_VERSION = "3.3.0"\n'
            b'SDK_SOURCE = "daa2d1ffacd083dcb1fc9a455adf541360a9e09c"\n'
            b'SUBSTANTIVE_GUARD = "retain me"\n'
        )
        metadata = b'[project]\nversion = "0.0.1"\n[project.optional-dependencies]\nproducer = ["haystack-ai==3.3.0"]\n'
        (profile / "probity_haystack/contract.py").write_bytes(contract)
        (profile / "pyproject.toml").write_bytes(metadata)
        (profile / "test_native.py").write_bytes(b"ORIGINAL_TESTS = True\n")
        return profile, contract, metadata

    def test_derivation_changes_only_declared_runtime_selectors(self):
        profile, original, metadata = self._reference_fixture()
        framework = {"commit": "a" * 40}
        receipt = _GATE._derive_host_reference(self.source, framework, "3.4.0rc0", self.root)
        retained = self.root / "reference-derivation"
        self.assertEqual((retained / "original-contract.py").read_bytes(), original)
        self.assertEqual((retained / "original-pyproject.toml").read_bytes(), metadata)
        derived = (profile / "probity_haystack/contract.py").read_bytes()
        changed = [
            (before, after) for before, after in zip(original.splitlines(), derived.splitlines()) if before != after
        ]
        self.assertEqual([before.split(b" = ")[0] for before, _ in changed], [b"SDK_VERSION", b"SDK_SOURCE"])
        self.assertIn(b'SUBSTANTIVE_GUARD = "retain me"', derived)
        self.assertEqual((profile / "test_native.py").read_bytes(), b"ORIGINAL_TESTS = True\n")
        self.assertEqual(receipt["derived_distribution_version"], "0.0.1+host.aaaaaaaaaaaa")
        self.assertFalse(receipt["substantive_guards_changed"])
        self.assertEqual(json.loads((profile / "probity_haystack/host-reference-derivation.json").read_text()), receipt)

    def test_derivation_refuses_changed_original_selector_without_mutation(self):
        profile, original, metadata = self._reference_fixture()
        path = profile / "probity_haystack/contract.py"
        changed = original.replace(b"3.3.0", b"3.3.1")
        path.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "original SDK identity declarations differ"):
            _GATE._derive_host_reference(self.source, {"commit": "a" * 40}, "3.4.0rc0", self.root)
        self.assertEqual(path.read_bytes(), changed)
        self.assertEqual((profile / "pyproject.toml").read_bytes(), metadata)
        self.assertFalse((self.root / "reference-derivation").exists())

    def test_derivation_refuses_changed_original_metadata_without_mutation(self):
        profile, original, metadata = self._reference_fixture()
        path = profile / "pyproject.toml"
        changed = metadata.replace(b"0.0.1", b"0.0.2")
        path.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "original reference metadata differs"):
            _GATE._derive_host_reference(self.source, {"commit": "a" * 40}, "3.4.0rc0", self.root)
        self.assertEqual(path.read_bytes(), changed)
        self.assertEqual((profile / "probity_haystack/contract.py").read_bytes(), original)
        self.assertFalse((self.root / "reference-derivation").exists())

    def test_launch_failure_retains_receipt_even_without_expected_exit(self):
        commands = _GATE._Commands(self.root, dict(os.environ))
        with self.assertRaisesRegex(RuntimeError, "missing failed"):
            commands.run("missing", [str(self.root / "missing-program")], expected=None)
        row = json.loads((self.root / "commands.json").read_text())[0]
        self.assertEqual(row["launch_error"], "FileNotFoundError")
        self.assertIsNone(row["exit_code"])

    def test_successful_command_retains_output_and_clear_process_group(self):
        commands = _GATE._Commands(self.root, dict(os.environ))
        row = commands.run("success", [sys.executable, "-I", "-c", "print('completed')"])
        self.assertEqual(row["exit_code"], 0)
        self.assertTrue(row["process_group_clear"])
        self.assertEqual((self.root / "success.stdout").read_text(), "completed\n")

    def test_nonzero_command_retains_exit_and_clear_process_group(self):
        commands = _GATE._Commands(self.root, dict(os.environ))
        with self.assertRaisesRegex(RuntimeError, "nonzero failed"):
            commands.run("nonzero", [sys.executable, "-I", "-c", "raise SystemExit(7)"])
        row = json.loads((self.root / "commands.json").read_text())[0]
        self.assertEqual(row["exit_code"], 7)
        self.assertTrue(row["process_group_clear"])

    def test_timeout_stops_child_and_grandchild(self):
        """A real timeout needs one second to observe the child tree before refusal."""
        pid_file = self.root / "child-pids.json"
        code = (
            "import json,os,pathlib,subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(60)']); "
            "pathlib.Path(sys.argv[1]).write_text(json.dumps([os.getpid(),child.pid])); "
            "time.sleep(60)"
        )
        commands = _GATE._Commands(self.root, dict(os.environ))
        with self.assertRaisesRegex(RuntimeError, "timeout failed"):
            commands.run("timeout", [sys.executable, "-I", "-c", code, str(pid_file)], timeout=1)
        row = json.loads((self.root / "commands.json").read_text())[0]
        pids = json.loads(pid_file.read_text())
        self.assertTrue(row["timeout"])
        self.assertTrue(row["process_group_clear"])
        self.assertTrue(set(pids).issubset(row["live_group_members_before_cleanup"]))
        self.assertEqual(_GATE._live_group_members(row["process_group"]), [])


if __name__ == "__main__":
    unittest._main()
