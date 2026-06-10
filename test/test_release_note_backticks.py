# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_note_backticks.py"
_spec = importlib.util.spec_from_file_location("release_note_backticks", _SCRIPT)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
fix_text = _module.fix_text


class TestFixText:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Use `OpenAIChatGenerator` now.", "Use ``OpenAIChatGenerator`` now."),
            ("Set `api_key` and `azure_endpoint`.", "Set ``api_key`` and ``azure_endpoint``."),
            ('Call `Secret.from_env_var("X")`.', 'Call ``Secret.from_env_var("X")``.'),
            ("Leading `code` token.", "Leading ``code`` token."),
        ],
    )
    def test_converts_single_to_double(self, text, expected):
        assert fix_text(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "Already correct: ``OpenAIChatGenerator``.",
            "Two literals ``Secret`` and ``api_key``.",
            "No inline code at all here.",
            "RST role :func:`do_thing` must stay single.",
            "Link `Haystack <https://haystack.deepset.ai>`_ must stay single.",
        ],
    )
    def test_leaves_valid_rst_untouched(self, text):
        assert fix_text(text) == text

    def test_only_single_backticks_in_mixed_text_are_converted(self):
        text = "Use ``Secret`` for `api_key` and `azure_endpoint`."
        expected = "Use ``Secret`` for ``api_key`` and ``azure_endpoint``."
        assert fix_text(text) == expected

    def test_is_idempotent(self):
        once = fix_text("Set `x` and `y`.")
        assert fix_text(once) == once

    def test_unbalanced_single_backtick_is_left_untouched(self):
        # A stray, unpaired backtick is ambiguous, so we never rewrite it.
        text = "An unbalanced `backtick stays as is.\n"
        assert fix_text(text) == text


class TestCli:
    def test_fix_rewrites_file_and_is_idempotent(self, tmp_path):
        note = tmp_path / "note.yaml"
        note.write_text("enhancements:\n  - |\n    Use `Foo` and `Bar` now.\n", encoding="utf-8")

        first = subprocess.run([sys.executable, str(_SCRIPT), str(note)], capture_output=True, text=True, check=False)
        assert first.returncode == 1
        assert "``Foo``" in note.read_text(encoding="utf-8")
        assert "``Bar``" in note.read_text(encoding="utf-8")

        # Running again on the now-fixed file is a no-op and succeeds.
        second = subprocess.run([sys.executable, str(_SCRIPT), str(note)], capture_output=True, text=True, check=False)
        assert second.returncode == 0

    def test_check_mode_reports_without_modifying(self, tmp_path):
        note = tmp_path / "note.yaml"
        content = "enhancements:\n  - |\n    Use `Foo` now.\n"
        note.write_text(content, encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--check", str(note)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 1
        assert note.read_text(encoding="utf-8") == content

    def test_clean_file_passes(self, tmp_path):
        note = tmp_path / "note.yaml"
        note.write_text("enhancements:\n  - |\n    Use ``Foo`` now.\n", encoding="utf-8")

        result = subprocess.run([sys.executable, str(_SCRIPT), str(note)], capture_output=True, text=True, check=False)
        assert result.returncode == 0
