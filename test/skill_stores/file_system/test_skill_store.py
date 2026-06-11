# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from haystack.skill_stores.file_system.skill_store import FileSystemSkillStore, _parse_frontmatter


def _write_skill(skills_dir, name, description=None, body="Instructions.", files=None):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    frontmatter = f"---\nname: {name}\n"
    if description is not None:
        frontmatter += f"description: {description}\n"
    frontmatter += "---\n"
    (skill_dir / "SKILL.md").write_text(frontmatter + body, encoding="utf-8")
    for rel_path, content in (files or {}).items():
        target = skill_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return skill_dir


class TestParseFrontmatter:
    def test_parses_frontmatter_and_body(self):
        frontmatter, body = _parse_frontmatter("---\nname: a\ndescription: d\n---\nThe body.")
        assert frontmatter == {"name": "a", "description": "d"}
        assert body == "The body."

    def test_no_frontmatter_returns_empty_mapping(self):
        frontmatter, body = _parse_frontmatter("Just a body, no frontmatter.")
        assert frontmatter == {}
        assert body == "Just a body, no frontmatter."

    def test_non_mapping_frontmatter_raises(self):
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            _parse_frontmatter("---\n- just\n- a\n- list\n---\nbody")

    def test_unterminated_frontmatter_raises(self):
        with pytest.raises(ValueError, match="never closed"):
            _parse_frontmatter("---\nname: a\ndescription: d\nThe body, no closing delimiter.")

    def test_invalid_yaml_frontmatter_raises(self):
        with pytest.raises(ValueError, match="not valid YAML"):
            _parse_frontmatter("---\nname: [unclosed\n---\nbody")

    def test_opening_line_must_be_exactly_dashes(self):
        # '--- extra' and '----' are not frontmatter delimiters; the whole text is the body.
        for text in ("--- extra\nname: a\n---\nbody", "----\nname: a\n---\nbody"):
            frontmatter, body = _parse_frontmatter(text)
            assert frontmatter == {}
            assert body == text

    def test_closing_line_must_be_exactly_dashes(self):
        # A '--- something' line does not close the frontmatter, so this block is unterminated.
        with pytest.raises(ValueError, match="never closed"):
            _parse_frontmatter("---\nname: a\n--- not a delimiter\nbody")

    def test_empty_frontmatter(self):
        frontmatter, body = _parse_frontmatter("---\n---\nThe body.")
        assert frontmatter == {}
        assert body == "The body."

    def test_dashes_in_body_are_kept(self):
        # Only the first '---' line after the opening closes the frontmatter; later ones belong to the body.
        frontmatter, body = _parse_frontmatter("---\nname: a\n---\nbody\n---\nmore body")
        assert frontmatter == {"name": "a"}
        assert body == "body\n---\nmore body"

    def test_crlf_line_endings(self):
        frontmatter, body = _parse_frontmatter("---\r\nname: a\r\n---\r\nThe body.")
        assert frontmatter == {"name": "a"}
        assert body == "The body."


class TestFileSystemSkillStore:
    def test_list_skills(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Fill PDF forms.")
        _write_skill(tmp_path, "excel", description="Edit spreadsheets.")

        store = FileSystemSkillStore(tmp_path)
        skills = store.list_skills()

        assert set(skills) == {"pdf-forms", "excel"}
        assert skills["pdf-forms"].description == "Fill PDF forms."

    def test_list_skills_returns_a_copy(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Fill PDF forms.")
        store = FileSystemSkillStore(tmp_path)
        store.list_skills().clear()
        assert set(store.list_skills()) == {"pdf-forms"}

    def test_construction_is_lazy(self, tmp_path):
        # Pointing at a non-existent directory is fine until the store is warmed up / used.
        store = FileSystemSkillStore(tmp_path / "nope")
        assert store._is_warmed_up is False

    def test_missing_directory_raises_on_warm_up(self, tmp_path):
        store = FileSystemSkillStore(tmp_path / "nope")
        with pytest.raises(ValueError, match="does not exist"):
            store.warm_up()

    def test_missing_directory_raises_on_first_use(self, tmp_path):
        store = FileSystemSkillStore(tmp_path / "nope")
        with pytest.raises(ValueError, match="does not exist"):
            store.list_skills()

    def test_missing_description_raises(self, tmp_path):
        _write_skill(tmp_path, "broken", description=None)
        store = FileSystemSkillStore(tmp_path)
        with pytest.raises(ValueError, match="missing a 'description'"):
            store.warm_up()

    def test_warm_up_retry_after_failure(self, tmp_path):
        # A failed scan must not leave partial state behind that poisons the retry
        # (e.g. spurious duplicate-name errors for skills registered before the failure).
        _write_skill(tmp_path, "a-good", description="Good skill.")
        broken_dir = _write_skill(tmp_path, "broken", description=None)

        store = FileSystemSkillStore(tmp_path)
        with pytest.raises(ValueError, match="missing a 'description'"):
            store.warm_up()

        (broken_dir / "SKILL.md").write_text("---\nname: broken\ndescription: Fixed.\n---\nBody.", encoding="utf-8")
        store.warm_up()
        assert set(store.list_skills()) == {"a-good", "broken"}

    def test_concurrent_warm_up(self, tmp_path):
        # Concurrent first use (e.g. parallel requests hitting a shared Agent) must neither raise spurious
        # duplicate-name errors nor expose a partial catalog.
        for i in range(5):
            _write_skill(tmp_path, f"skill-{i}", description=f"Skill {i}.")
        store = FileSystemSkillStore(tmp_path)

        num_threads = 8
        barrier = threading.Barrier(num_threads)

        def warm_up_and_list():
            barrier.wait()
            return set(store.list_skills())

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(lambda _: warm_up_and_list(), range(num_threads)))

        expected = {f"skill-{i}" for i in range(5)}
        assert all(result == expected for result in results)

    def test_load_skill_body(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d", body="Step 1. Do the thing.")
        store = FileSystemSkillStore(tmp_path)
        assert store.load_skill_body("pdf-forms") == "Step 1. Do the thing."

    def test_load_skill_body_unknown_raises(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        store = FileSystemSkillStore(tmp_path)
        with pytest.raises(KeyError, match="Unknown skill 'nope'. Available skills: pdf-forms."):
            store.load_skill_body("nope")

    def test_list_skill_files(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d", files={"reference/forms.md": "details"})
        store = FileSystemSkillStore(tmp_path)
        assert store.list_skill_files("pdf-forms") == ["reference/forms.md"]

    def test_list_skill_files_empty(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        store = FileSystemSkillStore(tmp_path)
        assert store.list_skill_files("pdf-forms") == []

    def test_list_skill_files_unknown_raises(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        store = FileSystemSkillStore(tmp_path)
        with pytest.raises(KeyError):
            store.list_skill_files("nope")

    def test_read_skill_file(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d", files={"reference/forms.md": "form details"})
        store = FileSystemSkillStore(tmp_path)
        assert store.read_skill_file("pdf-forms", "reference/forms.md") == "form details"

    def test_read_skill_file_blocks_traversal(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d", files={"reference/forms.md": "details"})
        (tmp_path / "secret.txt").write_text("top secret")
        store = FileSystemSkillStore(tmp_path)
        with pytest.raises(PermissionError, match="resolves outside the skill directory") as exc:
            store.read_skill_file("pdf-forms", "../secret.txt")
        # The message lists valid paths to retry with, and never leaks the out-of-bounds content.
        assert "reference/forms.md" in str(exc.value)
        assert "top secret" not in str(exc.value)

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks require elevated privileges on Windows")
    def test_read_skill_file_blocks_symlink_escape(self, tmp_path):
        skill_dir = _write_skill(tmp_path, "pdf-forms", description="d")
        (tmp_path / "secret.txt").write_text("top secret")
        (skill_dir / "link.md").symlink_to(tmp_path / "secret.txt")
        store = FileSystemSkillStore(tmp_path)
        with pytest.raises(PermissionError, match="resolves outside the skill directory") as exc:
            store.read_skill_file("pdf-forms", "link.md")
        assert "top secret" not in str(exc.value)

    def test_read_skill_file_binary_raises(self, tmp_path):
        skill_dir = _write_skill(tmp_path, "pdf-forms", description="d")
        (skill_dir / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe")
        store = FileSystemSkillStore(tmp_path)
        with pytest.raises(ValueError, match="Only text files can be read"):
            store.read_skill_file("pdf-forms", "logo.png")

    def test_read_skill_file_missing_raises(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d", files={"reference/forms.md": "details"})
        store = FileSystemSkillStore(tmp_path)
        with pytest.raises(FileNotFoundError, match="not found") as exc:
            store.read_skill_file("pdf-forms", "nope.md")
        assert "reference/forms.md" in str(exc.value)

    def test_read_skill_file_unknown_skill_raises(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        store = FileSystemSkillStore(tmp_path)
        with pytest.raises(KeyError):
            store.read_skill_file("nope", "anything.md")
