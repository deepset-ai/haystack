# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.skill_stores.file_system.skill_store import FileSystemSkillStore, parse_frontmatter
from haystack.skill_stores.types.protocol import SkillStore


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
        frontmatter, body = parse_frontmatter("---\nname: a\ndescription: d\n---\nThe body.")
        assert frontmatter == {"name": "a", "description": "d"}
        assert body == "The body."

    def test_no_frontmatter_returns_empty_mapping(self):
        frontmatter, body = parse_frontmatter("Just a body, no frontmatter.")
        assert frontmatter == {}
        assert body == "Just a body, no frontmatter."

    def test_non_mapping_frontmatter_raises(self):
        with pytest.raises(ValueError):
            parse_frontmatter("---\n- just\n- a\n- list\n---\nbody")


class TestFileSystemSkillStore:
    def test_list_skills(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Fill PDF forms.")
        _write_skill(tmp_path, "excel", description="Edit spreadsheets.")

        store = FileSystemSkillStore(tmp_path)
        skills = store.list_skills()

        assert set(skills) == {"pdf-forms", "excel"}
        assert skills["pdf-forms"].description == "Fill PDF forms."
        assert skills["pdf-forms"].path == tmp_path / "pdf-forms"

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            FileSystemSkillStore(tmp_path / "nope")

    def test_missing_description_raises(self, tmp_path):
        _write_skill(tmp_path, "broken", description=None)
        with pytest.raises(ValueError, match="missing a 'description'"):
            FileSystemSkillStore(tmp_path)

    def test_load_skill_body(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d", body="Step 1. Do the thing.")
        store = FileSystemSkillStore(tmp_path)
        assert store.load_skill_body("pdf-forms") == "Step 1. Do the thing."

    def test_load_skill_body_unknown_raises(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        store = FileSystemSkillStore(tmp_path)
        with pytest.raises(KeyError):
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
        _write_skill(tmp_path, "pdf-forms", description="d")
        (tmp_path / "secret.txt").write_text("top secret")
        store = FileSystemSkillStore(tmp_path)
        with pytest.raises(PermissionError, match="escapes"):
            store.read_skill_file("pdf-forms", "../secret.txt")

    def test_read_skill_file_missing_raises(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        store = FileSystemSkillStore(tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            store.read_skill_file("pdf-forms", "nope.md")

    def test_read_skill_file_unknown_skill_raises(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        store = FileSystemSkillStore(tmp_path)
        with pytest.raises(KeyError):
            store.read_skill_file("nope", "anything.md")

    def test_is_skill_store(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        assert isinstance(FileSystemSkillStore(tmp_path), SkillStore)
