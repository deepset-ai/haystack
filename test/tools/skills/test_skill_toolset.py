# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.tools import SkillToolset
from haystack.tools.skills.skill_toolset import _parse_frontmatter


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
        with pytest.raises(ValueError):
            _parse_frontmatter("---\n- just\n- a\n- list\n---\nbody")


class TestSkillToolset:
    def test_scans_skills(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        _write_skill(tmp_path, "excel", description="Use to edit spreadsheets.")

        toolset = SkillToolset(tmp_path)

        assert set(toolset.skills) == {"pdf-forms", "excel"}
        assert toolset.skills["pdf-forms"].description == "Use to fill PDF forms."
        assert {t.name for t in toolset} == {"load_skill", "read_skill_file"}

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            SkillToolset(tmp_path / "nope")

    def test_missing_description_raises(self, tmp_path):
        _write_skill(tmp_path, "broken", description=None)
        with pytest.raises(ValueError, match="missing a 'description'"):
            SkillToolset(tmp_path)

    def test_system_prompt_contribution_lists_skills(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        contribution = SkillToolset(tmp_path).system_prompt_contribution()
        assert "## Available Skills" in contribution
        assert "**pdf-forms**: Use to fill PDF forms." in contribution
        assert "load_skill" in contribution and "read_skill_file" in contribution

    def test_system_prompt_contribution_none_when_empty(self, tmp_path):
        assert SkillToolset(tmp_path).system_prompt_contribution() is None

    def test_load_skill_returns_body_and_manifest(self, tmp_path):
        _write_skill(
            tmp_path,
            "pdf-forms",
            description="Use to fill PDF forms.",
            body="Step 1. Do the thing.",
            files={"reference/forms.md": "details"},
        )
        load_skill = next(t for t in SkillToolset(tmp_path) if t.name == "load_skill")
        result = load_skill.invoke(name="pdf-forms")
        assert "Step 1. Do the thing." in result
        assert "reference/forms.md" in result

    def test_load_skill_unknown(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        load_skill = next(t for t in SkillToolset(tmp_path) if t.name == "load_skill")
        assert "Unknown skill 'nope'" in load_skill.invoke(name="nope")

    def test_read_skill_file(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d", files={"reference/forms.md": "form details"})
        read = next(t for t in SkillToolset(tmp_path) if t.name == "read_skill_file")
        assert read.invoke(name="pdf-forms", path="reference/forms.md") == "form details"

    def test_read_skill_file_blocks_traversal(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        (tmp_path / "secret.txt").write_text("top secret")
        read = next(t for t in SkillToolset(tmp_path) if t.name == "read_skill_file")
        result = read.invoke(name="pdf-forms", path="../secret.txt")
        assert "escapes" in result
        assert "top secret" not in result

    def test_read_skill_file_missing(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        read = next(t for t in SkillToolset(tmp_path) if t.name == "read_skill_file")
        assert "not found" in read.invoke(name="pdf-forms", path="nope.md")

    def test_to_dict_and_from_dict(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        toolset = SkillToolset(tmp_path)

        data = toolset.to_dict()
        assert data == {
            "type": "haystack.tools.skills.skill_toolset.SkillToolset",
            "data": {"skills_dir": str(tmp_path)},
        }

        restored = SkillToolset.from_dict(data)
        assert set(restored.skills) == {"pdf-forms"}
