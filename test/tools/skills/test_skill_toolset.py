# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses.skill_meta import SkillMeta
from haystack.skill_stores.file_system.skill_store import FileSystemSkillStore
from haystack.tools import SkillToolset


class _SerializableStore:
    """Module-level custom store used to test round-trip serialization."""

    def __init__(self, skills: dict[str, str]) -> None:
        self._data = skills

    def list_skills(self) -> dict[str, SkillMeta]:
        return {name: SkillMeta(name=name, description=desc) for name, desc in self._data.items()}

    def load_skill_body(self, name: str) -> str:
        if name not in self._data:
            raise KeyError(name)
        return f"Instructions for {name}."

    def list_skill_files(self, name: str) -> list[str]:
        return []

    def read_skill_file(self, name: str, path: str) -> str:
        raise FileNotFoundError

    def to_dict(self) -> dict:
        return {"type": generate_qualified_class_name(type(self)), "init_parameters": {"skills": self._data}}

    @classmethod
    def from_dict(cls, data: dict) -> "_SerializableStore":
        return cls(skills=data["init_parameters"]["skills"])


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


class TestSkillToolset:
    def test_scans_skills(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        _write_skill(tmp_path, "excel", description="Use to edit spreadsheets.")

        toolset = SkillToolset(FileSystemSkillStore(tmp_path))

        assert set(toolset.skills) == {"pdf-forms", "excel"}
        assert toolset.skills["pdf-forms"].description == "Use to fill PDF forms."
        assert {t.name for t in toolset} == {"load_skill", "read_skill_file"}

    def test_accepts_skill_store_instance(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        store = FileSystemSkillStore(tmp_path)
        toolset = SkillToolset(store)
        assert set(toolset.skills) == {"pdf-forms"}
        assert toolset._store is store

    def test_system_prompt_contribution_lists_skills(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        contribution = SkillToolset(FileSystemSkillStore(tmp_path)).system_prompt_contribution()
        assert contribution is not None
        assert "## Available Skills" in contribution
        assert "**pdf-forms**: Use to fill PDF forms." in contribution
        assert "load_skill" in contribution and "read_skill_file" in contribution

    def test_system_prompt_contribution_none_when_empty(self, tmp_path):
        assert SkillToolset(FileSystemSkillStore(tmp_path)).system_prompt_contribution() is None

    def test_load_skill_returns_body_and_manifest(self, tmp_path):
        _write_skill(
            tmp_path,
            "pdf-forms",
            description="Use to fill PDF forms.",
            body="Step 1. Do the thing.",
            files={"reference/forms.md": "details"},
        )
        load_skill = next(t for t in SkillToolset(FileSystemSkillStore(tmp_path)) if t.name == "load_skill")
        result = load_skill.invoke(name="pdf-forms")
        assert "Step 1. Do the thing." in result
        assert "reference/forms.md" in result

    def test_load_skill_unknown(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        load_skill = next(t for t in SkillToolset(FileSystemSkillStore(tmp_path)) if t.name == "load_skill")
        assert "Unknown skill 'nope'" in load_skill.invoke(name="nope")

    def test_read_skill_file(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d", files={"reference/forms.md": "form details"})
        read = next(t for t in SkillToolset(FileSystemSkillStore(tmp_path)) if t.name == "read_skill_file")
        assert read.invoke(name="pdf-forms", path="reference/forms.md") == "form details"

    def test_read_skill_file_blocks_traversal(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        (tmp_path / "secret.txt").write_text("top secret")
        read = next(t for t in SkillToolset(FileSystemSkillStore(tmp_path)) if t.name == "read_skill_file")
        result = read.invoke(name="pdf-forms", path="../secret.txt")
        assert "escapes" in result
        assert "top secret" not in result

    def test_read_skill_file_missing(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        read = next(t for t in SkillToolset(FileSystemSkillStore(tmp_path)) if t.name == "read_skill_file")
        assert "not found" in read.invoke(name="pdf-forms", path="nope.md")

    def test_to_dict_and_from_dict(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        toolset = SkillToolset(FileSystemSkillStore(tmp_path))

        data = toolset.to_dict()
        assert data == {
            "type": "haystack.tools.skills.skill_toolset.SkillToolset",
            "data": {
                "store": {
                    "type": "haystack.skill_stores.file_system.skill_store.FileSystemSkillStore",
                    "init_parameters": {"skills_dir": str(tmp_path)},
                }
            },
        }

        restored = SkillToolset.from_dict(data)
        assert set(restored.skills) == {"pdf-forms"}

    def test_to_dict_and_from_dict_with_custom_serializable_store(self):
        store = _SerializableStore(skills={"demo": "A demo skill."})
        toolset = SkillToolset(store)

        serialized = toolset.to_dict()
        assert serialized["data"]["store"]["init_parameters"]["skills"] == {"demo": "A demo skill."}

        restored = SkillToolset.from_dict(serialized)
        assert set(restored.skills) == {"demo"}

    def test_load_skill_via_custom_store(self, tmp_path):
        class _InMemoryStore:
            def list_skills(self):
                return {"demo": SkillMeta(name="demo", description="A demo skill.")}

            def load_skill_body(self, name):
                if name != "demo":
                    raise KeyError(name)
                return "Do the demo thing."

            def list_skill_files(self, name):
                return []

            def read_skill_file(self, name, path):
                raise FileNotFoundError

            def to_dict(self):
                raise NotImplementedError

            @classmethod
            def from_dict(cls, data):
                raise NotImplementedError

        toolset = SkillToolset(_InMemoryStore())
        load_skill = next(t for t in toolset if t.name == "load_skill")
        assert load_skill.invoke(name="demo") == "Do the demo thing."
        assert "Unknown skill 'nope'" in load_skill.invoke(name="nope")
