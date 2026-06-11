# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses.skill_info import SkillInfo
from haystack.skill_stores.file_system.skill_store import FileSystemSkillStore
from haystack.tools import SkillToolset, Tool
from haystack.tools.errors import ToolInvocationError


class _SerializableStore:
    """Module-level custom store used to test round-trip serialization."""

    def __init__(self, skills: dict[str, str]) -> None:
        self._data = skills

    def list_skills(self) -> dict[str, SkillInfo]:
        return {name: SkillInfo(name=name, description=desc) for name, desc in self._data.items()}

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


def _get_tool(toolset, name):
    """Warm up the toolset and return its tool with the given name."""
    toolset.warm_up()
    return next(t for t in toolset if t.name == name)


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
    def test_tools_present_before_warm_up_without_io(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        toolset = SkillToolset(FileSystemSkillStore(tmp_path))

        # The (static) tool set is available immediately, with no store access required.
        assert toolset._is_warmed_up is False
        assert len(toolset) == 2
        assert {t.name for t in toolset} == {"load_skill", "read_skill_file"}
        assert "load_skill" in toolset
        assert toolset._is_warmed_up is False

    def test_scans_skills_on_warm_up(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        _write_skill(tmp_path, "excel", description="Use to edit spreadsheets.")

        toolset = SkillToolset(FileSystemSkillStore(tmp_path))

        # The catalog is only scanned on warm_up.
        assert toolset._is_warmed_up is False

        toolset.warm_up()

        assert toolset._is_warmed_up is True
        assert set(toolset.skills) == {"pdf-forms", "excel"}
        assert toolset.skills["pdf-forms"].description == "Use to fill PDF forms."

    def test_skills_property_warms_up_lazily(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        toolset = SkillToolset(FileSystemSkillStore(tmp_path))
        # Accessing `skills` without an explicit warm_up triggers it.
        assert set(toolset.skills) == {"pdf-forms"}
        assert toolset._is_warmed_up is True

    def test_warm_up_is_idempotent(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        toolset = SkillToolset(FileSystemSkillStore(tmp_path))
        toolset.warm_up()
        toolset.warm_up()
        assert set(toolset.skills) == {"pdf-forms"}

    def test_warm_up_warms_up_the_store(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        store = FileSystemSkillStore(tmp_path)
        toolset = SkillToolset(store)
        toolset.warm_up()
        assert store._is_warmed_up is True

    def test_concurrent_warm_up(self, tmp_path):
        # Concurrent first use (e.g. parallel requests hitting a shared Agent) must produce a complete,
        # consistent catalog in every thread.
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        _write_skill(tmp_path, "excel", description="Use to edit spreadsheets.")
        toolset = SkillToolset(FileSystemSkillStore(tmp_path))

        num_threads = 8
        barrier = threading.Barrier(num_threads)

        def warm_up_and_list():
            barrier.wait()
            toolset.warm_up()
            return set(toolset.skills)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(lambda _: warm_up_and_list(), range(num_threads)))

        assert all(result == {"pdf-forms", "excel"} for result in results)
        assert "- pdf-forms: Use to fill PDF forms." in toolset._load_skill_tool.description

    def test_add_is_not_supported(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        toolset = SkillToolset(FileSystemSkillStore(tmp_path))
        extra = Tool(name="extra", description="d", parameters={"type": "object", "properties": {}}, function=len)
        with pytest.raises(NotImplementedError, match="does not support adding tools"):
            toolset.add(extra)

    def test_accepts_skill_store_instance(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        store = FileSystemSkillStore(tmp_path)
        toolset = SkillToolset(store)
        toolset.warm_up()
        assert set(toolset.skills) == {"pdf-forms"}
        assert toolset._store is store

    def test_load_skill_description_lists_skills(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        load_skill = _get_tool(SkillToolset(FileSystemSkillStore(tmp_path)), "load_skill")
        assert "Available skills:" in load_skill.description
        assert "- pdf-forms: Use to fill PDF forms." in load_skill.description

    def test_load_skill_description_when_empty(self, tmp_path):
        load_skill = _get_tool(SkillToolset(FileSystemSkillStore(tmp_path)), "load_skill")
        assert "No skills are currently available." in load_skill.description

    def test_load_skill_returns_body_and_manifest(self, tmp_path):
        _write_skill(
            tmp_path,
            "pdf-forms",
            description="Use to fill PDF forms.",
            body="Step 1. Do the thing.",
            files={"reference/forms.md": "details"},
        )
        load_skill = _get_tool(SkillToolset(FileSystemSkillStore(tmp_path)), "load_skill")
        result = load_skill.invoke(name="pdf-forms")
        assert "Step 1. Do the thing." in result
        assert "reference/forms.md" in result

    def test_load_skill_unknown_raises(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="Use to fill PDF forms.")
        load_skill = _get_tool(SkillToolset(FileSystemSkillStore(tmp_path)), "load_skill")
        # The error propagates (wrapped by Tool.invoke) so the Agent can apply its tool-failure policy.
        with pytest.raises(ToolInvocationError, match="Unknown skill 'nope'"):
            load_skill.invoke(name="nope")

    def test_read_skill_file(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d", files={"reference/forms.md": "form details"})
        read = _get_tool(SkillToolset(FileSystemSkillStore(tmp_path)), "read_skill_file")
        assert read.invoke(name="pdf-forms", path="reference/forms.md") == "form details"

    def test_read_skill_file_blocks_traversal(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        (tmp_path / "secret.txt").write_text("top secret")
        read = _get_tool(SkillToolset(FileSystemSkillStore(tmp_path)), "read_skill_file")
        with pytest.raises(ToolInvocationError, match="outside the skill directory") as exc:
            read.invoke(name="pdf-forms", path="../secret.txt")
        assert "top secret" not in str(exc.value)

    def test_read_skill_file_missing(self, tmp_path):
        _write_skill(tmp_path, "pdf-forms", description="d")
        read = _get_tool(SkillToolset(FileSystemSkillStore(tmp_path)), "read_skill_file")
        with pytest.raises(ToolInvocationError, match="not found"):
            read.invoke(name="pdf-forms", path="nope.md")

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
        restored.warm_up()
        assert set(restored.skills) == {"pdf-forms"}

    def test_to_dict_and_from_dict_with_custom_serializable_store(self):
        store = _SerializableStore(skills={"demo": "A demo skill."})
        toolset = SkillToolset(store)

        serialized = toolset.to_dict()
        assert serialized["data"]["store"]["init_parameters"]["skills"] == {"demo": "A demo skill."}

        restored = SkillToolset.from_dict(serialized)
        restored.warm_up()
        assert set(restored.skills) == {"demo"}

    def test_load_skill_via_custom_store(self, tmp_path):
        class _InMemoryStore:
            def list_skills(self):
                return {"demo": SkillInfo(name="demo", description="A demo skill.")}

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

        load_skill = _get_tool(SkillToolset(_InMemoryStore()), "load_skill")
        assert load_skill.invoke(name="demo") == "Do the demo thing."
        # An unknown skill raises (wrapped by Tool.invoke); the store decides the error message.
        with pytest.raises(ToolInvocationError):
            load_skill.invoke(name="nope")
