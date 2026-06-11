# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import Skill
from haystack.skill_stores.in_memory.skill_store import InMemorySkillStore


class TestInMemorySkillStore:
    def test_list_skills(self):
        store = InMemorySkillStore(
            [
                Skill(name="pdf-forms", description="Fill PDF forms."),
                Skill(name="excel", description="Edit spreadsheets."),
            ]
        )
        skills = store.list_skills()
        assert set(skills) == {"pdf-forms", "excel"}
        assert skills["pdf-forms"].description == "Fill PDF forms."

    def test_empty_store_lists_no_skills(self):
        assert InMemorySkillStore([]).list_skills() == {}

    def test_duplicate_skill_names_raise(self):
        with pytest.raises(ValueError, match="more than once"):
            InMemorySkillStore(
                [Skill(name="pdf-forms", description="One."), Skill(name="pdf-forms", description="Two.")]
            )

    def test_load_skill_body(self):
        store = InMemorySkillStore([Skill(name="pdf-forms", description="d", instructions="Step 1. Do the thing.")])
        assert store.load_skill_body("pdf-forms") == "Step 1. Do the thing."

    def test_load_skill_body_unknown_raises(self):
        store = InMemorySkillStore([Skill(name="pdf-forms", description="d")])
        with pytest.raises(KeyError, match="Unknown skill 'nope'. Available skills: pdf-forms."):
            store.load_skill_body("nope")

    def test_list_skill_files(self):
        store = InMemorySkillStore(
            [Skill(name="pdf-forms", description="d", files={"reference/forms.md": "details", "a.md": "x"})]
        )
        assert store.list_skill_files("pdf-forms") == ["a.md", "reference/forms.md"]

    def test_list_skill_files_empty(self):
        store = InMemorySkillStore([Skill(name="pdf-forms", description="d")])
        assert store.list_skill_files("pdf-forms") == []

    def test_read_skill_file(self):
        store = InMemorySkillStore([Skill(name="pdf-forms", description="d", files={"reference/forms.md": "details"})])
        assert store.read_skill_file("pdf-forms", "reference/forms.md") == "details"

    def test_read_skill_file_missing_raises(self):
        store = InMemorySkillStore([Skill(name="pdf-forms", description="d", files={"reference/forms.md": "details"})])
        with pytest.raises(FileNotFoundError, match="not found") as exc:
            store.read_skill_file("pdf-forms", "nope.md")
        assert "reference/forms.md" in str(exc.value)

    def test_read_skill_file_unknown_skill_raises(self):
        store = InMemorySkillStore([Skill(name="pdf-forms", description="d")])
        with pytest.raises(KeyError, match="Unknown skill 'nope'"):
            store.read_skill_file("nope", "anything.md")

    def test_to_dict_and_from_dict_round_trips_skills(self):
        store = InMemorySkillStore(
            [
                Skill(
                    name="pdf-forms",
                    description="Fill PDF forms.",
                    instructions="Do it.",
                    files={"reference/forms.md": "details"},
                )
            ]
        )

        data = store.to_dict()
        assert data == {
            "type": "haystack.skill_stores.in_memory.skill_store.InMemorySkillStore",
            "init_parameters": {
                "skills": [
                    {
                        "name": "pdf-forms",
                        "description": "Fill PDF forms.",
                        "instructions": "Do it.",
                        "files": {"reference/forms.md": "details"},
                    }
                ]
            },
        }

        restored = InMemorySkillStore.from_dict(data)
        assert set(restored.list_skills()) == {"pdf-forms"}
        assert restored.load_skill_body("pdf-forms") == "Do it."
        assert restored.read_skill_file("pdf-forms", "reference/forms.md") == "details"
