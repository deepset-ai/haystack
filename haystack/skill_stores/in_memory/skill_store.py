# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import default_to_dict
from haystack.dataclasses.skill import Skill
from haystack.dataclasses.skill_meta import SkillMeta


class InMemorySkillStore:
    """
    SkillStore that keeps skills in memory.

    Each skill has a `name`, a `description`, a markdown `instructions` body, and an optional set of bundled
    files (a mapping of relative path to text content). Skills are passed to the constructor.

    This store needs no external backend, so it is handy for tests and small in-process setups. It also serves
    as a reference for non-filesystem stores (e.g. database- or API-backed ones): a skill is looked up by its
    `name`, and bundled files by an opaque relative path, with no notion of a directory on disk.

    ### Usage example

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import Skill
    from haystack.tools import SkillToolset
    from haystack.skill_stores.in_memory import InMemorySkillStore

    store = InMemorySkillStore(
        skills=[
            Skill(
                name="pdf-forms",
                description="Use to fill PDF forms.",
                instructions="1. Open the form. 2. Fill each field.",
                files={"reference/fields.md": "Field reference ..."},
            )
        ]
    )
    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=SkillToolset(store))
    ```
    """

    def __init__(self, skills: list[Skill]) -> None:
        """
        Initialize the store with its skills.

        :param skills: The `Skill` objects to populate the store with.
        :raises ValueError: If `skills` contains two skills with the same name.
        """
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            if skill.name in self._skills:
                raise ValueError(f"A skill named '{skill.name}' was provided more than once.")
            self._skills[skill.name] = skill

    def _skill(self, name: str) -> Skill:
        """
        Return the stored skill with the given name.

        :param name: Skill name as returned by `list_skills`.
        :returns: The stored `Skill`.
        :raises KeyError: If no skill with `name` exists.
        """
        try:
            return self._skills[name]
        except KeyError:
            available = ", ".join(self._skills) or "none"
            # We suppress the original error since we are replacing it with a more informative error message
            raise KeyError(f"Unknown skill '{name}'. Available skills: {available}.") from None

    def _readable_files_hint(self, name: str) -> str:
        """
        Return a human-readable list of the files that can be read from the named skill.

        :param name: Skill name as returned by `list_skills`.
        :returns: Comma-separated relative paths, or `"none"` if the skill bundles no readable files.
        """
        return ", ".join(self.list_skill_files(name)) or "none"

    def list_skills(self) -> dict[str, SkillMeta]:
        """
        Return the metadata of all skills in the store.

        :returns: Mapping of skill name to its metadata.
        """
        return {name: SkillMeta(name=skill.name, description=skill.description) for name, skill in self._skills.items()}

    def load_skill_body(self, name: str) -> str:
        """
        Return the markdown instructions body of the named skill.

        :param name: Skill name as returned by `list_skills`.
        :returns: The skill's instruction body.
        :raises KeyError: If no skill with `name` exists.
        """
        return self._skill(name).instructions

    def list_skill_files(self, name: str) -> list[str]:
        """
        Return the relative paths of all files bundled with the named skill.

        :param name: Skill name as returned by `list_skills`.
        :returns: Sorted list of relative paths. Empty when the skill bundles no files.
        :raises KeyError: If no skill with `name` exists.
        """
        return sorted(self._skill(name).files)

    def read_skill_file(self, name: str, path: str) -> str:
        """
        Read a file bundled with the named skill.

        :param name: Skill name as returned by `list_skills`.
        :param path: Relative path of the file, as listed by `list_skill_files`.
        :returns: The file's text content.
        :raises KeyError: If no skill with `name` exists.
        :raises FileNotFoundError: If the skill bundles no file at `path`. The message lists the readable files
            so the caller can retry with a valid path.
        """
        files = self._skill(name).files
        try:
            return files[path]
        except KeyError:
            raise FileNotFoundError(
                f"File '{path}' not found in skill '{name}'. Readable files: {self._readable_files_hint(name)}."
            ) from None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this store, including its skills, to a dictionary for use with `from_dict`.

        :returns: Dictionary representation of the store.
        """
        return default_to_dict(self, skills=[skill.to_dict() for skill in self._skills.values()])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InMemorySkillStore":
        """
        Deserialize an `InMemorySkillStore`, including its skills, from its dictionary representation.

        :param data: Dictionary representation of the store, as produced by `to_dict`.
        :returns: A new `InMemorySkillStore` instance.
        """
        skills = [Skill.from_dict(skill_data) for skill_data in data.get("init_parameters", {}).get("skills", [])]
        return cls(skills=skills)
