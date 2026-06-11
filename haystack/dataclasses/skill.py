# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillMeta:
    """
    Lightweight metadata describing a skill: just enough for an agent to decide whether to load it.

    This is what a `SkillStore` returns when listing its skills, keeping the catalog cheap; the full `Skill`
    (including the instructions body and bundled files) is fetched on demand.

    :param name: The skill's name, used to look it up.
    :param description: A short description of when to use the skill. Shown to the agent up front.
    """

    name: str
    description: str


@dataclass
class Skill(SkillMeta):
    """
    A full skill: its metadata (`name`, `description`), its instructions, and any files it bundles.

    A skill packages reusable instructions for a specific kind of task, optionally with supporting files
    (reference docs, examples, templates). It is the unit stored by a `SkillStore`.

    :param name: The skill's name, used to look it up.
    :param description: A short description of when to use the skill. Shown to the agent up front.
    :param instructions: The markdown instructions body of the skill.
    :param files: Mapping of relative path to text content for files bundled with the skill.
    """

    instructions: str = ""
    files: dict[str, str] = field(default_factory=dict)

    @property
    def meta(self) -> SkillMeta:
        """The lightweight `SkillMeta` subset of this skill."""
        return SkillMeta(name=self.name, description=self.description)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the skill to a dictionary.

        :returns: Dictionary representation of the skill.
        """
        return {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "files": self.files,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        """
        Deserialize a skill from a dictionary produced by `to_dict`.

        :param data: Dictionary representation of the skill.
        :returns: The deserialized `Skill`.
        """
        return cls(
            name=data["name"],
            description=data["description"],
            instructions=data.get("instructions", ""),
            files=data.get("files") or {},
        )
