# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Skill:
    """
    A full skill: its identity, its instructions, and any files it bundles.

    A skill packages reusable instructions for a specific kind of task, optionally with supporting files
    (reference docs, examples, templates). It is the unit written to and stored by a `SkillStore`. When a store
    lists its skills it returns the lightweight `SkillMeta` (just `name` and `description`); the full `Skill`,
    including its `instructions` body and `files`, is fetched on demand.

    :param name: Unique name of the skill, used to look it up.
    :param description: Short description of when to use the skill. Shown to the agent up front.
    :param instructions: The markdown instructions body of the skill.
    :param files: Mapping of relative path to text content for files bundled with the skill.
    """

    name: str
    description: str
    instructions: str = ""
    files: dict[str, str] = field(default_factory=dict)

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
