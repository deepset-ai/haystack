# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol, runtime_checkable

from haystack.dataclasses.skill_meta import SkillMeta

SKILL_FILE_NAME = "SKILL.md"


@runtime_checkable
class SkillStore(Protocol):
    """
    Protocol for a skill storage layer.

    A `SkillStore` is responsible for discovering available skills and providing their content on demand.
    Implement this class to back `haystack.tools.SkillToolset` with any storage system — a local
    directory, a database, a remote API, or an in-memory fixture.

    The three content methods (`load_skill_body`, `list_skill_files`,
    `read_skill_file`) are called lazily at agent runtime, not at construction time, so
    implementations may defer I/O until a skill is actually needed.
    """

    def list_skills(self) -> dict[str, SkillMeta]:
        """
        Discover and return all available skills.

        Called once during `haystack.tools.SkillToolset` initialization to build the skills catalog.

        :returns: Mapping of skill name to its metadata.
        """
        ...

    def load_skill_body(self, name: str) -> str:
        """
        Return the markdown body of the named skill's instructions.

        :param name: Skill name as returned by `list_skills`.
        :returns: The raw markdown body (frontmatter stripped).
        :raises KeyError: If no skill with `name` exists.
        """
        ...

    def list_skill_files(self, name: str) -> list[str]:
        """
        Return the relative paths of any files bundled with the named skill.

        :param name: Skill name as returned by `list_skills`.
        :returns: Sorted list of POSIX-style paths relative to the skill root. Empty when there are no extras.
        :raises KeyError: If no skill with `name` exists.
        """
        ...

    def read_skill_file(self, name: str, path: str) -> str:
        """
        Read a file bundled with the named skill.

        :param name: Skill name as returned by `list_skills`.
        :param path: Path of the file relative to the skill root (e.g. `"reference/forms.md"`).
        :returns: The file's text content.
        :raises KeyError: If no skill with `name` exists.
        :raises PermissionError: If `path` escapes the skill's root (path-traversal attempt).
        :raises FileNotFoundError: If the file does not exist within the skill.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this store to a dictionary for use with `from_dict`.

        Override both this method and `from_dict` to make your custom store serializable with
        `haystack.tools.SkillToolset`.
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillStore":
        """
        Deserialize a store from a dictionary produced by `to_dict`.

        Override both this method and `to_dict` to make your custom store serializable with
        `haystack.tools.SkillToolset`.

        :param data: Dictionary as produced by `to_dict`.
        """
        ...
