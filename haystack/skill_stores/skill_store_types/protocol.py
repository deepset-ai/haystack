# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

from haystack.dataclasses.skill_info import SkillInfo


class SkillStore(Protocol):
    """
    Protocol for a skill storage layer.

    A `SkillStore` is responsible for discovering available skills and providing their content on demand. Implement
    this protocol to back a `haystack.tools.SkillToolset` with any storage system — a local directory, a database,
    a remote API, or an in-memory fixture.

    Skills are identified by their `name`, which must be unique within a store. The `name` is the lookup key for every
    method below; implementations resolve it to their own internal locator (a directory, a row id, an object key, ...).

    Implementations may defer all I/O (filesystem reads, database connections, ...) until a method is actually called,
    so a store can be constructed cheaply and only touch its backend on first use.

    Skill content is text: instruction bodies and bundled files are returned as strings. Binary assets (images,
    fonts, ...) are not supported.
    """

    def list_skills(self) -> dict[str, SkillInfo]:
        """
        Discover and return all available skills.

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
        Read a text file bundled with the named skill.

        :param name: Skill name as returned by `list_skills`.
        :param path: Path of the file relative to the skill root (e.g. `"reference/forms.md"`).
        :returns: The file's text content.
        :raises KeyError: If no skill with `name` exists.
        :raises FileNotFoundError: If the file does not exist within the skill.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this store to a dictionary for use with `from_dict`.

        Implement both this method and `from_dict` to make your custom store serializable.
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillStore":
        """
        Deserialize a store from a dictionary produced by `to_dict`.

        Implement both this method and `to_dict` to make your custom store serializable.

        :param data: Dictionary as produced by `to_dict`.
        """
        ...
