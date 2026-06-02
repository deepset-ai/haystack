# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from haystack.core.serialization import generate_qualified_class_name

SKILL_FILE_NAME = "SKILL.md"


@dataclass
class SkillMeta:
    """
    Metadata describing a single skill.

    :param name: The skill's name, used by the agent to load it.
    :param description: A short description of when to use the skill. Shown to the agent up front.
    :param path: The skill's directory. Set by :class:`FileSystemSkillStore`; ``None`` for other stores.
    """

    name: str
    description: str
    path: Path | None = field(default=None)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """
    Split a `SKILL.md` file into its YAML frontmatter and markdown body.

    The frontmatter is the YAML block delimited by leading and trailing `---` lines. If no frontmatter is
    present, an empty mapping and the original text are returned.

    :param text: The full contents of a `SKILL.md` file.
    :returns: A tuple of (frontmatter mapping, body).
    :raises ValueError: If the frontmatter is present but is not a valid YAML mapping.
    """
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text

    # Drop the leading '---' line, then split on the closing '---'.
    after_open = stripped[len("---") :].lstrip("\n")
    parts = after_open.split("\n---", 1)
    if len(parts) != 2:
        return {}, text

    frontmatter_block, body = parts
    loaded = yaml.safe_load(frontmatter_block) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Skill frontmatter must be a YAML mapping.")  # noqa: TRY004
    return loaded, body.lstrip("\n")


class SkillStore(ABC):
    """
    Abstract interface for a skill storage layer.

    A ``SkillStore`` is responsible for discovering available skills and providing their content on demand.
    Implement this class to back :class:`~haystack.tools.SkillToolset` with any storage system — a local
    directory, a database, a remote API, or an in-memory fixture.

    The three content methods (:meth:`load_skill_body`, :meth:`list_skill_files`,
    :meth:`read_skill_file`) are called lazily at agent runtime, not at construction time, so
    implementations may defer I/O until a skill is actually needed.
    """

    @abstractmethod
    def list_skills(self) -> dict[str, SkillMeta]:
        """
        Discover and return all available skills.

        Called once during :class:`~haystack.tools.SkillToolset` initialization to build the skills catalog.

        :returns: Mapping of skill name to its metadata.
        """

    @abstractmethod
    def load_skill_body(self, name: str) -> str:
        """
        Return the markdown body of the named skill's instructions.

        :param name: Skill name as returned by :meth:`list_skills`.
        :returns: The raw markdown body (frontmatter stripped).
        :raises KeyError: If no skill with ``name`` exists.
        """

    @abstractmethod
    def list_skill_files(self, name: str) -> list[str]:
        """
        Return the relative paths of any files bundled with the named skill.

        :param name: Skill name as returned by :meth:`list_skills`.
        :returns: Sorted list of POSIX-style paths relative to the skill root. Empty when there are no extras.
        :raises KeyError: If no skill with ``name`` exists.
        """

    @abstractmethod
    def read_skill_file(self, name: str, path: str) -> str:
        """
        Read a file bundled with the named skill.

        :param name: Skill name as returned by :meth:`list_skills`.
        :param path: Path of the file relative to the skill root (e.g. ``"reference/forms.md"``).
        :returns: The file's text content.
        :raises KeyError: If no skill with ``name`` exists.
        :raises PermissionError: If ``path`` escapes the skill's root (path-traversal attempt).
        :raises FileNotFoundError: If the file does not exist within the skill.
        """

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this store to a dictionary for use with :meth:`from_dict`.

        The default implementation raises :exc:`NotImplementedError`. Override both this method
        and :meth:`from_dict` to make your custom store serializable with
        :class:`~haystack.tools.SkillToolset`.

        The returned dictionary must include a ``"type"`` key holding the fully-qualified class name
        (use :func:`haystack.core.serialization.generate_qualified_class_name`) so that
        :meth:`~haystack.tools.SkillToolset.from_dict` can reconstruct the correct class.

        :raises NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"'{type(self).__name__}' does not implement to_dict(). "
            "Override to_dict() and from_dict() to enable serialization."
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillStore":
        """
        Deserialize a store from a dictionary produced by :meth:`to_dict`.

        The default implementation raises :exc:`NotImplementedError`. Override both this method
        and :meth:`to_dict` to make your custom store serializable with
        :class:`~haystack.tools.SkillToolset`.

        :param data: Dictionary as produced by :meth:`to_dict`.
        :raises NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"'{cls.__name__}' does not implement from_dict(). "
            "Override to_dict() and from_dict() to enable deserialization."
        )


class FileSystemSkillStore(SkillStore):
    """
    :class:`SkillStore` backed by a directory of skill sub-directories on the local filesystem.

    Expected layout::

        skills/
          pdf-forms/
            SKILL.md            # frontmatter (name, description) + markdown instructions
            reference/forms.md  # optional bundled file

    Only the frontmatter of each ``SKILL.md`` is read at construction time (cheap); bodies and bundled
    files are read lazily when the agent calls the corresponding tool.

    :param skills_dir: Root directory that contains one sub-directory per skill.
    :raises ValueError: If ``skills_dir`` does not exist, is not a directory, a skill is missing a required
        frontmatter field, or two skills share the same name.
    """

    def __init__(self, skills_dir: str | Path) -> None:
        self.skills_dir = Path(skills_dir)
        self._skills = self._scan()

    def _scan(self) -> dict[str, SkillMeta]:
        if not self.skills_dir.is_dir():
            raise ValueError(f"Skills directory '{self.skills_dir}' does not exist or is not a directory.")

        skills: dict[str, SkillMeta] = {}
        for skill_file in sorted(self.skills_dir.glob(f"*/{SKILL_FILE_NAME}")):
            skill_dir = skill_file.parent
            frontmatter, _ = _parse_frontmatter(skill_file.read_text(encoding="utf-8"))

            name = frontmatter.get("name", skill_dir.name)
            description = frontmatter.get("description")
            if not description:
                raise ValueError(f"Skill '{name}' ({skill_file}) is missing a 'description' in its frontmatter.")
            if name in skills:
                raise ValueError(f"Duplicate skill name '{name}' found in '{self.skills_dir}'.")

            skills[name] = SkillMeta(name=name, description=description, path=skill_dir)
        return skills

    def list_skills(self) -> dict[str, SkillMeta]:
        """Lists all skills available on disk"""
        return self._skills

    def load_skill_body(self, name: str) -> str:
        """Loads the skill body from disk"""
        meta = self._skills.get(name)
        if meta is None:
            raise KeyError(name)
        assert meta.path is not None  # guaranteed by _scan
        _, body = _parse_frontmatter((meta.path / SKILL_FILE_NAME).read_text(encoding="utf-8"))
        return body

    def list_skill_files(self, name: str) -> list[str]:
        """List all files in a skill directory, excluding the SKILL.md file."""
        meta = self._skills.get(name)
        if meta is None:
            raise KeyError(name)
        assert meta.path is not None  # guaranteed by _scan
        return sorted(
            p.relative_to(meta.path).as_posix()
            for p in meta.path.rglob("*")
            if p.is_file() and p.name != SKILL_FILE_NAME
        )

    def read_skill_file(self, name: str, path: str) -> str:
        """read_skill_file implementation that prevents path traversal outside the skill directory."""
        meta = self._skills.get(name)
        if meta is None:
            raise KeyError(name)
        assert meta.path is not None  # guaranteed by _scan
        skill_dir = meta.path.resolve()
        target = (skill_dir / path).resolve()
        if skill_dir != target and skill_dir not in target.parents:
            raise PermissionError(f"path escapes the '{name}' skill directory")
        if not target.is_file():
            raise FileNotFoundError(f"File '{path}' not found in skill '{name}'")
        return target.read_text(encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this store to a dictionary for use with :meth:`from_dict`."""
        return default_to_dict(self, skills_dir=str(self.skills_dir))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileSystemSkillStore":
        """Deserialize a FileSystemSkillStore from its dictionary representation."""
        return cls(skills_dir=data["data"]["skills_dir"])
