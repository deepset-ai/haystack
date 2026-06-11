# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import yaml

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.skill_meta import SkillMeta

SKILL_FILE_NAME = "SKILL.md"


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


class FileSystemSkillStore:
    """
    SkillStore backed by a directory of skill sub-directories on the local filesystem.

    Expected layout:

    ```
    skills/
      pdf-forms/
        SKILL.md            # frontmatter (name, description) + markdown instructions
        reference/forms.md  # optional bundled file
    ```

    The skill catalog is built by reading the frontmatter of each `SKILL.md` on `warm_up`; bodies and bundled files
    are read lazily when the agent calls the corresponding tool.
    """

    def __init__(self, skills_dir: str | Path) -> None:
        """
        Initialize the store with the root directory to scan.

        No filesystem access happens here; the directory is scanned lazily on first use (see `warm_up`), so the store
        can be constructed cheaply.

        :param skills_dir: Root directory that contains one sub-directory per skill.
        """
        self.skills_dir = Path(skills_dir)
        # Public metadata catalog returned by `list_skills`, populated on warm_up.
        self._skills: dict[str, SkillMeta] = {}
        # Private locator: maps each skill name to its directory, used to read content lazily.
        self._skill_dirs: dict[str, Path] = {}
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Scan `skills_dir` and build the skill catalog by reading each skill's `SKILL.md` frontmatter.

        Only the frontmatter is read here; bodies and bundled files are read lazily when the corresponding method is
        called. Idempotent: repeated calls after the first are no-ops.

        :raises ValueError: If `skills_dir` does not exist, is not a directory, a skill is missing a required
            frontmatter field, or two skills share the same name.
        """
        if self._is_warmed_up:
            return
        if not self.skills_dir.is_dir():
            raise ValueError(f"Skills directory '{self.skills_dir}' does not exist or is not a directory.")

        for skill_file in sorted(self.skills_dir.glob(f"*/{SKILL_FILE_NAME}")):
            skill_dir = skill_file.parent
            frontmatter, _ = _parse_frontmatter(skill_file.read_text(encoding="utf-8"))

            name = frontmatter.get("name", skill_dir.name)
            description = frontmatter.get("description")
            if not description:
                raise ValueError(f"Skill '{name}' ({skill_file}) is missing a 'description' in its frontmatter.")
            if name in self._skills:
                raise ValueError(f"Duplicate skill name '{name}' found in '{self.skills_dir}'.")

            self._skills[name] = SkillMeta(name=name, description=description)
            self._skill_dirs[name] = skill_dir
        self._is_warmed_up = True

    def _skill_dir(self, name: str) -> Path:
        """
        Return the directory of the named skill, warming up the store first if needed.

        :param name: Skill name as returned by `list_skills`.
        :returns: The skill's directory.
        :raises KeyError: If no skill with `name` exists.
        """
        self.warm_up()
        try:
            return self._skill_dirs[name]
        except KeyError:
            available = ", ".join(self._skills) or "none"
            # We suppress the original error since we are replacing it with a more informative error message
            raise KeyError(f"Unknown skill '{name}'. Available skills: {available}.") from None

    def _readable_files_hint(self, name: str) -> str:
        """
        Return a human-readable list of the files that can be read from the named skill.

        Used to make `read_skill_file` errors actionable by telling the caller which paths are valid.

        :param name: Skill name as returned by `list_skills`.
        :returns: Comma-separated relative paths, or `"none"` if the skill bundles no readable files.
        """
        return ", ".join(self.list_skill_files(name)) or "none"

    def list_skills(self) -> dict[str, SkillMeta]:
        """
        Return all skills discovered on disk, warming up the store first if needed.

        :returns: Mapping of skill name to its metadata.
        :raises ValueError: If the skills directory is invalid or a skill's frontmatter is malformed.
        """
        self.warm_up()
        return self._skills

    def load_skill_body(self, name: str) -> str:
        """
        Read the markdown body of the named skill's `SKILL.md` (frontmatter stripped).

        :param name: Skill name as returned by `list_skills`.
        :returns: The skill's instruction body.
        :raises KeyError: If no skill with `name` exists.
        """
        _, body = _parse_frontmatter((self._skill_dir(name) / SKILL_FILE_NAME).read_text(encoding="utf-8"))
        return body

    def list_skill_files(self, name: str) -> list[str]:
        """
        Return the relative paths of all files bundled with the named skill, excluding its `SKILL.md`.

        :param name: Skill name as returned by `list_skills`.
        :returns: Sorted list of POSIX-style paths relative to the skill directory. Empty when there are none.
        :raises KeyError: If no skill with `name` exists.
        """
        skill_dir = self._skill_dir(name)
        return sorted(
            p.relative_to(skill_dir).as_posix()
            for p in skill_dir.rglob("*")
            if p.is_file() and p.name != SKILL_FILE_NAME
        )

    def read_skill_file(self, name: str, path: str) -> str:
        """
        Read a file bundled with the named skill, preventing path traversal outside the skill directory.

        :param name: Skill name as returned by `list_skills`.
        :param path: Path of the file relative to the skill directory (e.g. `"reference/forms.md"`).
        :returns: The file's text content.
        :raises KeyError: If no skill with `name` exists.
        :raises PermissionError: If `path` resolves outside the skill's directory (path-traversal attempt). The
            message lists the readable files so the caller can retry with a valid path.
        :raises FileNotFoundError: If the file does not exist within the skill. The message lists the readable
            files so the caller can retry with a valid path.
        """
        skill_dir = self._skill_dir(name).resolve()
        target = (skill_dir / path).resolve()
        if skill_dir != target and skill_dir not in target.parents:
            raise PermissionError(
                f"Cannot read '{path}' from skill '{name}': the path resolves outside the skill directory. "
                f"Use a path relative to the skill root. Readable files: {self._readable_files_hint(name)}."
            )
        if not target.is_file():
            raise FileNotFoundError(
                f"File '{path}' not found in skill '{name}'. Readable files: {self._readable_files_hint(name)}."
            )
        return target.read_text(encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this store to a dictionary for use with `from_dict`.

        :returns: Dictionary representation of the store.
        """
        return default_to_dict(self, skills_dir=str(self.skills_dir))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileSystemSkillStore":
        """
        Deserialize a `FileSystemSkillStore` from its dictionary representation.

        :param data: Dictionary representation of the store, as produced by `to_dict`.
        :returns: A new `FileSystemSkillStore` instance.
        """
        return default_from_dict(cls, data)
