# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import yaml

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.skill_info import SkillInfo

SKILL_FILE_NAME = "SKILL.md"


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """
    Split a `SKILL.md` file into its YAML frontmatter and markdown body.

    The frontmatter is the YAML block delimited by a leading and a trailing line containing exactly `---`.
    If the first line is not `---`, no frontmatter is present and an empty mapping and the original text
    are returned.

    :param text: The full contents of a `SKILL.md` file.
    :returns: A tuple of (frontmatter mapping, body).
    :raises ValueError: If the frontmatter is opened with `---` but never closed, is not valid YAML, or is
        not a YAML mapping.
    """
    lines = text.lstrip().split("\n")
    if lines[0].rstrip() != "---":
        return {}, text

    # Find the closing delimiter: the next line containing exactly '---'.
    closing_index = next((i for i, line in enumerate(lines[1:], start=1) if line.rstrip() == "---"), None)
    if closing_index is None:
        raise ValueError("Skill frontmatter is opened with '---' but never closed with a matching '---' line.")

    frontmatter_block = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1 :])
    try:
        loaded = yaml.safe_load(frontmatter_block) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Skill frontmatter is not valid YAML: {e}") from e
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
        self._skills: dict[str, SkillInfo] = {}
        # Private locator: maps each skill name to its directory, used to read content lazily.
        self._skill_dirs: dict[str, Path] = {}
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Scan `skills_dir` and build the skill catalog by reading each skill's `SKILL.md` frontmatter.

        Only the frontmatter is read here; bodies and bundled files are read lazily when the corresponding method is
        called. Idempotent: repeated calls after the first are no-ops.

        :raises ValueError: If `skills_dir` does not exist, is not a directory, a skill's frontmatter is missing,
            malformed, or missing a required field, or two skills share the same name.
        """
        if self._is_warmed_up:
            return
        if not self.skills_dir.is_dir():
            raise ValueError(f"Skills directory '{self.skills_dir}' does not exist or is not a directory.")

        # Build into locals and swap at the end: if the scan fails halfway, no partial state is left behind (a retry
        # after fixing the offending skill starts clean), and concurrent callers only ever observe either an empty or
        # a complete catalog.
        skills: dict[str, SkillInfo] = {}
        skill_dirs: dict[str, Path] = {}
        for skill_file in sorted(self.skills_dir.glob(f"*/{SKILL_FILE_NAME}")):
            skill_dir = skill_file.parent
            frontmatter, _ = _parse_frontmatter(skill_file.read_text(encoding="utf-8"))

            name = frontmatter.get("name", skill_dir.name)
            description = frontmatter.get("description")
            if not description:
                raise ValueError(f"Skill '{name}' ({skill_file}) is missing a 'description' in its frontmatter.")
            if name in skills:
                raise ValueError(f"Duplicate skill name '{name}' found in '{self.skills_dir}'.")

            skills[name] = SkillInfo(name=name, description=description)
            skill_dirs[name] = skill_dir

        self._skills = skills
        self._skill_dirs = skill_dirs
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
        return ", ".join(self._list_skill_files(name)) or "none"

    def list_skills(self) -> dict[str, SkillInfo]:
        """
        Return all skills discovered on disk, warming up the store first if needed.

        :returns: Mapping of skill name to its metadata.
        :raises ValueError: If the skills directory is invalid or a skill's frontmatter is malformed.
        """
        self.warm_up()
        # We return a copy to prevent callers from mutating our internal state.
        return dict(self._skills)

    def load_skill(self, name: str) -> tuple[str, list[str]]:
        """
        Read the named skill's instruction body and the manifest of its bundled files.

        :param name: Skill name as returned by `list_skills`.
        :returns: A tuple of (markdown body of the skill's `SKILL.md` with frontmatter stripped, sorted list of
            POSIX-style paths relative to the skill directory for any bundled files). The file list is empty when
            the skill bundles no extras.
        :raises KeyError: If no skill with `name` exists.
        """
        _, body = _parse_frontmatter((self._skill_dir(name) / SKILL_FILE_NAME).read_text(encoding="utf-8"))
        return body, self._list_skill_files(name)

    def _list_skill_files(self, name: str) -> list[str]:
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
        Read a text file bundled with the named skill, preventing path traversal outside the skill directory.

        :param name: Skill name as returned by `list_skills`.
        :param path: Path of the file relative to the skill directory (e.g. `"reference/forms.md"`).
        :returns: The file's text content.
        :raises KeyError: If no skill with `name` exists.
        :raises PermissionError: If `path` resolves outside the skill's directory (path-traversal attempt). The
            message lists the readable files so the caller can retry with a valid path.
        :raises FileNotFoundError: If the file does not exist within the skill. The message lists the readable
            files so the caller can retry with a valid path.
        :raises ValueError: If the file is not UTF-8 text (e.g. an image or other binary asset).
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
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"File '{path}' in skill '{name}' is not UTF-8 text. Only text files can be read.") from e

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
