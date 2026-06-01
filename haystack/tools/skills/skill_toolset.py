# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import yaml

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools.from_function import create_tool_from_function
from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset

SKILL_FILE_NAME = "SKILL.md"


@dataclass
class SkillMeta:
    """
    Metadata describing a single skill discovered on disk.

    :param name: The skill's name, used by the agent to load it.
    :param description: A short description of when to use the skill. Shown to the agent up front.
    :param path: The skill's directory.
    """

    name: str
    description: str
    path: Path


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


class SkillToolset(Toolset):
    """
    A Toolset that lets an Agent discover and read filesystem "skills" via progressive disclosure.

    A skill is a directory containing a `SKILL.md` file with YAML frontmatter (`name` and `description`) and a
    markdown body of instructions. Skills may bundle additional files (reference docs, examples, templates).
    This mirrors how Claude Code and Codex expose skills:

    - The name and description of every skill are injected into the Agent's system prompt
      (via `system_prompt_contribution`) so the model knows which skills exist.
    - `load_skill` returns a skill's full instructions on demand, plus a manifest of its bundled files.
    - `read_skill_file` reads a bundled file on demand.

    Expected layout:

    ```
    skills/
      pdf-forms/
        SKILL.md            # frontmatter (name, description) + markdown instructions
        reference/forms.md
    ```

    ### Usage example

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import SkillToolset

    skills = SkillToolset("skills/")
    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=skills)
    # The skills catalog is appended to the system prompt automatically.
    result = agent.run(messages=[ChatMessage.from_user("Fill in this PDF form for me.")])
    ```
    """

    def __init__(self, skills_dir: str | Path) -> None:
        """
        Initialize the SkillToolset by scanning a directory for skills.

        Only the frontmatter of each `SKILL.md` is read at construction time (cheap); bodies and bundled files
        are read lazily when the agent calls `load_skill` / `read_skill_file`.

        :param skills_dir: Directory containing one subdirectory per skill, each with a `SKILL.md`.
        :raises ValueError: If `skills_dir` does not exist, is not a directory, a skill is missing a required
            frontmatter field, or two skills share the same name.
        """
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, SkillMeta] = self._scan()
        super().__init__(tools=[self._create_load_skill_tool(), self._create_read_skill_file_tool()])

    @property
    def skills(self) -> dict[str, SkillMeta]:
        """Mapping of skill name to its metadata."""
        return self._skills

    def _scan(self) -> dict[str, SkillMeta]:
        """
        Scan `skills_dir` for skills, reading only the frontmatter of each `SKILL.md`.

        :returns: Mapping of skill name to metadata.
        :raises ValueError: On a missing directory, missing required frontmatter, or duplicate skill names.
        """
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

    def system_prompt_contribution(self) -> str | None:
        """
        Render the skills catalog and usage instructions for injection into the Agent's system prompt.

        :returns: The catalog text, or `None` if no skills were found.
        """
        if not self._skills:
            return None

        lines = [
            "## Available Skills",
            "Specialized instruction sets for specific task types. Load one before doing matching work.",
            "",
        ]
        lines += [f"- **{meta.name}**: {meta.description}" for meta in self._skills.values()]
        lines += [
            "",
            "When a task matches a skill, call `load_skill` with its name BEFORE starting, then follow the loaded "
            "instructions exactly (they override your general approach). Load skills only when relevant; if a skill "
            "references a file, fetch it with `read_skill_file`. If no skill matches, proceed normally.",
        ]
        return "\n".join(lines)

    def _create_load_skill_tool(self) -> Tool:
        """Create the `load_skill` tool, closed over this toolset's skill registry."""

        def load_skill(name: Annotated[str, "Exact name of the skill to load, from the Available Skills list."]) -> str:
            """Load a skill's full instructions. Call this before doing a task the skill covers."""
            meta = self._skills.get(name)
            if meta is None:
                available = ", ".join(self._skills) or "none"
                return f"Unknown skill '{name}'. Available skills: {available}."

            _, body = _parse_frontmatter((meta.path / SKILL_FILE_NAME).read_text(encoding="utf-8"))

            bundled = sorted(
                str(p.relative_to(meta.path)) for p in meta.path.rglob("*") if p.is_file() and p.name != SKILL_FILE_NAME
            )
            if bundled:
                manifest = "\n".join(f"- {path}" for path in bundled)
                body = f"{body}\n\n---\nBundled files (read with `read_skill_file`):\n{manifest}"
            return body

        return create_tool_from_function(function=load_skill, name="load_skill")

    def _create_read_skill_file_tool(self) -> Tool:
        """Create the `read_skill_file` tool, closed over this toolset's skill registry."""

        def read_skill_file(
            name: Annotated[str, "Name of the skill that owns the file."],
            path: Annotated[str, "Path of the file relative to the skill directory, e.g. 'reference/forms.md'."],
        ) -> str:
            """Read a file bundled with a skill (reference docs, examples, templates)."""
            meta = self._skills.get(name)
            if meta is None:
                available = ", ".join(self._skills) or "none"
                return f"Unknown skill '{name}'. Available skills: {available}."

            skill_dir = meta.path.resolve()
            target = (skill_dir / path).resolve()
            if skill_dir != target and skill_dir not in target.parents:
                return f"Refusing to read '{path}': path escapes the '{name}' skill directory."
            if not target.is_file():
                return f"File '{path}' not found in skill '{name}'."
            return target.read_text(encoding="utf-8")

        return create_tool_from_function(function=read_skill_file, name="read_skill_file")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the toolset to a dictionary.

        Only the skills directory is serialized; tools are rebuilt by rescanning on deserialization.

        :returns: Dictionary representation of the toolset.
        """
        return {"type": generate_qualified_class_name(type(self)), "data": {"skills_dir": str(self.skills_dir)}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillToolset":
        """
        Deserialize a toolset from a dictionary.

        :param data: Dictionary representation of the toolset.
        :returns: A new SkillToolset instance.
        """
        return cls(skills_dir=data["data"]["skills_dir"])
