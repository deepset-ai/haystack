# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated, Any

from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.tools.from_function import create_tool_from_function

# Re-exported for backward compatibility with code that imports these from this module.
from haystack.tools.skills.skill_store import (  # noqa: F401
    SKILL_FILE_NAME,
    FileSystemSkillStore,
    SkillMeta,
    SkillStore,
    _parse_frontmatter,
)
from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset


class SkillToolset(Toolset):
    """
    A Toolset that lets an Agent discover and read skills via progressive disclosure.

    A skill is a directory (or equivalent storage unit) containing a ``SKILL.md`` file with YAML frontmatter
    (``name`` and ``description``) and a markdown body of instructions. Skills may bundle additional files
    (reference docs, examples, templates). This mirrors how Claude Code and Codex expose skills:

    - The name and description of every skill are injected into the Agent's system prompt
      (via ``system_prompt_contribution``) so the model knows which skills exist.
    - ``load_skill`` returns a skill's full instructions on demand, plus a manifest of its bundled files.
    - ``read_skill_file`` reads a bundled file on demand.

    **Filesystem usage** — pass a directory path directly:

    .. code-block:: python

        from haystack.components.agents import Agent
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.dataclasses import ChatMessage
        from haystack.tools import SkillToolset

        skills = SkillToolset("skills/")
        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=skills)
        result = agent.run(messages=[ChatMessage.from_user("Fill in this PDF form for me.")])

    **Custom store** — pass any :class:`~haystack.tools.SkillStore` implementation:

    .. code-block:: python

        from haystack.tools import SkillToolset
        from haystack.tools.skills import FileSystemSkillStore

        store = FileSystemSkillStore("skills/")
        skills = SkillToolset(store)

    Expected filesystem layout:

    .. code-block::

        skills/
          pdf-forms/
            SKILL.md            # frontmatter (name, description) + markdown instructions
            reference/forms.md
    """

    def __init__(self, skills_dir: "SkillStore | str | Path") -> None:
        """
        Initialize the SkillToolset.

        :param skills_dir: Either a :class:`~haystack.tools.SkillStore` instance, or a ``str``/``Path``
            pointing to a directory of skills. A plain path is automatically wrapped in a
            :class:`~haystack.tools.skills.FileSystemSkillStore`.
        :raises ValueError: If a plain path is given and the directory does not exist, a skill is missing a
            required frontmatter field, or two skills share the same name.
        """
        if isinstance(skills_dir, SkillStore):
            self._store = skills_dir
        else:
            self._store = FileSystemSkillStore(skills_dir)

        self._skills: dict[str, SkillMeta] = self._store.list_skills()
        super().__init__(tools=[self._create_load_skill_tool(), self._create_read_skill_file_tool()])

    @property
    def skills_dir(self) -> Path | None:
        """The skills directory, if this toolset is backed by a :class:`FileSystemSkillStore`; ``None`` otherwise."""
        return getattr(self._store, "skills_dir", None)

    @property
    def skills(self) -> dict[str, SkillMeta]:
        """Mapping of skill name to its metadata."""
        return self._skills

    def system_prompt_contribution(self) -> str | None:
        """
        Render the skills catalog and usage instructions for injection into the Agent's system prompt.

        :returns: The catalog text, or ``None`` if no skills were found.
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
        """Create the ``load_skill`` tool, closed over this toolset's store."""

        def load_skill(name: Annotated[str, "Exact name of the skill to load, from the Available Skills list."]) -> str:
            """Load a skill's full instructions. Call this before doing a task the skill covers."""
            try:
                body = self._store.load_skill_body(name)
                bundled = self._store.list_skill_files(name)
            except KeyError:
                available = ", ".join(self._skills) or "none"
                return f"Unknown skill '{name}'. Available skills: {available}."

            if bundled:
                manifest = "\n".join(f"- {path}" for path in bundled)
                body = f"{body}\n\n---\nBundled files (read with `read_skill_file`):\n{manifest}"
            return body

        return create_tool_from_function(function=load_skill, name="load_skill")

    def _create_read_skill_file_tool(self) -> Tool:
        """Create the ``read_skill_file`` tool, closed over this toolset's store."""

        def read_skill_file(
            name: Annotated[str, "Name of the skill that owns the file."],
            path: Annotated[str, "Path of the file relative to the skill directory, e.g. 'reference/forms.md'."],
        ) -> str:
            """Read a file bundled with a skill (reference docs, examples, templates)."""
            try:
                return self._store.read_skill_file(name, path)
            except KeyError:
                available = ", ".join(self._skills) or "none"
                return f"Unknown skill '{name}'. Available skills: {available}."
            except PermissionError:
                return f"Refusing to read '{path}': path escapes the '{name}' skill directory."
            except FileNotFoundError:
                return f"File '{path}' not found in skill '{name}'."

        return create_tool_from_function(function=read_skill_file, name="read_skill_file")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the toolset to a dictionary.

        Delegates to the backing store's :meth:`~haystack.tools.skills.SkillStore.to_dict` method.
        Tools are rebuilt by re-scanning on deserialization — only the store descriptor is persisted.

        :returns: Dictionary representation of the toolset.
        :raises NotImplementedError: If the backing store does not implement ``to_dict()``.
        """
        return {"type": generate_qualified_class_name(type(self)), "data": {"store": self._store.to_dict()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillToolset":
        """
        Deserialize a toolset from a dictionary.

        Reads ``data["data"]["store"]``, resolves the store class via its ``"type"`` key, and calls
        ``store_class.from_dict(store_data)`` to reconstruct the store.

        Also accepts the legacy format where ``data["data"]`` contains a ``"skills_dir"`` key directly
        (produced by the previous serialization implementation).

        :param data: Dictionary representation of the toolset, as produced by :meth:`to_dict`.
        :returns: A new SkillToolset instance.
        :raises TypeError: If the resolved class is not a :class:`~haystack.tools.skills.SkillStore` subclass.
        """
        inner = data["data"]
        if "skills_dir" in inner:
            # Legacy format: {"data": {"skills_dir": "..."}}
            return cls(skills_dir=inner["skills_dir"])
        store_data = inner["store"]
        store_class = import_class_by_name(store_data["type"])
        if not issubclass(store_class, SkillStore):
            raise TypeError(
                f"Expected a SkillStore subclass, got '{store_class.__name__}'. "
                "Ensure the 'type' field in the store dictionary points to a SkillStore implementation."
            )
        return cls(skills_dir=store_class.from_dict(store_data))
