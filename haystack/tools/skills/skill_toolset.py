# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses.skill_meta import SkillMeta
from haystack.skill_stores.skill_store_types.protocol import SkillStore
from haystack.tools.from_function import create_tool_from_function
from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset
from haystack.utils.deserialization import deserialize_component_inplace


class SkillToolset(Toolset):
    """
    A Toolset that lets an Agent discover and read skills via progressive disclosure.

    A skill is a directory (or equivalent storage unit) containing a `SKILL.md` file with YAML frontmatter
    (`name` and `description`) and a markdown body of instructions. Skills may bundle additional files
    (reference docs, examples, templates). This mirrors how Claude Code and Codex expose skills:

    - On `warm_up`, the name and description of every discovered skill are baked into the `load_skill` tool
      description so the model knows which skills exist without any system prompt injection.
    - `load_skill` returns a skill's full instructions on demand, plus a manifest of its bundled files.
    - `read_skill_file` reads a bundled file on demand.

    **Example usage:**

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import SkillToolset
    from haystack.skill_stores.file_system import FileSystemSkillStore

    store = FileSystemSkillStore("skills/")
    skills = SkillToolset(store)
    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=skills)
    result = agent.run(messages=[ChatMessage.from_user("Fill in this PDF form for me.")])
    ```

    Expected filesystem layout:

    ```
    skills/
      pdf-forms/
        SKILL.md            # frontmatter (name, description) + markdown instructions
        reference/forms.md
    ```
    """

    def __init__(self, store: SkillStore) -> None:
        """
        Initialize the SkillToolset.

        Both tools (`load_skill` and `read_skill_file`) are created here, so the toolset behaves like a complete
        collection (`len`, `in`, iteration) without any I/O. The store itself is not touched until `warm_up()`:
        the skill catalog is discovered there and baked into the `load_skill` description. This lets stores that
        perform I/O (a filesystem scan, a database connection) defer that work until warm-up, which the Agent
        runs before using the toolset.

        :param store: A `haystack.skill_stores.SkillStore` instance to back this toolset.
        """
        self._store = store
        self._skills: dict[str, SkillMeta] = {}
        self._is_warmed_up = False

        # We create both tools now and dynamically update the `load_skill` description at warm-up with the discovered
        # catalog
        self._load_skill_tool = self._create_load_skill_tool()
        super().__init__(tools=[self._load_skill_tool, self._create_read_skill_file_tool()])

    @property
    def skills(self) -> dict[str, SkillMeta]:
        """Mapping of skill name to its metadata. Triggers `warm_up()` on first access if not already warmed up."""
        if not self._is_warmed_up:
            self.warm_up()
        return self._skills

    def warm_up(self) -> None:
        """
        Discover the available skills from the store and bake the catalog into the `load_skill` description.

        Only the description content is dynamic, so the (static) tools created in `__init__` are reused; this
        refreshes `load_skill`'s description once the catalog is known. Idempotent: repeated calls after the
        first are no-ops.
        """
        if self._is_warmed_up:
            return
        self._skills = self._store.list_skills()
        self._load_skill_tool.description = self._load_skill_description()
        self._is_warmed_up = True

    def _load_skill_description(self) -> str:
        """
        Build the `load_skill` tool description, including the catalog of discovered skills.

        The available skills (name + description) are baked into the description so the model can see which skills
        exist and decide when to load one, without relying on any system prompt injection.

        :returns: The tool description text.
        """
        lines = [
            "Load a skill's full instructions before doing a task it covers. Skills are specialized instruction "
            "sets for specific task types; once loaded, follow them exactly (they override your general approach). "
            "If a loaded skill references a bundled file, fetch it with `read_skill_file`."
        ]
        if self._skills:
            lines += ["", "Available skills:"]
            lines += [f"- {meta.name}: {meta.description}" for meta in self._skills.values()]
        else:
            lines += ["", "No skills are currently available."]
        return "\n".join(lines)

    def _create_load_skill_tool(self) -> Tool:
        """Create the `load_skill` tool, closed over this toolset's store."""

        def load_skill(name: Annotated[str, "Exact name of the skill to load, from the Available skills list."]) -> str:
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

        return create_tool_from_function(
            function=load_skill, name="load_skill", description=self._load_skill_description()
        )

    def _create_read_skill_file_tool(self) -> Tool:
        """Create the `read_skill_file` tool, closed over this toolset's store."""

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
            # TODO Should we just capture all errors and send the string version back to the model?
            #      Or actually the raise on tool failure is controlled within the Agent so we should really just
            #      raise the error with a good error message and let the Agent handle the final raising or stringifying.
            except (PermissionError, FileNotFoundError) as e:
                # The store raises an actionable message (it lists the readable files); surface it to the model.
                return str(e)

        return create_tool_from_function(function=read_skill_file, name="read_skill_file")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the toolset to a dictionary.

        :returns: Dictionary representation of the toolset.
        """
        return {"type": generate_qualified_class_name(type(self)), "data": {"store": self._store.to_dict()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillToolset":
        """
        Deserialize a toolset from a dictionary.

        :param data: Dictionary representation of the toolset, as produced by `to_dict`.
        :returns: A new SkillToolset instance.
        """
        inner_data = data["data"]
        deserialize_component_inplace(inner_data, key="store")
        return cls(**inner_data)
