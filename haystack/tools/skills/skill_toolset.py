# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses.skill_info import SkillInfo
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
    (reference docs, examples, templates).

    - On `warm_up`, the name and description of every discovered skill are baked into the `load_skill` tool
      description so the model knows which skills exist without any system prompt injection.
    - `load_skill` returns a skill's full instructions on demand, plus a manifest of its bundled files.
    - `read_skill_file` reads a bundled file on demand.

    ### Usage example

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import SkillToolset
    from haystack.skill_stores.file_system import FileSystemSkillStore

    store = FileSystemSkillStore("skills/")
    skills_toolset = SkillToolset(store)
    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=skills_toolset)
    result = agent.run(messages=[ChatMessage.from_user("Fill in this PDF form for me.")])
    ```

    Expected filesystem layout:

    ```
    skills/
      pdf-forms/
        SKILL.md            # frontmatter (name, description) + markdown instructions
        reference/forms.md
    ```

    The tool names `load_skill` and `read_skill_file` are fixed, so an `Agent` can use at most one
    `SkillToolset`. To serve skills from multiple sources, back a single toolset with a custom store that
    merges them.
    """

    def __init__(self, store: SkillStore) -> None:
        """
        Initialize the SkillToolset.

        Constructing the toolset does not read any skills. The store is queried for the available skills on
        `warm_up()`, so stores that do I/O (reading a directory, connecting to a database) stay cheap to
        construct.

        The `load_skill` and `read_skill_file` tools are created right away, so the toolset can be used as a
        collection (length, membership checks, iteration) immediately.

        :param store: A `haystack.skill_stores.skill_store_types.SkillStore` instance to back this toolset.
        """
        self._store = store
        self._skills: dict[str, SkillInfo] = {}
        self._is_warmed_up = False

        # We create both tools now and dynamically update the `load_skill` description at warm-up with the discovered
        # catalog
        self._load_skill_tool = self._create_load_skill_tool()
        super().__init__(tools=[self._load_skill_tool, self._create_read_skill_file_tool()])

    @property
    def skills(self) -> dict[str, SkillInfo]:
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
        if hasattr(self._store, "warm_up"):
            self._store.warm_up()
        self._skills = self._store.list_skills()
        self._load_skill_tool.description = self._load_skill_description()
        self._is_warmed_up = True

    def add(self, tool: Tool | Toolset) -> None:
        """Adding tools is not supported: a SkillToolset's tools are fixed and defined by its store."""
        raise NotImplementedError(
            "SkillToolset does not support adding tools. To combine it with other tools, pass it to the Agent "
            "alongside them, e.g. tools=[skill_toolset, other_tool]."
        )

    def __add__(self, other: Tool | Toolset | list[Tool]) -> "Toolset":
        """Concatenation is not supported for SearchableToolset."""
        raise NotImplementedError(
            "SkillToolset does not support concatenation. To combine it with other tools, pass it to the Agent "
            "alongside them, e.g. tools=[skill_toolset, other_tool]."
        )

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
            # The store raises an actionable error (e.g. unknown skill) on failure. We let it propagate so the Agent
            # applies its own tool-failure policy.
            body, bundled = self._store.load_skill(name)
            if bundled:
                manifest = "\n".join(f"- {path}" for path in bundled)
                body = f"{body}\n\nBundled files (read with `read_skill_file`):\n{manifest}"
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
            # The store raises an actionable error (e.g. unknown skill) on failure. We let it propagate so the Agent
            # applies its own tool-failure policy.
            return self._store.read_skill_file(name, path)

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
