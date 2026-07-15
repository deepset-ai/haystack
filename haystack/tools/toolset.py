# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import copy
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.tools.tool import Tool, _check_duplicate_tool_names


@dataclass
class Toolset:
    """
    A collection of related Tools that can be used and managed as a cohesive unit.

    Toolset serves two main purposes:

    1. Group related tools together:
       Toolset allows you to organize related tools into a single collection, making it easier
       to manage and use them as a unit in Haystack pipelines.

       Example:
    ```python
    from typing import Annotated
    from haystack.tools import tool, Toolset
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator

    # Create tools with the @tool decorator (the recommended way)
    @tool
    def add(a: Annotated[int, "first number"], b: Annotated[int, "second number"]) -> int:
        '''Add two numbers.'''
        return a + b

    @tool
    def subtract(a: Annotated[int, "first number"], b: Annotated[int, "second number"]) -> int:
        '''Subtract b from a.'''
        return a - b

    # Create a toolset with the math tools
    math_toolset = Toolset([add, subtract])

    # Use the toolset with an Agent
    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=math_toolset)
    ```

    2. Base class for dynamic tool loading:
       By subclassing Toolset, you can create implementations that dynamically load tools from external sources like
       OpenAPI URLs, MCP servers, or other resources.

       Example:
    ```python
    from typing import Annotated
    from haystack.core.serialization import generate_qualified_class_name
    from haystack.tools import tool, Toolset
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator

    class CalculatorToolset(Toolset):
        '''A toolset for calculator operations.'''

        def __init__(self) -> None:
            super().__init__(self._create_tools())

        def _create_tools(self):
            # These tools are defined statically for illustration purposes only.
            # In a real-world scenario, you would dynamically load tools from an external source here.
            @tool
            def add(a: Annotated[int, "first number"], b: Annotated[int, "second number"]) -> int:
                '''Add two numbers.'''
                return a + b

            @tool
            def multiply(a: Annotated[int, "first number"], b: Annotated[int, "second number"]) -> int:
                '''Multiply two numbers.'''
                return a * b

            return [add, multiply]

        def to_dict(self):
            return {
                "type": generate_qualified_class_name(type(self)),
                "data": {},  # no data to serialize as we define the tools dynamically
            }

        @classmethod
        def from_dict(cls, data):
            return cls()  # Recreate the tools dynamically during deserialization

    # Create the dynamic toolset and use it with an Agent
    calculator_toolset = CalculatorToolset()
    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=calculator_toolset)
    ```

    Toolset implements the collection interface (__iter__, __contains__, __len__, __getitem__), making it behave like
    a list of Tools. This makes it compatible with components that expect iterable tools, such as Agent or Haystack
    chat generators.

    When implementing a custom Toolset subclass for dynamic tool loading:
    - Perform the dynamic loading in the __init__ method
    - Override to_dict() and from_dict() methods if your tools are defined dynamically
    - Serialize endpoint descriptors rather than tool instances if your tools are loaded from external sources
    """

    # Use field() with default_factory to initialize the list
    tools: list[Tool] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Validate and set up the toolset after initialization.

        This handles the case when tools are provided during initialization.
        """
        # If initialization was done a single Tool, raise an error
        if isinstance(self.tools, Tool):
            raise TypeError("A single Tool cannot be directly passed to Toolset. Please use a list: Toolset([tool])")

        # Check for duplicate tool names in the initial set
        _check_duplicate_tool_names(self.tools)

        # Tracks whether warm_up() has already run so subsequent calls become a no-op.
        self._is_warmed_up = False

        # Optional per-run name filter. When set, iteration only yields tools whose name is in this set.
        # None means no filtering. Set on a per-run spawn(), so it never leaks across runs.
        self._selected_tool_names: set[str] | None = None

    def __iter__(self) -> Iterator[Tool]:
        """
        Return an iterator over the Tools in this Toolset.

        This allows the Toolset to be used wherever a list of Tools is expected. If a name filter is active,
        only the tools whose names are in it are yielded.

        :returns: An iterator yielding Tool instances
        """
        for tool in self.tools:
            if self._selected_tool_names is None or tool.name in self._selected_tool_names:
                yield tool

    def get_selectable_tools(self) -> list[Tool]:
        """
        Return the full set of tools that can be selected by name, ignoring any active name filter.

        This differs from iteration, which yields only the tools currently exposed (and respects the name filter).
        Override this when a Toolset's iteration does not surface every selectable tool, so name-based selection
        can still target the full set.

        Warms up the Toolset first if needed, so lazily loaded tools (those a Toolset fetches in `warm_up()`)
        are available for selection.

        :returns: The list of tools available for name-based selection.
        """
        if not self._is_warmed_up:
            self.warm_up()
        return list(self.tools)

    def spawn(self) -> "Toolset":
        """
        Return an isolated copy of this Toolset for a single run.

        The copy shares this Toolset's read-only state (its tools and any warmed-up resources) but gets fresh
        run-scoped state, so concurrent runs that share the same configured Toolset don't corrupt each other (for
        example, one run's name selection leaking into another). Warms up first if needed so the copy shares the
        warmed state. Subclasses with additional run-scoped state should override this.

        :returns: A run-scoped copy of this Toolset.
        """
        if not self._is_warmed_up:
            self.warm_up()
        new = copy.copy(self)
        new._selected_tool_names = None
        return new

    def __contains__(self, item: str | Tool) -> bool:
        """
        Check if a tool is in this Toolset.

        Supports checking by:
        - Tool instance: tool in toolset
        - Tool name: "tool_name" in toolset

        :param item: Tool instance or tool name string
        :returns: True if contained, False otherwise
        """
        if isinstance(item, str):
            return any(tool.name == item for tool in self)
        if isinstance(item, Tool):
            return any(tool is item or tool == item for tool in self)
        return False

    def warm_up(self) -> None:
        """
        Prepare the Toolset for use.

        By default, this method iterates through and warms up all tools in the Toolset.
        Subclasses can override this method to customize initialization behavior, such as:

        - Setting up shared resources (database connections, HTTP sessions) instead of
          warming individual tools
        - Implementing custom initialization logic for dynamically loaded tools
        - Controlling when and how tools are initialized

        For example, a Toolset that manages tools from an external service (like MCPToolset)
        might override this to initialize a shared connection rather than warming up
        individual tools:

        ```python
        class MCPToolset(Toolset):
            def warm_up(self) -> None:
                # Only warm up the shared MCP connection, not individual tools
                self.mcp_connection = establish_connection(self.server_url)
        ```

        This method is idempotent: it only warms up the tools the first time it is called.
        Subclasses overriding it should preserve this contract (for example by guarding on
        `self._is_warmed_up`).
        """
        if self._is_warmed_up:
            return
        for tool in self.tools:
            if hasattr(tool, "warm_up"):
                tool.warm_up()
        self._is_warmed_up = True

    def add(self, tool: "Tool | Toolset") -> None:
        """
        Add a new Tool or merge another Toolset.

        If this Toolset has already been warmed up, the newly added Tool (or the tools of the
        added Toolset) are warmed up immediately so they are ready to use without requiring a
        second `warm_up()` call on the whole Toolset.

        Note: adding a Toolset flattens it into its individual tools, so this is only recommended
        for Toolsets that don't manage shared resources in their `warm_up()` (or `__init__`).
        For example, combining with an `MCPToolset`, which owns a shared connection, is not
        recommended: the connection's lifecycle would no longer be managed by the original
        Toolset. In those cases combine Toolsets with `+` (which preserves each Toolset as a
        unit via `_ToolsetWrapper`) instead.

        :param tool: A Tool instance or another Toolset to add
        :raises ValueError: If adding the tool would result in duplicate tool names
        :raises TypeError: If the provided object is not a Tool or Toolset
        """
        if not isinstance(tool, (Tool, Toolset)):
            raise TypeError(f"Expected Tool or Toolset, got {type(tool).__name__}")

        # Warm up the source before flattening so that lazily-loaded toolsets (e.g. MCPToolset)
        # expose their tools, and so newly added tools are ready to use right away.
        if self._is_warmed_up and hasattr(tool, "warm_up"):
            tool.warm_up()

        new_tools = [tool] if isinstance(tool, Tool) else list(tool)

        # Check for duplicates before adding
        combined_tools = self.tools + new_tools
        _check_duplicate_tool_names(combined_tools)

        self.tools.extend(new_tools)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the Toolset to a dictionary.

        :returns: A dictionary representation of the Toolset

        Note for subclass implementers:
        The default implementation is ideal for scenarios where Tool resolution is static. However, if your subclass
        of Toolset dynamically resolves Tool instances from external sources—such as an MCP server, OpenAPI URL, or
        a local OpenAPI specification—you should consider serializing the endpoint descriptor instead of the Tool
        instances themselves. This strategy preserves the dynamic nature of your Toolset and minimizes the overhead
        associated with serializing potentially large collections of Tool objects. Moreover, by serializing the
        descriptor, you ensure that the deserialization process can accurately reconstruct the Tool instances, even
        if they have been modified or removed since the last serialization. Failing to serialize the descriptor may
        lead to issues where outdated or incorrect Tool configurations are loaded, potentially causing errors or
        unexpected behavior.
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"tools": [tool.to_dict() for tool in self.tools]},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Toolset":
        """
        Deserialize a Toolset from a dictionary.

        :param data: Dictionary representation of the Toolset
        :returns: A new Toolset instance
        """
        inner_data = data["data"]
        tools_data = inner_data.get("tools", [])

        tools = []
        for tool_data in tools_data:
            tool_class = import_class_by_name(tool_data["type"])
            if not issubclass(tool_class, Tool):
                raise TypeError(f"Class '{tool_class}' is not a subclass of Tool")
            tools.append(tool_class.from_dict(tool_data))

        return cls(tools=tools)

    def __add__(self, other: "Tool | Toolset | list[Tool]") -> "Toolset":
        """
        Concatenate this Toolset with another Tool, Toolset, or list of Tools.

        :param other: Another Tool, Toolset, or list of Tools to concatenate
        :returns: A new Toolset containing all tools
        :raises TypeError: If the other parameter is not a Tool, Toolset, or list of Tools
        :raises ValueError: If the combination would result in duplicate tool names
        """
        if isinstance(other, Tool):
            return Toolset(tools=self.tools + [other])
        if isinstance(other, Toolset):
            return _ToolsetWrapper([self, other])
        if isinstance(other, list) and all(isinstance(item, Tool) for item in other):
            return Toolset(tools=self.tools + other)
        raise TypeError(f"Cannot add {type(other).__name__} to Toolset")

    def __len__(self) -> int:
        """
        Return the number of Tools in this Toolset (respecting any active name filter).

        :returns: Number of Tools
        """
        return sum(1 for _ in self)

    def __getitem__(self, index: int) -> Tool:
        """
        Get a Tool by index (respecting any active name filter).

        :param index: Index of the Tool to get
        :returns: The Tool at the specified index
        """
        return list(self)[index]


class _ToolsetWrapper(Toolset):
    """
    A wrapper that holds multiple toolsets and provides a unified interface.

    This is used internally when combining different types of toolsets to preserve
    their individual configurations while still being usable with Agent and Haystack chat generators.
    """

    def __init__(self, toolsets: list[Toolset]) -> None:
        super().__init__([tool for toolset in toolsets for tool in toolset])
        self.toolsets = toolsets
        # Tracks whether warm_up() has already run so subsequent calls become a no-op.
        self._is_warmed_up = False

    def __iter__(self) -> Iterator[Tool]:
        """Iterate over all tools from all toolsets, honoring any active name filter."""
        for toolset in self.toolsets:
            for tool in toolset:
                if self._selected_tool_names is None or tool.name in self._selected_tool_names:
                    yield tool

    def get_selectable_tools(self) -> list[Tool]:
        """Return every selectable tool across all wrapped toolsets, ignoring any active filter."""
        return [tool for toolset in self.toolsets for tool in toolset.get_selectable_tools()]

    def spawn(self) -> "_ToolsetWrapper":
        """Return an isolated copy with each wrapped toolset spawned."""
        return _ToolsetWrapper([toolset.spawn() for toolset in self.toolsets])

    def __contains__(self, item: Any) -> bool:
        """Check if a tool is in any of the toolsets."""
        return any(item in toolset for toolset in self.toolsets)

    def warm_up(self) -> None:
        """
        Warm up all wrapped toolsets.

        This method is idempotent: it only warms up the wrapped toolsets the first time it is
        called. The individual toolsets are themselves expected to have idempotent `warm_up()`
        methods.
        """
        if self._is_warmed_up:
            return
        for toolset in self.toolsets:
            toolset.warm_up()
        self._is_warmed_up = True

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the wrapper to a dictionary.

        Each wrapped toolset is serialized via its own `to_dict()`, so any subclass that
        overrides serialization (e.g. a toolset that serializes a connection/endpoint
        descriptor) is preserved.

        :returns: A dictionary representation of the wrapper.
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"toolsets": [toolset.to_dict() for toolset in self.toolsets]},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_ToolsetWrapper":
        """
        Deserialize a wrapper from a dictionary.

        :param data: Dictionary representation of the wrapper.
        :returns: A new `_ToolsetWrapper` instance.
        :raises TypeError: If any serialized entry is not a subclass of Toolset.
        """
        inner_data = data["data"]
        toolsets_data = inner_data.get("toolsets", [])

        toolsets = []
        for toolset_data in toolsets_data:
            toolset_class = import_class_by_name(toolset_data["type"])
            if not issubclass(toolset_class, Toolset):
                raise TypeError(f"Class '{toolset_class}' is not a subclass of Toolset")
            toolsets.append(toolset_class.from_dict(toolset_data))

        return cls(toolsets=toolsets)

    def __len__(self) -> int:
        """Return total number of tools across all toolsets (respecting any active name filter)."""
        return sum(1 for _ in self)

    def __getitem__(self, index: int) -> Tool:
        """Get a tool by index across all toolsets."""
        # Leverage iteration instead of manual index tracking
        for i, tool in enumerate(self):
            if i == index:
                return tool
        raise IndexError("ToolsetWrapper index out of range")

    def __add__(self, other: Toolset | Tool | list[Tool]) -> "_ToolsetWrapper":
        """Add another toolset or tool to this wrapper."""
        if isinstance(other, Toolset):
            return _ToolsetWrapper(self.toolsets + [other])
        if isinstance(other, Tool):
            return _ToolsetWrapper(self.toolsets + [Toolset([other])])
        if isinstance(other, list) and all(isinstance(item, Tool) for item in other):
            return _ToolsetWrapper(self.toolsets + [Toolset(other)])
        raise TypeError(f"Cannot add {type(other).__name__} to _ToolsetWrapper")
