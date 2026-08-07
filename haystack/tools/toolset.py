# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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

       When implementing a custom Toolset subclass for dynamic tool loading:
       - Load the tools in `warm_up()` and assign them to `self.tools`. Following the framework-wide `warm_up()`
         convention, make it idempotent by guarding on your own state (e.g. `if self._client is not None: return`),
         as it may be called before every run.
       - Override `to_dict()` and `from_dict()` to serialize the endpoint descriptor (URL, server info) rather than
         the dynamically loaded Tool instances.

       Example:
    ```python
    from haystack.core.serialization import generate_qualified_class_name
    from haystack.tools import Toolset

    class RemoteServiceToolset(Toolset):
        def __init__(self, endpoint: str) -> None:
            self.endpoint = endpoint
            self._client = None
            super().__init__(tools=[])  # tools are loaded on warm_up()

        def warm_up(self) -> None:
            if self._client is not None:
                return
            self._client = connect(self.endpoint)
            self.tools = self._client.fetch_tools()

        def to_dict(self):
            return {
                "type": generate_qualified_class_name(type(self)),
                "data": {"endpoint": self.endpoint},
            }

        @classmethod
        def from_dict(cls, data):
            return cls(endpoint=data["data"]["endpoint"])
    ```

    Toolset implements the collection interface (__iter__, __contains__, __len__, __getitem__), making it behave like
    a list of Tools. This makes it compatible with components that expect iterable tools, such as Agent or Haystack
    chat generators.

    To combine multiple Toolsets, pass them as a list wherever tools are accepted, e.g.
    `Agent(tools=[toolset_a, toolset_b])`. This keeps each Toolset as a unit, preserving its lifecycle and
    serialization.
    """

    tools: list[Tool] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Validate the tools provided during initialization.
        """
        if isinstance(self.tools, Tool):
            raise TypeError("A single Tool cannot be directly passed to Toolset. Please use a list: Toolset([tool])")

        _check_duplicate_tool_names(self.tools)

    def __iter__(self) -> Iterator[Tool]:
        """
        Return an iterator over the Tools in this Toolset.

        :returns: An iterator yielding Tool instances
        """
        return iter(self.tools)

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

    def __len__(self) -> int:
        """
        Return the number of Tools in this Toolset.

        :returns: Number of Tools
        """
        return sum(1 for _ in self)

    def __getitem__(self, index: int) -> Tool:
        """
        Get a Tool by index.

        :param index: Index of the Tool to get
        :returns: The Tool at the specified index
        """
        return list(self)[index]

    def warm_up(self) -> None:
        """
        Prepare the Toolset for use.

        By default, this method warms up all tools in the Toolset. Subclasses that load tools dynamically
        (e.g. from an MCP server or an OpenAPI spec) should override this method to fetch their tools and assign
        them to `self.tools`.

        Following the framework-wide convention, `warm_up()` may be called multiple times (e.g. before every run)
        and implementations are responsible for making it idempotent. The default implementation delegates to the
        tools' own idempotent `warm_up()`. Subclasses should guard on their own state, for example
        `if self._client is not None: return`.
        """
        for tool in self.tools:
            if hasattr(tool, "warm_up"):
                tool.warm_up()

    def add(self, tool: Tool) -> None:
        """
        Add a new Tool to this Toolset.

        To combine whole Toolsets, pass them as a list wherever tools are accepted instead, e.g.
        `Agent(tools=[toolset_a, toolset_b])`. This keeps each Toolset as a unit, preserving its lifecycle
        and serialization.

        :param tool: A Tool instance to add
        :raises ValueError: If adding the tool would result in duplicate tool names
        :raises TypeError: If the provided object is not a Tool
        """
        if not isinstance(tool, Tool):
            raise TypeError(f"Expected Tool, got {type(tool).__name__}")

        _check_duplicate_tool_names(self.tools + [tool])
        self.tools.append(tool)

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
