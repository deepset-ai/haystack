# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Union

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
       from haystack.tools import Tool, Toolset
       from haystack.components.tools import ToolInvoker

       # Define math functions
       def add_numbers(a: int, b: int) -> int:
           return a + b

       def subtract_numbers(a: int, b: int) -> int:
           return a - b

       # Create tools with proper schemas
       add_tool = Tool(
           name="add",
           description="Add two numbers",
           parameters={
               "type": "object",
               "properties": {
                   "a": {"type": "integer"},
                   "b": {"type": "integer"}
               },
               "required": ["a", "b"]
           },
           function=add_numbers
       )

       subtract_tool = Tool(
           name="subtract",
           description="Subtract b from a",
           parameters={
               "type": "object",
               "properties": {
                   "a": {"type": "integer"},
                   "b": {"type": "integer"}
               },
               "required": ["a", "b"]
           },
           function=subtract_numbers
       )

       # Create a toolset with the math tools
       math_toolset = Toolset([add_tool, subtract_tool])

       # Use the toolset with a ToolInvoker or ChatGenerator component
       invoker = ToolInvoker(tools=math_toolset)
       ```

    2. Base class for dynamic tool loading:
       By subclassing Toolset, you can create implementations that dynamically load tools
       from external sources like OpenAPI URLs, MCP servers, or other resources.

       Example:
       ```python
       from haystack.core.serialization import generate_qualified_class_name
       from haystack.tools import Tool, Toolset
       from haystack.components.tools import ToolInvoker

       class CalculatorToolset(Toolset):
           '''A toolset for calculator operations.'''

           def __init__(self):
               tools = self._create_tools()
               super().__init__(tools)

           def _create_tools(self):
               # These Tool instances are obviously defined statically and for illustration purposes only.
               # In a real-world scenario, you would dynamically load tools from an external source here.
               tools = []
               add_tool = Tool(
                   name="add",
                   description="Add two numbers",
                   parameters={
                       "type": "object",
                       "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                       "required": ["a", "b"],
                   },
                   function=lambda a, b: a + b,
               )

               multiply_tool = Tool(
                   name="multiply",
                   description="Multiply two numbers",
                   parameters={
                       "type": "object",
                       "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                       "required": ["a", "b"],
                   },
                   function=lambda a, b: a * b,
               )

               tools.append(add_tool)
               tools.append(multiply_tool)

               return tools

           def to_dict(self):
               return {
                   "type": generate_qualified_class_name(type(self)),
                   "data": {},  # no data to serialize as we define the tools dynamically
               }

           @classmethod
           def from_dict(cls, data):
               return cls()  # Recreate the tools dynamically during deserialization

       # Create the dynamic toolset and use it with ToolInvoker
       calculator_toolset = CalculatorToolset()
       invoker = ToolInvoker(tools=calculator_toolset)
       ```

    Toolset implements the collection interface (__iter__, __contains__, __len__, __getitem__),
    making it behave like a list of Tools. This makes it compatible with components that expect
    iterable tools, such as ToolInvoker or Haystack chat generators.

    When implementing a custom Toolset subclass for dynamic tool loading:
    - Perform the dynamic loading in the __init__ method
    - Override to_dict() and from_dict() methods if your tools are defined dynamically
    - Serialize endpoint descriptors rather than tool instances if your tools
      are loaded from external sources
    """

    # Use field() with default_factory to initialize the list
    tools: List[Tool] = field(default_factory=list)

    def __post_init__(self):
        """
        Validate and set up the toolset after initialization.

        This handles the case when tools are provided during initialization.
        """
        # If initialization was done a single Tool, raise an error
        if isinstance(self.tools, Tool):
            raise TypeError("A single Tool cannot be directly passed to Toolset. Please use a list: Toolset([tool])")

        # Check for duplicate tool names in the initial set
        _check_duplicate_tool_names(self.tools)

    def __iter__(self) -> Iterator[Tool]:
        """
        Return an iterator over the Tools in this Toolset.

        This allows the Toolset to be used wherever a list of Tools is expected.

        :returns: An iterator yielding Tool instances
        """
        return iter(self.tools)

    def __contains__(self, item: Any) -> bool:
        """
        Check if a tool is in this Toolset.

        Supports checking by:
        - Tool instance: tool in toolset
        - Tool name: "tool_name" in toolset

        :param item: Tool instance or tool name string
        :returns: True if contained, False otherwise
        """
        if isinstance(item, str):
            return any(tool.name == item for tool in self.tools)
        if isinstance(item, Tool):
            return item in self.tools
        return False

    def add(self, tool: Union[Tool, "Toolset"]) -> None:
        """
        Add a new Tool or merge another Toolset.

        :param tool: A Tool instance or another Toolset to add
        :raises ValueError: If adding the tool would result in duplicate tool names
        :raises TypeError: If the provided object is not a Tool or Toolset
        """
        new_tools = []

        if isinstance(tool, Tool):
            new_tools = [tool]
        elif isinstance(tool, Toolset):
            new_tools = list(tool)
        else:
            raise TypeError(f"Expected Tool or Toolset, got {type(tool).__name__}")

        # Check for duplicates before adding
        combined_tools = self.tools + new_tools
        _check_duplicate_tool_names(combined_tools)

        self.tools.extend(new_tools)

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "Toolset":
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

    def __add__(self, other: Union[Tool, "Toolset", List[Tool]]) -> "Toolset":
        """
        Concatenate this Toolset with another Tool, Toolset, or list of Tools.

        :param other: Another Tool, Toolset, or list of Tools to concatenate
        :returns: A new Toolset containing all tools
        :raises TypeError: If the other parameter is not a Tool, Toolset, or list of Tools
        :raises ValueError: If the combination would result in duplicate tool names
        """
        if isinstance(other, Tool):
            combined_tools = self.tools + [other]
        elif isinstance(other, Toolset):
            combined_tools = self.tools + list(other)
        elif isinstance(other, list) and all(isinstance(item, Tool) for item in other):
            combined_tools = self.tools + other
        else:
            raise TypeError(f"Cannot add {type(other).__name__} to Toolset")

        # Check for duplicates
        _check_duplicate_tool_names(combined_tools)

        return Toolset(tools=combined_tools)

    def __len__(self) -> int:
        """
        Return the number of Tools in this Toolset.

        :returns: Number of Tools
        """
        return len(self.tools)

    def __getitem__(self, index):
        """
        Get a Tool by index.

        :param index: Index of the Tool to get
        :returns: The Tool at the specified index
        """
        return self.tools[index]
