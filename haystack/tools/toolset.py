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
    A collection of related Tools that can be bootstrapped collectively and managed as a cohesive unit.

    1. It provides a way to group related tools together, making it easier to manage and pass them around as a single
       unit.
    2. It provides a base class for implementing dynamic tool loading - by subclassing Toolset, you can create
       custom implementations that load tools from resource endpoints (e.g., an API endpoint or a configuration file).
       This allows you to define tools externally and load them at runtime, rather than defining each tool explicitly in
       code.


    Note: To implement dynamic tool loading, you must create a subclass of Toolset that implements the loading logic.
    The base Toolset class provides the infrastructure (like registration, deduplication, and serialization), but the
    actual loading from an endpoint must be implemented in your subclass.

    Important: When implementing a custom Toolset subclass, you should perform the dynamic loading of tools in the
    `__init__` method. This ensures tools are loaded and added when the Toolset is instantiated. The base class
    will automatically handle tool registration and deduplication. For example:

    ```python
    class CustomToolset(Toolset):
        def __init__(self, api_endpoint: str):
            # Always call parent's __init__ first to initialize the tools list
            super().__init__([])
            self.api_endpoint = api_endpoint
            # Load and add tools during initialization
            self._load_tools_from_endpoint()

        def _load_tools_from_endpoint(self):
            # Fetch tool definitions from the endpoint
            tool_definitions = self._fetch_from_endpoint()
            # Create and add each tool
            for definition in tool_definitions:
                tool = Tool(
                    name=definition["name"],
                    description=definition["description"],
                    parameters=definition["parameters"],
                    function=self._create_tool_function(definition)
                )
                # add handles adding to internal list and deduplication
                self.add(tool)
    ```

    Toolset implements the __iter__, __contains__, and __len__ methods, making it behave like a collection.
    This makes it compatible with any component that expects iterable tools, such as ToolInvoker or
    any of the Haystack chat generators.

    Example:
    ```python
    from haystack import Pipeline
    from haystack.tools import Tool, Toolset
    from haystack.components.tools import ToolInvoker
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.converters import OutputAdapter
    from haystack.dataclasses import ChatMessage

    # Define a simple math function
    def add_numbers(a: int, b: int) -> int:
        return a + b

    # Create a tool with proper schema
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

    # Create a toolset with the math tool
    math_toolset = Toolset([add_tool])

    # Create a complete pipeline that can use the toolset
    pipeline = Pipeline()
    pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-3.5-turbo", tools=math_toolset))
    pipeline.add_component("tool_invoker", ToolInvoker(tools=math_toolset))
    pipeline.add_component(
        "adapter",
        OutputAdapter(
            template="{{ initial_msg + initial_tool_messages + tool_messages }}",
            output_type=list[ChatMessage],
            unsafe=True,
        ),
    )
    pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-3.5-turbo"))

    # Connect the components
    pipeline.connect("llm.replies", "tool_invoker.messages")
    pipeline.connect("llm.replies", "adapter.initial_tool_messages")
    pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
    pipeline.connect("adapter.output", "response_llm.messages")

    # Use the pipeline with the toolset
    user_input = "What is 5 plus 3?"
    user_input_msg = ChatMessage.from_user(text=user_input)
    result = pipeline.run({"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}})
    # Result will contain "8" in the response
    ```
    """

    # Use field() with default_factory to initialize the list
    tools: List[Tool] = field(default_factory=list)

    def __post_init__(self):
        """
        Validate and set up the toolset after initialization.

        This handles the case when tools are provided during initialization.
        """
        # If initialization was done a single Tool, convert it to a list
        if isinstance(self.tools, Tool):
            self.tools = [self.tools]

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
