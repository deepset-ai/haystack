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
    `__init__` method. This ensures tools are loaded and registered when the Toolset is instantiated. The base class
    will automatically handle tool registration and deduplication. For example:

    ```python
    class CustomToolset(Toolset):
        def __init__(self, api_endpoint: str):
            # Always call parent's __init__ first to initialize the tools list
            super().__init__([])
            self.api_endpoint = api_endpoint
            # Load and register tools during initialization
            self._load_tools_from_endpoint()

        def _load_tools_from_endpoint(self):
            # Fetch tool definitions from the endpoint
            tool_definitions = self._fetch_from_endpoint()
            # Create and register each tool
            for definition in tool_definitions:
                tool = Tool(
                    name=definition["name"],
                    description=definition["description"],
                    parameters=definition["parameters"],
                    function=self._create_tool_function(definition)
                )
                # register_tool handles adding to internal list and deduplication
                self.register_tool(tool)
    ```

    Toolset implements the Iterable protocol, making it compatible with any component that accepts a list of Tools,
    such as ToolInvoker or any of the Haystack chat generators.

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
    pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-3.5-turbo", tools=list(math_toolset)))
    pipeline.add_component("tool_invoker", ToolInvoker(tools=list(math_toolset)))
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

    # Example of a custom toolset that loads tools dynamically:
    class CalculatorToolset(Toolset):
        def __init__(self):
            # Always initialize parent first
            super().__init__([])
            # In a real implementation, you would load tool definitions from your endpoint here
            self._create_tools()

        def _create_tools(self):
            # This is where you would implement the actual loading logic, for example:
            # - Fetching tool definitions from a REST API
            # - Reading from a configuration file
            # - Loading from an MCP server
            # For this example, we'll create a tool directly:
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
            # register_tool handles adding to the internal list and deduplication
            self.register_tool(add_tool)
    ```

    The Toolset class is particularly useful when:
    - You have a collection of related tools that should be managed together
    - You want to implement custom tool loading logic by subclassing Toolset
    - You need to combine multiple toolsets into a larger set of tools
    - You want to ensure there are no naming conflicts between tools
    - You need to serialize/deserialize a collection of tools as a unit
    """

    # Use field() with default_factory to initialize the list
    tools: List[Tool] = field(default_factory=list)

    def __post_init__(self):
        """
        Validate and set up the toolset after initialization.

        This handles the case when tools are provided during initialization.
        """
        # If initialization was done with a Toolset, list of Tools, or a single Tool
        if isinstance(self.tools, Toolset):
            self.tools = list(self.tools)
        elif isinstance(self.tools, Tool):
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

    def register_tool(self, tool: Union[Tool, "Toolset"]) -> None:
        """
        Register a new Tool or merge another Toolset.

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
