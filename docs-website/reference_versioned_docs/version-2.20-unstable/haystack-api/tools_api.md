---
title: "Tools"
id: tools-api
description: "Unified abstractions to represent tools across the framework."
slug: "/tools-api"
---

<a id="tool"></a>

## Module tool

<a id="tool.Tool"></a>

### Tool

Data class representing a Tool that Language Models can prepare a call for.

Accurate definitions of the textual attributes such as `name` and `description`
are important for the Language Model to correctly prepare the call.

For resource-intensive operations like establishing connections to remote services or
loading models, override the `warm_up()` method. This method is called before the Tool
is used and should be idempotent, as it may be called multiple times during
pipeline/agent setup.

**Arguments**:

- `name`: Name of the Tool.
- `description`: Description of the Tool.
- `parameters`: A JSON schema defining the parameters expected by the Tool.
- `function`: The function that will be invoked when the Tool is called.
- `outputs_to_string`: Optional dictionary defining how a tool outputs should be converted into a string.
If the source is provided only the specified output key is sent to the handler.
If the source is omitted the whole tool result is sent to the handler.
Example:
```python
{
    "source": "docs", "handler": format_documents
}
```
- `inputs_from_state`: Optional dictionary mapping state keys to tool parameter names.
Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
- `outputs_to_state`: Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
If the source is provided only the specified output key is sent to the handler.
Example:
```python
{
    "documents": {"source": "docs", "handler": custom_handler}
}
```
If the source is omitted the whole tool result is sent to the handler.
Example:
```python
{
    "documents": {"handler": custom_handler}
}
```

<a id="tool.Tool.tool_spec"></a>

#### Tool.tool\_spec

```python
@property
def tool_spec() -> dict[str, Any]
```

Return the Tool specification to be used by the Language Model.

<a id="tool.Tool.warm_up"></a>

#### Tool.warm\_up

```python
def warm_up() -> None
```

Prepare the Tool for use.

Override this method to establish connections to remote services, load models,
or perform other resource-intensive initialization. This method should be idempotent,
as it may be called multiple times.

<a id="tool.Tool.invoke"></a>

#### Tool.invoke

```python
def invoke(**kwargs: Any) -> Any
```

Invoke the Tool with the provided keyword arguments.

<a id="tool.Tool.to_dict"></a>

#### Tool.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the Tool to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="tool.Tool.from_dict"></a>

#### Tool.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "Tool"
```

Deserializes the Tool from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized Tool.

<a id="from_function"></a>

## Module from\_function

<a id="from_function.create_tool_from_function"></a>

#### create\_tool\_from\_function

```python
def create_tool_from_function(
        function: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        inputs_from_state: Optional[dict[str, str]] = None,
        outputs_to_state: Optional[dict[str, dict[str,
                                                  Any]]] = None) -> "Tool"
```

Create a Tool instance from a function.

Allows customizing the Tool name and description.
For simpler use cases, consider using the `@tool` decorator.

### Usage example

```python
from typing import Annotated, Literal
from haystack.tools import create_tool_from_function

def get_weather(
    city: Annotated[str, "the city for which to get the weather"] = "Munich",
    unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius"):
    '''A simple function to get the current weather for a location.'''
    return f"Weather report for {city}: 20 {unit}, sunny"

tool = create_tool_from_function(get_weather)

print(tool)
>>> Tool(name='get_weather', description='A simple function to get the current weather for a location.',
>>> parameters={
>>> 'type': 'object',
>>> 'properties': {
>>>     'city': {'type': 'string', 'description': 'the city for which to get the weather', 'default': 'Munich'},
>>>     'unit': {
>>>         'type': 'string',
>>>         'enum': ['Celsius', 'Fahrenheit'],
>>>         'description': 'the unit for the temperature',
>>>         'default': 'Celsius',
>>>     },
>>>     }
>>> },
>>> function=<function get_weather at 0x7f7b3a8a9b80>)
```

**Arguments**:

- `function`: The function to be converted into a Tool.
The function must include type hints for all parameters.
The function is expected to have basic python input types (str, int, float, bool, list, dict, tuple).
Other input types may work but are not guaranteed.
If a parameter is annotated using `typing.Annotated`, its metadata will be used as parameter description.
- `name`: The name of the Tool. If not provided, the name of the function will be used.
- `description`: The description of the Tool. If not provided, the docstring of the function will be used.
To intentionally leave the description empty, pass an empty string.
- `inputs_from_state`: Optional dictionary mapping state keys to tool parameter names.
Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
- `outputs_to_state`: Optional dictionary defining how tool outputs map to state and message handling.
Example:
```python
{
    "documents": {"source": "docs", "handler": custom_handler},
    "message": {"source": "summary", "handler": format_summary}
}
```

**Raises**:

- `ValueError`: If any parameter of the function lacks a type hint.
- `SchemaGenerationError`: If there is an error generating the JSON schema for the Tool.

**Returns**:

The Tool created from the function.

<a id="from_function.tool"></a>

#### tool

```python
def tool(
    function: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    inputs_from_state: Optional[dict[str, str]] = None,
    outputs_to_state: Optional[dict[str, dict[str, Any]]] = None
) -> Union[Tool, Callable[[Callable], Tool]]
```

Decorator to convert a function into a Tool.

Can be used with or without parameters:
@tool  # without parameters
def my_function(): ...

@tool(name="custom_name")  # with parameters
def my_function(): ...

### Usage example
```python
from typing import Annotated, Literal
from haystack.tools import tool

@tool
def get_weather(
    city: Annotated[str, "the city for which to get the weather"] = "Munich",
    unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius"):
    '''A simple function to get the current weather for a location.'''
    return f"Weather report for {city}: 20 {unit}, sunny"

print(get_weather)
>>> Tool(name='get_weather', description='A simple function to get the current weather for a location.',
>>> parameters={
>>> 'type': 'object',
>>> 'properties': {
>>>     'city': {'type': 'string', 'description': 'the city for which to get the weather', 'default': 'Munich'},
>>>     'unit': {
>>>         'type': 'string',
>>>         'enum': ['Celsius', 'Fahrenheit'],
>>>         'description': 'the unit for the temperature',
>>>         'default': 'Celsius',
>>>     },
>>>     }
>>> },
>>> function=<function get_weather at 0x7f7b3a8a9b80>)
```

**Arguments**:

- `function`: The function to decorate (when used without parameters)
- `name`: Optional custom name for the tool
- `description`: Optional custom description
- `inputs_from_state`: Optional dictionary mapping state keys to tool parameter names
- `outputs_to_state`: Optional dictionary defining how tool outputs map to state and message handling

**Returns**:

Either a Tool instance or a decorator function that will create one

<a id="component_tool"></a>

## Module component\_tool

<a id="component_tool.ComponentTool"></a>

### ComponentTool

A Tool that wraps Haystack components, allowing them to be used as tools by LLMs.

ComponentTool automatically generates LLM-compatible tool schemas from component input sockets,
which are derived from the component's `run` method signature and type hints.


Key features:
- Automatic LLM tool calling schema generation from component input sockets
- Type conversion and validation for component inputs
- Support for types:
- Dataclasses
- Lists of dataclasses
- Basic types (str, int, float, bool, dict)
- Lists of basic types
- Automatic name generation from component class name
- Description extraction from component docstrings

To use ComponentTool, you first need a Haystack component - either an existing one or a new one you create.
You can create a ComponentTool from the component by passing the component to the ComponentTool constructor.
Below is an example of creating a ComponentTool from an existing SerperDevWebSearch component.

## Usage Example:

```python
from haystack import component, Pipeline
from haystack.tools import ComponentTool
from haystack.components.websearch import SerperDevWebSearch
from haystack.utils import Secret
from haystack.components.tools.tool_invoker import ToolInvoker
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

# Create a SerperDev search component
search = SerperDevWebSearch(api_key=Secret.from_env_var("SERPERDEV_API_KEY"), top_k=3)

# Create a tool from the component
tool = ComponentTool(
    component=search,
    name="web_search",  # Optional: defaults to "serper_dev_web_search"
    description="Search the web for current information on any topic"  # Optional: defaults to component docstring
)

# Create pipeline with OpenAIChatGenerator and ToolInvoker
pipeline = Pipeline()
pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))

# Connect components
pipeline.connect("llm.replies", "tool_invoker.messages")

message = ChatMessage.from_user("Use the web search tool to find information about Nikola Tesla")

# Run pipeline
result = pipeline.run({"llm": {"messages": [message]}})

print(result)
```

<a id="component_tool.ComponentTool.__init__"></a>

#### ComponentTool.\_\_init\_\_

```python
def __init__(
    component: Component,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[dict[str, Any]] = None,
    *,
    outputs_to_string: Optional[dict[str, Union[str, Callable[[Any],
                                                              str]]]] = None,
    inputs_from_state: Optional[dict[str, str]] = None,
    outputs_to_state: Optional[dict[str, dict[str, Union[str,
                                                         Callable]]]] = None
) -> None
```

Create a Tool instance from a Haystack component.

**Arguments**:

- `component`: The Haystack component to wrap as a tool.
- `name`: Optional name for the tool (defaults to snake_case of component class name).
- `description`: Optional description (defaults to component's docstring).
- `parameters`: A JSON schema defining the parameters expected by the Tool.
Will fall back to the parameters defined in the component's run method signature if not provided.
- `outputs_to_string`: Optional dictionary defining how a tool outputs should be converted into a string.
If the source is provided only the specified output key is sent to the handler.
If the source is omitted the whole tool result is sent to the handler.
Example:
```python
{
    "source": "docs", "handler": format_documents
}
```
- `inputs_from_state`: Optional dictionary mapping state keys to tool parameter names.
Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
- `outputs_to_state`: Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
If the source is provided only the specified output key is sent to the handler.
Example:
```python
{
    "documents": {"source": "docs", "handler": custom_handler}
}
```
If the source is omitted the whole tool result is sent to the handler.
Example:
```python
{
    "documents": {"handler": custom_handler}
}
```

**Raises**:

- `ValueError`: If the component is invalid or schema generation fails.

<a id="component_tool.ComponentTool.warm_up"></a>

#### ComponentTool.warm\_up

```python
def warm_up()
```

Prepare the ComponentTool for use.

<a id="component_tool.ComponentTool.to_dict"></a>

#### ComponentTool.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the ComponentTool to a dictionary.

<a id="component_tool.ComponentTool.from_dict"></a>

#### ComponentTool.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ComponentTool"
```

Deserializes the ComponentTool from a dictionary.

<a id="component_tool.ComponentTool.tool_spec"></a>

#### ComponentTool.tool\_spec

```python
@property
def tool_spec() -> dict[str, Any]
```

Return the Tool specification to be used by the Language Model.

<a id="component_tool.ComponentTool.invoke"></a>

#### ComponentTool.invoke

```python
def invoke(**kwargs: Any) -> Any
```

Invoke the Tool with the provided keyword arguments.

<a id="toolset"></a>

## Module toolset

<a id="toolset.Toolset"></a>

### Toolset

A collection of related Tools that can be used and managed as a cohesive unit.

Toolset serves two main purposes:

1. Group related tools together:
Toolset allows you to organize related tools into a single collection, making it easier
to manage and use them as a unit in Haystack pipelines.

**Example**:

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


**Example**:

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

<a id="toolset.Toolset.__post_init__"></a>

#### Toolset.\_\_post\_init\_\_

```python
def __post_init__()
```

Validate and set up the toolset after initialization.

This handles the case when tools are provided during initialization.

<a id="toolset.Toolset.__iter__"></a>

#### Toolset.\_\_iter\_\_

```python
def __iter__() -> Iterator[Tool]
```

Return an iterator over the Tools in this Toolset.

This allows the Toolset to be used wherever a list of Tools is expected.

**Returns**:

An iterator yielding Tool instances

<a id="toolset.Toolset.__contains__"></a>

#### Toolset.\_\_contains\_\_

```python
def __contains__(item: Any) -> bool
```

Check if a tool is in this Toolset.

Supports checking by:
- Tool instance: tool in toolset
- Tool name: "tool_name" in toolset

**Arguments**:

- `item`: Tool instance or tool name string

**Returns**:

True if contained, False otherwise

<a id="toolset.Toolset.warm_up"></a>

#### Toolset.warm\_up

```python
def warm_up() -> None
```

Prepare the Toolset for use.

Override this method to set up shared resources like database connections or HTTP sessions.
This method should be idempotent, as it may be called multiple times.

<a id="toolset.Toolset.add"></a>

#### Toolset.add

```python
def add(tool: Union[Tool, "Toolset"]) -> None
```

Add a new Tool or merge another Toolset.

**Arguments**:

- `tool`: A Tool instance or another Toolset to add

**Raises**:

- `ValueError`: If adding the tool would result in duplicate tool names
- `TypeError`: If the provided object is not a Tool or Toolset

<a id="toolset.Toolset.to_dict"></a>

#### Toolset.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the Toolset to a dictionary.

**Returns**:

A dictionary representation of the Toolset
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

<a id="toolset.Toolset.from_dict"></a>

#### Toolset.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "Toolset"
```

Deserialize a Toolset from a dictionary.

**Arguments**:

- `data`: Dictionary representation of the Toolset

**Returns**:

A new Toolset instance

<a id="toolset.Toolset.__add__"></a>

#### Toolset.\_\_add\_\_

```python
def __add__(other: Union[Tool, "Toolset", list[Tool]]) -> "Toolset"
```

Concatenate this Toolset with another Tool, Toolset, or list of Tools.

**Arguments**:

- `other`: Another Tool, Toolset, or list of Tools to concatenate

**Raises**:

- `TypeError`: If the other parameter is not a Tool, Toolset, or list of Tools
- `ValueError`: If the combination would result in duplicate tool names

**Returns**:

A new Toolset containing all tools

<a id="toolset.Toolset.__len__"></a>

#### Toolset.\_\_len\_\_

```python
def __len__() -> int
```

Return the number of Tools in this Toolset.

**Returns**:

Number of Tools

<a id="toolset.Toolset.__getitem__"></a>

#### Toolset.\_\_getitem\_\_

```python
def __getitem__(index)
```

Get a Tool by index.

**Arguments**:

- `index`: Index of the Tool to get

**Returns**:

The Tool at the specified index

<a id="toolset._ToolsetWrapper"></a>

### \_ToolsetWrapper

A wrapper that holds multiple toolsets and provides a unified interface.

This is used internally when combining different types of toolsets to preserve
their individual configurations while still being usable with ToolInvoker.

<a id="toolset._ToolsetWrapper.__iter__"></a>

#### \_ToolsetWrapper.\_\_iter\_\_

```python
def __iter__()
```

Iterate over all tools from all toolsets.

<a id="toolset._ToolsetWrapper.__contains__"></a>

#### \_ToolsetWrapper.\_\_contains\_\_

```python
def __contains__(item)
```

Check if a tool is in any of the toolsets.

<a id="toolset._ToolsetWrapper.warm_up"></a>

#### \_ToolsetWrapper.warm\_up

```python
def warm_up()
```

Warm up all toolsets.

<a id="toolset._ToolsetWrapper.__len__"></a>

#### \_ToolsetWrapper.\_\_len\_\_

```python
def __len__()
```

Return total number of tools across all toolsets.

<a id="toolset._ToolsetWrapper.__getitem__"></a>

#### \_ToolsetWrapper.\_\_getitem\_\_

```python
def __getitem__(index)
```

Get a tool by index across all toolsets.

<a id="toolset._ToolsetWrapper.__add__"></a>

#### \_ToolsetWrapper.\_\_add\_\_

```python
def __add__(other)
```

Add another toolset or tool to this wrapper.

<a id="toolset._ToolsetWrapper.__post_init__"></a>

#### \_ToolsetWrapper.\_\_post\_init\_\_

```python
def __post_init__()
```

Validate and set up the toolset after initialization.

This handles the case when tools are provided during initialization.

<a id="toolset._ToolsetWrapper.add"></a>

#### \_ToolsetWrapper.add

```python
def add(tool: Union[Tool, "Toolset"]) -> None
```

Add a new Tool or merge another Toolset.

**Arguments**:

- `tool`: A Tool instance or another Toolset to add

**Raises**:

- `ValueError`: If adding the tool would result in duplicate tool names
- `TypeError`: If the provided object is not a Tool or Toolset

<a id="toolset._ToolsetWrapper.to_dict"></a>

#### \_ToolsetWrapper.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the Toolset to a dictionary.

**Returns**:

A dictionary representation of the Toolset
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

<a id="toolset._ToolsetWrapper.from_dict"></a>

#### \_ToolsetWrapper.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "Toolset"
```

Deserialize a Toolset from a dictionary.

**Arguments**:

- `data`: Dictionary representation of the Toolset

**Returns**:

A new Toolset instance
