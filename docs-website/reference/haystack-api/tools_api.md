---
title: "Tools"
id: tools-api
description: "Unified abstractions to represent tools across the framework."
slug: "/tools-api"
---


## `haystack.tools.component_tool`

### `haystack.tools.component_tool.ComponentTool`

Bases: <code>Tool</code>

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
pipeline.add_component("llm", OpenAIChatGenerator(tools=[tool]))
pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))

# Connect components
pipeline.connect("llm.replies", "tool_invoker.messages")

message = ChatMessage.from_user("Use the web search tool to find information about Nikola Tesla")

# Run pipeline
result = pipeline.run({"llm": {"messages": [message]}})

print(result)
```

#### `__init__`

```python
__init__(
    component: Component,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    *,
    outputs_to_string: dict[str, str | Callable[[Any], str]] | None = None,
    inputs_from_state: dict[str, str] | None = None,
    outputs_to_state: dict[str, dict[str, str | Callable]] | None = None
) -> None
```

Create a Tool instance from a Haystack component.

**Parameters:**

- **component** (<code>Component</code>) – The Haystack component to wrap as a tool.
- **name** (<code>str | None</code>) – Optional name for the tool (defaults to snake_case of component class name).
- **description** (<code>str | None</code>) – Optional description (defaults to component's docstring).
- **parameters** (<code>dict\[str, Any\] | None</code>) – A JSON schema defining the parameters expected by the Tool.
  Will fall back to the parameters defined in the component's run method signature if not provided.
- **outputs_to_string** (<code>dict\[str, str | Callable\\[[Any\], str\]\] | None</code>) – Optional dictionary defining how tool outputs should be converted into string(s) or results.
  If not provided, the tool result is converted to a string using a default handler.

`outputs_to_string` supports two formats:

1. Single output format - use "source", "handler", and/or "raw_result" at the root level:

   ```python
   {
       "source": "docs", "handler": format_documents, "raw_result": False
   }
   ```

   - `source`: If provided, only the specified output key is sent to the handler.
   - `handler`: A function that takes the tool output (or the extracted source value) and returns the
     final result.
   - `raw_result`: If `True`, the result is returned raw without string conversion, but applying the
     `handler` if provided. This is intended for tools that return images. In this mode, the Tool
     function or the `handler` function must return a list of `TextContent`/`ImageContent` objects to
     ensure compatibility with Chat Generators.

1. Multiple output format - map keys to individual configurations:

   ```python
   {
       "formatted_docs": {"source": "docs", "handler": format_documents},
       "summary": {"source": "summary_text", "handler": str.upper}
   }
   ```

   Each key maps to a dictionary that can contain "source" and/or "handler".
   Note that `raw_result` is not supported in the multiple output format.

- **inputs_from_state** (<code>dict\[str, str\] | None</code>) – Optional dictionary mapping state keys to tool parameter names.
  Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
- **outputs_to_state** (<code>dict\[str, dict\[str, str | Callable\]\] | None</code>) – Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
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

**Raises:**

- <code>ValueError</code> – If the component is invalid or schema generation fails.

#### `tool_spec`

```python
tool_spec: dict[str, Any]
```

Return the Tool specification to be used by the Language Model.

#### `invoke`

```python
invoke(**kwargs: Any) -> Any
```

Invoke the Tool with the provided keyword arguments.

#### `warm_up`

```python
warm_up()
```

Prepare the ComponentTool for use.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the ComponentTool to a dictionary.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ComponentTool
```

Deserializes the ComponentTool from a dictionary.

## `haystack.tools.from_function`

### `haystack.tools.from_function.create_tool_from_function`

```python
create_tool_from_function(
    function: Callable,
    name: str | None = None,
    description: str | None = None,
    inputs_from_state: dict[str, str] | None = None,
    outputs_to_state: dict[str, dict[str, Any]] | None = None,
    outputs_to_string: dict[str, Any] | None = None,
) -> Tool
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

**Parameters:**

- **function** (<code>Callable</code>) – The function to be converted into a Tool.
  The function must include type hints for all parameters.
  The function is expected to have basic python input types (str, int, float, bool, list, dict, tuple).
  Other input types may work but are not guaranteed.
  If a parameter is annotated using `typing.Annotated`, its metadata will be used as parameter description.
- **name** (<code>str | None</code>) – The name of the Tool. If not provided, the name of the function will be used.
- **description** (<code>str | None</code>) – The description of the Tool. If not provided, the docstring of the function will be used.
  To intentionally leave the description empty, pass an empty string.
- **inputs_from_state** (<code>dict\[str, str\] | None</code>) – Optional dictionary mapping state keys to tool parameter names.
  Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
- **outputs_to_state** (<code>dict\[str, dict\[str, Any\]\] | None</code>) – Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
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

- **outputs_to_string** (<code>dict\[str, Any\] | None</code>) – Optional dictionary defining how tool outputs should be converted into string(s) or results.
  If not provided, the tool result is converted to a string using a default handler.

`outputs_to_string` supports two formats:

1. Single output format - use "source", "handler", and/or "raw_result" at the root level:

   ```python
   {
       "source": "docs", "handler": format_documents, "raw_result": False
   }
   ```

   - `source`: If provided, only the specified output key is sent to the handler. If not provided, the whole
     tool result is sent to the handler.
   - `handler`: A function that takes the tool output (or the extracted source value) and returns the
     final result.
   - `raw_result`: If `True`, the result is returned raw without string conversion, but applying the `handler`
     if provided. This is intended for tools that return images. In this mode, the Tool function or the
     `handler` must return a list of `TextContent`/`ImageContent` objects to ensure compatibility with Chat
     Generators.

1. Multiple output format - map keys to individual configurations:

   ```python
   {
       "formatted_docs": {"source": "docs", "handler": format_documents},
       "summary": {"source": "summary_text", "handler": str.upper}
   }
   ```

   Each key maps to a dictionary that can contain "source" and/or "handler".
   Note that `raw_result` is not supported in the multiple output format.

**Returns:**

- <code>Tool</code> – The Tool created from the function.

**Raises:**

- <code>ValueError</code> – If any parameter of the function lacks a type hint.
- <code>SchemaGenerationError</code> – If there is an error generating the JSON schema for the Tool.

### `haystack.tools.from_function.tool`

```python
tool(
    function: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    inputs_from_state: dict[str, str] | None = None,
    outputs_to_state: dict[str, dict[str, Any]] | None = None,
    outputs_to_string: dict[str, Any] | None = None
) -> Tool | Callable[[Callable], Tool]
```

Decorator to convert a function into a Tool.

Can be used with or without parameters:
@tool # without parameters
def my_function(): ...

@tool(name="custom_name") # with parameters
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

**Parameters:**

- **function** (<code>Callable | None</code>) – The function to decorate (when used without parameters)
- **name** (<code>str | None</code>) – Optional custom name for the tool
- **description** (<code>str | None</code>) – Optional custom description
- **inputs_from_state** (<code>dict\[str, str\] | None</code>) – Optional dictionary mapping state keys to tool parameter names.
  Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
- **outputs_to_state** (<code>dict\[str, dict\[str, Any\]\] | None</code>) – Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
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

- **outputs_to_string** (<code>dict\[str, Any\] | None</code>) – Optional dictionary defining how tool outputs should be converted into string(s) or results.
  If not provided, the tool result is converted to a string using a default handler.

`outputs_to_string` supports two formats:

1. Single output format - use "source", "handler", and/or "raw_result" at the root level:

   ```python
   {
       "source": "docs", "handler": format_documents, "raw_result": False
   }
   ```

   - `source`: If provided, only the specified output key is sent to the handler. If not provided, the whole
     tool result is sent to the handler.
   - `handler`: A function that takes the tool output (or the extracted source value) and returns the
     final result.
   - `raw_result`: If `True`, the result is returned raw without string conversion, but applying the `handler`
     if provided. This is intended for tools that return images. In this mode, the Tool function or the
     `handler` must return a list of `TextContent`/`ImageContent` objects to ensure compatibility with Chat
     Generators.

1. Multiple output format - map keys to individual configurations:

   ```python
   {
       "formatted_docs": {"source": "docs", "handler": format_documents},
       "summary": {"source": "summary_text", "handler": str.upper}
   }
   ```

   Each key maps to a dictionary that can contain "source" and/or "handler".
   Note that `raw_result` is not supported in the multiple output format.

**Returns:**

- <code>Tool | Callable\\[[Callable\], Tool\]</code> – Either a Tool instance or a decorator function that will create one

## `haystack.tools.pipeline_tool`

### `haystack.tools.pipeline_tool.PipelineTool`

Bases: <code>ComponentTool</code>

A Tool that wraps Haystack Pipelines, allowing them to be used as tools by LLMs.

PipelineTool automatically generates LLM-compatible tool schemas from pipeline input sockets,
which are derived from the underlying components in the pipeline.

Key features:

- Automatic LLM tool calling schema generation from pipeline inputs
- Description extraction of pipeline inputs based on the underlying component docstrings

To use PipelineTool, you first need a Haystack pipeline.
Below is an example of creating a PipelineTool

## Usage Example:

```python
from haystack import Document, Pipeline
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder
from haystack.components.embedders.sentence_transformers_document_embedder import (
    SentenceTransformersDocumentEmbedder
)
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.agents import Agent
from haystack.tools import PipelineTool

# Initialize a document store and add some documents
document_store = InMemoryDocumentStore()
document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
documents = [
    Document(content="Nikola Tesla was a Serbian-American inventor and electrical engineer."),
    Document(
        content="He is best known for his contributions to the design of the modern alternating current (AC) "
                "electricity supply system."
    ),
]
document_embedder.warm_up()
docs_with_embeddings = document_embedder.run(documents=documents)["documents"]
document_store.write_documents(docs_with_embeddings)

# Build a simple retrieval pipeline
retrieval_pipeline = Pipeline()
retrieval_pipeline.add_component(
    "embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)
retrieval_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))

retrieval_pipeline.connect("embedder.embedding", "retriever.query_embedding")

# Wrap the pipeline as a tool
retriever_tool = PipelineTool(
    pipeline=retrieval_pipeline,
    input_mapping={"query": ["embedder.text"]},
    output_mapping={"retriever.documents": "documents"},
    name="document_retriever",
    description="For any questions about Nikola Tesla, always use this tool",
)

# Create an Agent with the tool
agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"),
    tools=[retriever_tool]
)

# Let the Agent handle a query
result = agent.run([ChatMessage.from_user("Who was Nikola Tesla?")])

# Print result of the tool call
print("Tool Call Result:")
print(result["messages"][2].tool_call_result.result)
print("")

# Print answer
print("Answer:")
print(result["messages"][-1].text)
```

#### `__init__`

```python
__init__(
    pipeline: Pipeline | AsyncPipeline,
    *,
    name: str,
    description: str,
    input_mapping: dict[str, list[str]] | None = None,
    output_mapping: dict[str, str] | None = None,
    parameters: dict[str, Any] | None = None,
    outputs_to_string: dict[str, str | Callable[[Any], str]] | None = None,
    inputs_from_state: dict[str, str] | None = None,
    outputs_to_state: dict[str, dict[str, str | Callable]] | None = None
) -> None
```

Create a Tool instance from a Haystack pipeline.

**Parameters:**

- **pipeline** (<code>Pipeline | AsyncPipeline</code>) – The Haystack pipeline to wrap as a tool.
- **name** (<code>str</code>) – Name of the tool.
- **description** (<code>str</code>) – Description of the tool.
- **input_mapping** (<code>dict\[str, list\[str\]\] | None</code>) – A dictionary mapping component input names to pipeline input socket paths.
  If not provided, a default input mapping will be created based on all pipeline inputs.
  Example:

```python
input_mapping={
    "query": ["retriever.query", "prompt_builder.query"],
}
```

- **output_mapping** (<code>dict\[str, str\] | None</code>) – A dictionary mapping pipeline output socket paths to component output names.
  If not provided, a default output mapping will be created based on all pipeline outputs.
  Example:

```python
output_mapping={
    "retriever.documents": "documents",
    "generator.replies": "replies",
}
```

- **parameters** (<code>dict\[str, Any\] | None</code>) – A JSON schema defining the parameters expected by the Tool.
  Will fall back to the parameters defined in the component's run method signature if not provided.
- **outputs_to_string** (<code>dict\[str, str | Callable\\[[Any\], str\]\] | None</code>) – Optional dictionary defining how tool outputs should be converted into string(s) or results.
  If not provided, the tool result is converted to a string using a default handler.

`outputs_to_string` supports two formats:

1. Single output format - use "source", "handler", and/or "raw_result" at the root level:

   ```python
   {
       "source": "docs", "handler": format_documents, "raw_result": False
   }
   ```

   - `source`: If provided, only the specified output key is sent to the handler.
   - `handler`: A function that takes the tool output (or the extracted source value) and returns the
     final result.
   - `raw_result`: If `True`, the result is returned raw without string conversion, but applying the
     `handler` if provided. This is intended for tools that return images. In this mode, the Tool
     function or the `handler` function must return a list of `TextContent`/`ImageContent` objects to
     ensure compatibility with Chat Generators.

1. Multiple output format - map keys to individual configurations:

   ```python
   {
       "formatted_docs": {"source": "docs", "handler": format_documents},
       "summary": {"source": "summary_text", "handler": str.upper}
   }
   ```

   Each key maps to a dictionary that can contain "source" and/or "handler".
   Note that `raw_result` is not supported in the multiple output format.

- **inputs_from_state** (<code>dict\[str, str\] | None</code>) – Optional dictionary mapping state keys to tool parameter names.
  Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
- **outputs_to_state** (<code>dict\[str, dict\[str, str | Callable\]\] | None</code>) – Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
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

**Raises:**

- <code>ValueError</code> – If the provided pipeline is not a valid Haystack Pipeline instance.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the PipelineTool to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized dictionary representation of PipelineTool.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> PipelineTool
```

Deserializes the PipelineTool from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary representation of PipelineTool.

**Returns:**

- <code>PipelineTool</code> – The deserialized PipelineTool instance.

#### `tool_spec`

```python
tool_spec: dict[str, Any]
```

Return the Tool specification to be used by the Language Model.

#### `invoke`

```python
invoke(**kwargs: Any) -> Any
```

Invoke the Tool with the provided keyword arguments.

#### `warm_up`

```python
warm_up()
```

Prepare the ComponentTool for use.

## `haystack.tools.tool`

### `haystack.tools.tool.Tool`

Data class representing a Tool that Language Models can prepare a call for.

Accurate definitions of the textual attributes such as `name` and `description`
are important for the Language Model to correctly prepare the call.

For resource-intensive operations like establishing connections to remote services or
loading models, override the `warm_up()` method. This method is called before the Tool
is used and should be idempotent, as it may be called multiple times during
pipeline/agent setup.

**Parameters:**

- **name** (<code>str</code>) – Name of the Tool.
- **description** (<code>str</code>) – Description of the Tool.
- **parameters** (<code>dict\[str, Any\]</code>) – A JSON schema defining the parameters expected by the Tool.
- **function** (<code>Callable</code>) – The function that will be invoked when the Tool is called.
  Must be a synchronous function; async functions are not supported.
- **outputs_to_string** (<code>dict\[str, Any\] | None</code>) – Optional dictionary defining how tool outputs should be converted into string(s) or results.
  If not provided, the tool result is converted to a string using a default handler.

`outputs_to_string` supports two formats:

1. Single output format - use "source", "handler", and/or "raw_result" at the root level:

   ```python
   {
       "source": "docs", "handler": format_documents, "raw_result": False
   }
   ```

   - `source`: If provided, only the specified output key is sent to the handler. If not provided, the whole
     tool result is sent to the handler.
   - `handler`: A function that takes the tool output (or the extracted source value) and returns the
     final result.
   - `raw_result`: If `True`, the result is returned raw without string conversion, but applying the `handler`
     if provided. This is intended for tools that return images. In this mode, the Tool function or the
     `handler` must return a list of `TextContent`/`ImageContent` objects to ensure compatibility with Chat
     Generators.

1. Multiple output format - map keys to individual configurations:

   ```python
   {
       "formatted_docs": {"source": "docs", "handler": format_documents},
       "summary": {"source": "summary_text", "handler": str.upper}
   }
   ```

   Each key maps to a dictionary that can contain "source" and/or "handler".
   Note that `raw_result` is not supported in the multiple output format.

- **inputs_from_state** (<code>dict\[str, str\] | None</code>) – Optional dictionary mapping state keys to tool parameter names.
  Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
- **outputs_to_state** (<code>dict\[str, dict\[str, Any\]\] | None</code>) – Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
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

#### `tool_spec`

```python
tool_spec: dict[str, Any]
```

Return the Tool specification to be used by the Language Model.

#### `warm_up`

```python
warm_up() -> None
```

Prepare the Tool for use.

Override this method to establish connections to remote services, load models,
or perform other resource-intensive initialization. This method should be idempotent,
as it may be called multiple times.

#### `invoke`

```python
invoke(**kwargs: Any) -> Any
```

Invoke the Tool with the provided keyword arguments.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the Tool to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> Tool
```

Deserializes the Tool from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>Tool</code> – Deserialized Tool.

## `haystack.tools.toolset`

### `haystack.tools.toolset.Toolset`

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

1. Base class for dynamic tool loading:
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

#### `warm_up`

```python
warm_up() -> None
```

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

This method should be idempotent, as it may be called multiple times.

#### `add`

```python
add(tool: Union[Tool, Toolset]) -> None
```

Add a new Tool or merge another Toolset.

**Parameters:**

- **tool** (<code>Union\[Tool, Toolset\]</code>) – A Tool instance or another Toolset to add

**Raises:**

- <code>ValueError</code> – If adding the tool would result in duplicate tool names
- <code>TypeError</code> – If the provided object is not a Tool or Toolset

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the Toolset to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary representation of the Toolset

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

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> Toolset
```

Deserialize a Toolset from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary representation of the Toolset

**Returns:**

- <code>Toolset</code> – A new Toolset instance
