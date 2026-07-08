---
title: "Tools"
id: tools-api
description: "Unified abstractions to represent tools across the framework."
slug: "/tools-api"
---


## component_tool

### ComponentTool

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
Below is an example of creating a ComponentTool from an existing SerperDevWebSearch component
from the `serperdev-haystack` integration package (`pip install serperdev-haystack`).

## Usage Example:

<!-- test-ignore -->

```python
from haystack import component
from haystack.tools import ComponentTool
from haystack.utils import Secret
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.websearch.serperdev import SerperDevWebSearch

# Create a SerperDev search component
search = SerperDevWebSearch(api_key=Secret.from_env_var("SERPERDEV_API_KEY"), top_k=3)

# Create a tool from the component
tool = ComponentTool(
    component=search,
    name="web_search",  # Optional: defaults to "serper_dev_web_search"
    description="Search the web for current information on any topic"  # Optional: defaults to component docstring
)

# Create an Agent with an OpenAIChatGenerator and the tool
agent = Agent(chat_generator=OpenAIChatGenerator(), tools=[tool])

message = ChatMessage.from_user("Use the web search tool to find information about Nikola Tesla")

# Run the Agent
result = agent.run(messages=[message])

print(result)
```

#### __init__

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

- <code>TypeError</code> – If the object passed is not a Haystack Component instance.
- <code>ValueError</code> – If the component has already been added to a pipeline, or if schema generation fails.

#### warm_up

```python
warm_up() -> None
```

Prepare the ComponentTool for use.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the ComponentTool to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ComponentTool
```

Deserializes the ComponentTool from a dictionary.

## from_function

### create_tool_from_function

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
# >> Tool(name='get_weather', description='A simple function to get the current weather for a location.',
# >> parameters={
# >> 'type': 'object',
# >> 'properties': {
# >>     'city': {'type': 'string', 'description': 'the city for which to get the weather', 'default': 'Munich'},
# >>     'unit': {
# >>         'type': 'string',
# >>         'enum': ['Celsius', 'Fahrenheit'],
# >>         'description': 'the unit for the temperature',
# >>         'default': 'Celsius',
# >>     },
# >>     }
# >> },
# >> function=<function get_weather at 0x7f7b3a8a9b80>)
```

**Parameters:**

- **function** (<code>Callable</code>) – The function to be converted into a Tool. May be either a regular function (assigned to the
  resulting Tool's `function` field) or a coroutine function defined with `async def` (assigned
  to `async_function`).
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

### tool

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
# >> Tool(name='get_weather', description='A simple function to get the current weather for a location.',
# >> parameters={
# >> 'type': 'object',
# >> 'properties': {
# >>     'city': {'type': 'string', 'description': 'the city for which to get the weather', 'default': 'Munich'},
# >>     'unit': {
# >>         'type': 'string',
# >>         'enum': ['Celsius', 'Fahrenheit'],
# >>         'description': 'the unit for the temperature',
# >>         'default': 'Celsius',
# >>     },
# >>     }
# >> },
# >> function=<function get_weather at 0x7f7b3a8a9b80>)
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

## pipeline_tool

### PipelineTool

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
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.agents import Agent
from haystack.tools import PipelineTool

# Initialize a document store and add some documents
document_store = InMemoryDocumentStore()
document_embedder = OpenAIDocumentEmbedder()
documents = [
    Document(content="Nikola Tesla was a Serbian-American inventor and electrical engineer."),
    Document(
        content="He is best known for his contributions to the design of the modern alternating current (AC) "
                "electricity supply system."
    ),
]
docs_with_embeddings = document_embedder.run(documents=documents)["documents"]
document_store.write_documents(docs_with_embeddings)

# Build a simple retrieval pipeline
retrieval_pipeline = Pipeline()
retrieval_pipeline.add_component("embedder", OpenAITextEmbedder())
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

#### __init__

```python
__init__(
    pipeline: Pipeline,
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

- **pipeline** (<code>Pipeline</code>) – The Haystack pipeline to wrap as a tool.
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

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the PipelineTool to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized dictionary representation of PipelineTool.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> PipelineTool
```

Deserializes the PipelineTool from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary representation of PipelineTool.

**Returns:**

- <code>PipelineTool</code> – The deserialized PipelineTool instance.

## searchable_toolset

### SearchableToolset

Bases: <code>Toolset</code>

Dynamic tool discovery from large catalogs using BM25 search.

This Toolset enables LLMs to discover and use tools from large catalogs through BM25-based search.
Instead of exposing all tools at once (which can overwhelm the LLM context), it provides a `search_tools` bootstrap
tool that allows the LLM to find and load specific tools as needed.

For very small catalogs (below `search_threshold`), acts as a simple passthrough exposing all tools directly
without any discovery mechanism.

### Usage Example

```python
from typing import Annotated

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import SearchableToolset, tool

@tool
def get_weather(city: Annotated[str, "The city to get the weather for"]) -> str:
    '''Get the current weather for a city.'''
    return f"The weather in {city} is 22°C and sunny."

@tool
def search_web(query: Annotated[str, "The query to search the web for"]) -> str:
    '''Search the web for a query.'''
    return f"Top result for '{query}': ..."

@tool
def convert_currency(
    amount: Annotated[float, "The amount to convert"],
    to_currency: Annotated[str, "The currency to convert to, e.g. 'EUR'"],
) -> str:
    '''Convert an amount in USD to another currency.'''
    return f"{amount} USD is {amount * 0.9} {to_currency}"

# search_threshold=2 means a catalog of 2+ tools activates discovery: the agent only sees the
# `search_tools` tool and must search to load the others (set it higher for larger catalogs).
toolset = SearchableToolset(catalog=[get_weather, search_web, convert_currency], search_threshold=2)

agent = Agent(chat_generator=OpenAIChatGenerator(), tools=toolset)

# The agent is initially provided only with the search_tools tool and will use it to find relevant tools.
result = agent.run(messages=[ChatMessage.from_user("What's the weather in Milan?")])
print(result["last_message"].text)
```

#### __init__

```python
__init__(
    catalog: ToolsType,
    *,
    top_k: int = 3,
    search_threshold: int = 8,
    search_tool_name: str = "search_tools",
    search_tool_description: str | None = None,
    search_tool_parameters_description: dict[str, str] | None = None
) -> None
```

Initialize the SearchableToolset.

**Parameters:**

- **catalog** (<code>ToolsType</code>) – Source of tools - a list of Tools, list of Toolsets, or a single Toolset.
- **top_k** (<code>int</code>) – Default number of results for search_tools.
- **search_threshold** (<code>int</code>) – Minimum catalog size to activate search. If catalog has fewer tools, acts as
  passthrough (all tools visible). Default is 8.
- **search_tool_name** (<code>str</code>) – Custom name for the bootstrap search tool. Default is "search_tools".
- **search_tool_description** (<code>str | None</code>) – Custom description for the bootstrap search tool. If not provided, uses a
  default description.
- **search_tool_parameters_description** (<code>dict\[str, str\] | None</code>) – Custom descriptions for the bootstrap search tool's parameters.
  Keys must be a subset of `{"tool_keywords", "k"}`.
  Example: `{"tool_keywords": "Keywords to find tools, e.g. 'email send'"}`

#### add

```python
add(tool: Tool | Toolset) -> None
```

Adding new tools after initialization is not supported for SearchableToolset.

#### warm_up

```python
warm_up() -> None
```

Prepare the toolset for use.

Warms up the catalog (so lazy toolsets like MCPToolset can connect) and flattens it. Above the passthrough
threshold, it also indexes the catalog and creates the search_tools bootstrap tool.

This method is idempotent: it only warms up the toolset the first time it is called.

**Raises:**

- <code>ValueError</code> – If the flattened catalog contains tools with duplicate names.

#### get_selectable_tools

```python
get_selectable_tools() -> list[Tool]
```

Return the full catalog of tools that can be selected by name.

Iteration only exposes the search tool plus already-discovered tools, but name-based selection can target
any tool in the catalog, so this returns the entire flattened catalog (warming up first if needed).

**Returns:**

- <code>list\[Tool\]</code> – The flattened catalog of tools.

#### clear

```python
clear() -> None
```

Clear all discovered tools.

This method allows resetting the toolset's discovered tools between agent runs when the same toolset instance
is reused. This can be useful for long-running applications to control memory usage or to start fresh searches.

#### spawn

```python
spawn() -> SearchableToolset
```

Return an isolated copy for a single run.

The copy shares the read-only catalog and BM25 index but gets fresh discovered tools and name selection,
plus a bootstrap search tool bound to the copy. This way concurrent runs sharing the same configured
SearchableToolset don't share discovered tools or collide on the active selection.

**Returns:**

- <code>SearchableToolset</code> – A run-scoped copy of this SearchableToolset.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the toolset to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary representation of the toolset.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SearchableToolset
```

Deserialize a toolset from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary representation of the toolset.

**Returns:**

- <code>SearchableToolset</code> – New SearchableToolset instance.

**Raises:**

- <code>TypeError</code> – If a serialized catalog entry is not a subclass of Tool or Toolset.

## skills/skill_toolset

### SkillToolset

Bases: <code>Toolset</code>

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

#### __init__

```python
__init__(store: SkillStore) -> None
```

Initialize the SkillToolset.

Constructing the toolset does not read any skills. The store is queried for the available skills on
`warm_up()`, so stores that do I/O (reading a directory, connecting to a database) stay cheap to
construct.

The `load_skill` and `read_skill_file` tools are created right away, so the toolset can be used as a
collection (length, membership checks, iteration) immediately.

**Parameters:**

- **store** (<code>SkillStore</code>) – A `haystack.skill_stores.types.SkillStore` instance to back this toolset.

#### skills

```python
skills: dict[str, SkillInfo]
```

Mapping of skill name to its metadata. Triggers `warm_up()` on first access if not already warmed up.

#### warm_up

```python
warm_up() -> None
```

Discover the available skills from the store and bake the catalog into the `load_skill` description.

Only the description content is dynamic, so the (static) tools created in `__init__` are reused; this
refreshes `load_skill`'s description once the catalog is known. Idempotent: repeated calls after the
first are no-ops.

#### add

```python
add(tool: Tool | Toolset) -> None
```

Adding tools is not supported: a SkillToolset's tools are fixed and defined by its store.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the toolset to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary representation of the toolset.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SkillToolset
```

Deserialize a toolset from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary representation of the toolset, as produced by `to_dict`.

**Returns:**

- <code>SkillToolset</code> – A new SkillToolset instance.

## tool

### Tool

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
- **function** (<code>Callable | None</code>) – The synchronous function invoked by `Tool.invoke`. Must be a regular function — coroutine functions should
  be passed to `async_function` instead. Either `function` or `async_function` (or both) must be set.
- **async_function** (<code>Callable | None</code>) – Optional coroutine function awaited by `Tool.invoke_async`. When only `async_function` is set, `invoke` raises
  a `ToolInvocationError`. When only `function` is set, `invoke_async` falls back to running `function` in a
  worker thread via `asyncio.to_thread`.
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

**Raises:**

- <code>ValueError</code> – If neither `function` nor `async_function` is provided, if `function` is a
  coroutine function, if `async_function` is not a coroutine function, if `parameters` is not a
  valid JSON schema, or if the `outputs_to_state`, `outputs_to_string`, or `inputs_from_state`
  configurations are invalid.
- <code>TypeError</code> – If any configuration value in `outputs_to_state`, `outputs_to_string`, or
  `inputs_from_state` has the wrong type.

#### tool_spec

```python
tool_spec: dict[str, Any]
```

Return the Tool specification to be used by the Language Model.

#### warm_up

```python
warm_up() -> None
```

Prepare the Tool for use.

Override this method to establish connections to remote services, load models,
or perform other resource-intensive initialization. This method should be idempotent,
as it may be called multiple times.

#### invoke

```python
invoke(**kwargs: Any) -> Any
```

Invoke the Tool synchronously with the provided keyword arguments.

**Raises:**

- <code>ToolInvocationError</code> – If the Tool has no sync `function`, or if the underlying call
  raises an exception.

#### invoke_async

```python
invoke_async(**kwargs: Any) -> Any
```

Invoke the Tool asynchronously with the provided keyword arguments.

If `async_function` is set, it is awaited directly. Otherwise the sync `function` is dispatched to a worker
thread via `asyncio.to_thread`, which propagates the current context to the worker.

**Raises:**

- <code>ToolInvocationError</code> – If the underlying call raises an exception.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the Tool to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> Tool
```

Deserializes the Tool from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>Tool</code> – Deserialized Tool.

## toolset

### Toolset

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

#### get_selectable_tools

```python
get_selectable_tools() -> list[Tool]
```

Return the full set of tools that can be selected by name, ignoring any active name filter.

This differs from iteration, which yields only the tools currently exposed (and respects the name filter).
Override this when a Toolset's iteration does not surface every selectable tool, so name-based selection
can still target the full set.

Warms up the Toolset first if needed, so lazily loaded tools (those a Toolset fetches in `warm_up()`)
are available for selection.

**Returns:**

- <code>list\[Tool\]</code> – The list of tools available for name-based selection.

#### spawn

```python
spawn() -> Toolset
```

Return an isolated copy of this Toolset for a single run.

The copy shares this Toolset's read-only state (its tools and any warmed-up resources) but gets fresh
run-scoped state, so concurrent runs that share the same configured Toolset don't corrupt each other (for
example, one run's name selection leaking into another). Warms up first if needed so the copy shares the
warmed state. Subclasses with additional run-scoped state should override this.

**Returns:**

- <code>Toolset</code> – A run-scoped copy of this Toolset.

#### warm_up

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

This method is idempotent: it only warms up the tools the first time it is called.
Subclasses overriding it should preserve this contract (for example by guarding on
`self._is_warmed_up`).

#### add

```python
add(tool: Tool | Toolset) -> None
```

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

**Parameters:**

- **tool** (<code>Tool | Toolset</code>) – A Tool instance or another Toolset to add

**Raises:**

- <code>ValueError</code> – If adding the tool would result in duplicate tool names
- <code>TypeError</code> – If the provided object is not a Tool or Toolset

#### to_dict

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

#### from_dict

```python
from_dict(data: dict[str, Any]) -> Toolset
```

Deserialize a Toolset from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary representation of the Toolset

**Returns:**

- <code>Toolset</code> – A new Toolset instance
