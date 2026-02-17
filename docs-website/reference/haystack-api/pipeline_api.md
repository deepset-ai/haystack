---
title: "Pipeline"
id: pipeline-api
description: "Arranges components and integrations in flow."
slug: "/pipeline-api"
---


## `haystack.core.pipeline.async_pipeline`

### `AsyncPipeline`

Bases: <code>PipelineBase</code>

Asynchronous version of the Pipeline orchestration engine.

Manages components in a pipeline allowing for concurrent processing when the pipeline's execution graph permits.
This enables efficient processing of components by minimizing idle time and maximizing resource utilization.

#### `__init__`

```python
__init__(
    metadata: dict[str, Any] | None = None,
    max_runs_per_component: int = 100,
    connection_type_validation: bool = True,
)
```

Creates the Pipeline.

**Parameters:**

- **metadata** (<code>dict\[str, Any\] | None</code>) – Arbitrary dictionary to store metadata about this `Pipeline`. Make sure all the values contained in
  this dictionary can be serialized and deserialized if you wish to save this `Pipeline` to file.
- **max_runs_per_component** (<code>int</code>) – How many times the `Pipeline` can run the same Component.
  If this limit is reached a `PipelineMaxComponentRuns` exception is raised.
  If not set defaults to 100 runs per Component.
- **connection_type_validation** (<code>bool</code>) – Whether the pipeline will validate the types of the connections.
  Defaults to True.

#### `run_async_generator`

```python
run_async_generator(
    data: dict[str, Any],
    include_outputs_from: set[str] | None = None,
    concurrency_limit: int = 4,
) -> AsyncIterator[dict[str, Any]]
```

Executes the pipeline step by step asynchronously, yielding partial outputs when any component finishes.

Usage:

```python
from haystack import Document
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack import AsyncPipeline
import asyncio

# Write documents to InMemoryDocumentStore
document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome.")
])

prompt_template = [
    ChatMessage.from_user(
        '''
        Given these documents, answer the question.
        Documents:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}
        Question: {{question}}
        Answer:
        ''')
]

# Create and connect pipeline components
retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = ChatPromptBuilder(template=prompt_template)
llm = OpenAIChatGenerator()

rag_pipeline = AsyncPipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Prepare input data
question = "Who lives in Paris?"
data = {
    "retriever": {"query": question},
    "prompt_builder": {"question": question},
}


# Process results as they become available
async def process_results():
    async for partial_output in rag_pipeline.run_async_generator(
            data=data,
            include_outputs_from={"retriever", "llm"}
    ):
        # Each partial_output contains the results from a completed component
        if "retriever" in partial_output:
            print("Retrieved documents:", len(partial_output["retriever"]["documents"]))
        if "llm" in partial_output:
            print("Generated answer:", partial_output["llm"]["replies"][0])


asyncio.run(process_results())
```

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Initial input data to the pipeline.
- **concurrency_limit** (<code>int</code>) – The maximum number of components that are allowed to run concurrently.
- **include_outputs_from** (<code>set\[str\] | None</code>) – Set of component names whose individual outputs are to be
  included in the pipeline's output. For components that are
  invoked multiple times (in a loop), only the last-produced
  output is included.

**Returns:**

- <code>AsyncIterator\[dict\[str, Any\]\]</code> – An async iterator containing partial (and final) outputs.

**Raises:**

- <code>ValueError</code> – If invalid inputs are provided to the pipeline.
- <code>PipelineMaxComponentRuns</code> – If a component exceeds the maximum number of allowed executions within the pipeline.
- <code>PipelineRuntimeError</code> – If the Pipeline contains cycles with unsupported connections that would cause
  it to get stuck and fail running.
  Or if a Component fails or returns output in an unsupported type.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the pipeline to a dictionary.

This is meant to be an intermediate representation but it can be also used to save a pipeline to file.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(
    data: dict[str, Any],
    callbacks: DeserializationCallbacks | None = None,
    **kwargs: Any
) -> T
```

Deserializes the pipeline from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.
- **callbacks** (<code>DeserializationCallbacks | None</code>) – Callbacks to invoke during deserialization.
- **kwargs** (<code>Any</code>) – `components`: a dictionary of `{name: instance}` to reuse instances of components instead of creating new
  ones.

**Returns:**

- <code>T</code> – Deserialized component.

#### `dumps`

```python
dumps(marshaller: Marshaller = DEFAULT_MARSHALLER) -> str
```

Returns the string representation of this pipeline according to the format dictated by the `Marshaller` in use.

**Parameters:**

- **marshaller** (<code>Marshaller</code>) – The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.

**Returns:**

- <code>str</code> – A string representing the pipeline.

#### `dump`

```python
dump(fp: TextIO, marshaller: Marshaller = DEFAULT_MARSHALLER) -> None
```

Writes the string representation of this pipeline to the file-like object passed in the `fp` argument.

**Parameters:**

- **fp** (<code>TextIO</code>) – A file-like object ready to be written to.
- **marshaller** (<code>Marshaller</code>) – The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.

#### `loads`

```python
loads(
    data: str | bytes | bytearray,
    marshaller: Marshaller = DEFAULT_MARSHALLER,
    callbacks: DeserializationCallbacks | None = None,
) -> T
```

Creates a `Pipeline` object from the string representation passed in the `data` argument.

**Parameters:**

- **data** (<code>str | bytes | bytearray</code>) – The string representation of the pipeline, can be `str`, `bytes` or `bytearray`.
- **marshaller** (<code>Marshaller</code>) – The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
- **callbacks** (<code>DeserializationCallbacks | None</code>) – Callbacks to invoke during deserialization.

**Returns:**

- <code>T</code> – A `Pipeline` object.

**Raises:**

- <code>DeserializationError</code> – If an error occurs during deserialization.

#### `load`

```python
load(
    fp: TextIO,
    marshaller: Marshaller = DEFAULT_MARSHALLER,
    callbacks: DeserializationCallbacks | None = None,
) -> T
```

Creates a `Pipeline` object a string representation.

The string representation is read from the file-like object passed in the `fp` argument.

**Parameters:**

- **fp** (<code>TextIO</code>) – A file-like object ready to be read from.
- **marshaller** (<code>Marshaller</code>) – The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
- **callbacks** (<code>DeserializationCallbacks | None</code>) – Callbacks to invoke during deserialization.

**Returns:**

- <code>T</code> – A `Pipeline` object.

**Raises:**

- <code>DeserializationError</code> – If an error occurs during deserialization.

#### `add_component`

```python
add_component(name: str, instance: Component) -> None
```

Add the given component to the pipeline.

Components are not connected to anything by default: use `Pipeline.connect()` to connect components together.
Component names must be unique, but component instances can be reused if needed.

**Parameters:**

- **name** (<code>str</code>) – The name of the component to add.
- **instance** (<code>Component</code>) – The component instance to add.

**Raises:**

- <code>ValueError</code> – If a component with the same name already exists.
- <code>PipelineValidationError</code> – If the given instance is not a component.

#### `remove_component`

```python
remove_component(name: str) -> Component
```

Remove and returns component from the pipeline.

Remove an existing component from the pipeline by providing its name.
All edges that connect to the component will also be deleted.

**Parameters:**

- **name** (<code>str</code>) – The name of the component to remove.

**Returns:**

- <code>Component</code> – The removed Component instance.

**Raises:**

- <code>ValueError</code> – If there is no component with that name already in the Pipeline.

#### `connect`

```python
connect(sender: str, receiver: str) -> PipelineBase
```

Connects two components together.

All components to connect must exist in the pipeline.
If connecting to a component that has several output connections, specify the inputs and output names as
'component_name.connections_name'.

**Parameters:**

- **sender** (<code>str</code>) – The component that delivers the value. This can be either just a component name or can be
  in the format `component_name.connection_name` if the component has multiple outputs.
- **receiver** (<code>str</code>) – The component that receives the value. This can be either just a component name or can be
  in the format `component_name.connection_name` if the component has multiple inputs.

**Returns:**

- <code>PipelineBase</code> – The Pipeline instance.

**Raises:**

- <code>PipelineConnectError</code> – If the two components cannot be connected (for example if one of the components is
  not present in the pipeline, or the connections don't match by type, and so on).

#### `run_async`

```python
run_async(
    data: dict[str, Any],
    include_outputs_from: set[str] | None = None,
    concurrency_limit: int = 4,
) -> dict[str, Any]
```

Provides an asynchronous interface to run the pipeline with provided input data.

This method allows the pipeline to be integrated into an asynchronous workflow, enabling non-blocking
execution of pipeline components.

Usage:

```python
import asyncio

from haystack import Document
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.pipeline import AsyncPipeline
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Write documents to InMemoryDocumentStore
document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome.")
])

prompt_template = [
    ChatMessage.from_user(
        '''
        Given these documents, answer the question.
        Documents:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}
        Question: {{question}}
        Answer:
        ''')
]

retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = ChatPromptBuilder(template=prompt_template)
llm = OpenAIChatGenerator()

rag_pipeline = AsyncPipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Ask a question
question = "Who lives in Paris?"

async def run_inner(data, include_outputs_from):
    return await rag_pipeline.run_async(data=data, include_outputs_from=include_outputs_from)

data = {
    "retriever": {"query": question},
    "prompt_builder": {"question": question},
}

results = asyncio.run(run_inner(data, include_outputs_from={"retriever", "llm"}))

print(results["llm"]["replies"])
# [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text='Jean lives in Paris.')],
# _name=None, _meta={'model': 'gpt-5-mini', 'index': 0, 'finish_reason': 'stop', 'usage':
# {'completion_tokens': 6, 'prompt_tokens': 69, 'total_tokens': 75,
# 'completion_tokens_details': CompletionTokensDetails(accepted_prediction_tokens=0,
# audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), 'prompt_tokens_details':
# PromptTokensDetails(audio_tokens=0, cached_tokens=0)}})]
```

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – A dictionary of inputs for the pipeline's components. Each key is a component name
  and its value is a dictionary of that component's input parameters:

```
data = {
    "comp1": {"input1": 1, "input2": 2},
}
```

For convenience, this format is also supported when input names are unique:

```
data = {
    "input1": 1, "input2": 2,
}
```

- **include_outputs_from** (<code>set\[str\] | None</code>) – Set of component names whose individual outputs are to be
  included in the pipeline's output. For components that are
  invoked multiple times (in a loop), only the last-produced
  output is included.
- **concurrency_limit** (<code>int</code>) – The maximum number of components that should be allowed to run concurrently.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary where each entry corresponds to a component name
  and its output. If `include_outputs_from` is `None`, this dictionary
  will only contain the outputs of leaf components, i.e., components
  without outgoing connections.

**Raises:**

- <code>ValueError</code> – If invalid inputs are provided to the pipeline.
- <code>PipelineRuntimeError</code> – If the Pipeline contains cycles with unsupported connections that would cause
  it to get stuck and fail running.
  Or if a Component fails or returns output in an unsupported type.
- <code>PipelineMaxComponentRuns</code> – If a Component reaches the maximum number of times it can be run in this Pipeline.

#### `run`

```python
run(
    data: dict[str, Any],
    include_outputs_from: set[str] | None = None,
    concurrency_limit: int = 4,
) -> dict[str, Any]
```

Provides a synchronous interface to run the pipeline with given input data.

Internally, the pipeline components are executed asynchronously, but the method itself
will block until the entire pipeline execution is complete.

In case you need asynchronous methods, consider using `run_async` or `run_async_generator`.

Usage:

```python
from haystack import Document
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.pipeline import AsyncPipeline
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Write documents to InMemoryDocumentStore
document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome.")
])

prompt_template = [
    ChatMessage.from_user(
        '''
        Given these documents, answer the question.
        Documents:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}
        Question: {{question}}
        Answer:
        ''')
]


retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = ChatPromptBuilder(template=prompt_template)
llm = OpenAIChatGenerator()

rag_pipeline = AsyncPipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Ask a question
question = "Who lives in Paris?"

data = {
    "retriever": {"query": question},
    "prompt_builder": {"question": question},
}

results = rag_pipeline.run(data)

print(results["llm"]["replies"])
# [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text='Jean lives in Paris.')],
# _name=None, _meta={'model': 'gpt-5-mini', 'index': 0, 'finish_reason': 'stop', 'usage':
# {'completion_tokens': 6, 'prompt_tokens': 69, 'total_tokens': 75, 'completion_tokens_details':
# CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0,
# rejected_prediction_tokens=0), 'prompt_tokens_details': PromptTokensDetails(audio_tokens=0,
# cached_tokens=0)}})]
```

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – A dictionary of inputs for the pipeline's components. Each key is a component name
  and its value is a dictionary of that component's input parameters:

```
data = {
    "comp1": {"input1": 1, "input2": 2},
}
```

For convenience, this format is also supported when input names are unique:

```
data = {
    "input1": 1, "input2": 2,
}
```

- **include_outputs_from** (<code>set\[str\] | None</code>) – Set of component names whose individual outputs are to be
  included in the pipeline's output. For components that are
  invoked multiple times (in a loop), only the last-produced
  output is included.
- **concurrency_limit** (<code>int</code>) – The maximum number of components that should be allowed to run concurrently.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary where each entry corresponds to a component name
  and its output. If `include_outputs_from` is `None`, this dictionary
  will only contain the outputs of leaf components, i.e., components
  without outgoing connections.

**Raises:**

- <code>ValueError</code> – If invalid inputs are provided to the pipeline.
- <code>PipelineRuntimeError</code> – If the Pipeline contains cycles with unsupported connections that would cause
  it to get stuck and fail running.
  Or if a Component fails or returns output in an unsupported type.
- <code>PipelineMaxComponentRuns</code> – If a Component reaches the maximum number of times it can be run in this Pipeline.
- <code>RuntimeError</code> – If called from within an async context. Use `run_async` instead.

#### `get_component`

```python
get_component(name: str) -> Component
```

Get the component with the specified name from the pipeline.

**Parameters:**

- **name** (<code>str</code>) – The name of the component.

**Returns:**

- <code>Component</code> – The instance of that component.

**Raises:**

- <code>ValueError</code> – If a component with that name is not present in the pipeline.

#### `get_component_name`

```python
get_component_name(instance: Component) -> str
```

Returns the name of the Component instance if it has been added to this Pipeline or an empty string otherwise.

**Parameters:**

- **instance** (<code>Component</code>) – The Component instance to look for.

**Returns:**

- <code>str</code> – The name of the Component instance.

#### `inputs`

```python
inputs(
    include_components_with_connected_inputs: bool = False,
) -> dict[str, dict[str, Any]]
```

Returns a dictionary containing the inputs of a pipeline.

Each key in the dictionary corresponds to a component name, and its value is another dictionary that describes
the input sockets of that component, including their types and whether they are optional.

**Parameters:**

- **include_components_with_connected_inputs** (<code>bool</code>) – If `False`, only components that have disconnected input edges are
  included in the output.

**Returns:**

- <code>dict\[str, dict\[str, Any\]\]</code> – A dictionary where each key is a pipeline component name and each value is a dictionary of
  inputs sockets of that component.

#### `outputs`

```python
outputs(
    include_components_with_connected_outputs: bool = False,
) -> dict[str, dict[str, Any]]
```

Returns a dictionary containing the outputs of a pipeline.

Each key in the dictionary corresponds to a component name, and its value is another dictionary that describes
the output sockets of that component.

**Parameters:**

- **include_components_with_connected_outputs** (<code>bool</code>) – If `False`, only components that have disconnected output edges are
  included in the output.

**Returns:**

- <code>dict\[str, dict\[str, Any\]\]</code> – A dictionary where each key is a pipeline component name and each value is a dictionary of
  output sockets of that component.

#### `show`

```python
show(
    *,
    server_url: str = "https://mermaid.ink",
    params: dict | None = None,
    timeout: int = 30,
    super_component_expansion: bool = False
) -> None
```

Display an image representing this `Pipeline` in a Jupyter notebook.

This function generates a diagram of the `Pipeline` using a Mermaid server and displays it directly in
the notebook.

**Parameters:**

- **server_url** (<code>str</code>) – The base URL of the Mermaid server used for rendering (default: 'https://mermaid.ink').
  See https://github.com/jihchi/mermaid.ink and https://github.com/mermaid-js/mermaid-live-editor for more
  info on how to set up your own Mermaid server.
- **params** (<code>dict | None</code>) – Dictionary of customization parameters to modify the output. Refer to Mermaid documentation for more details
  Supported keys:
  - format: Output format ('img', 'svg', or 'pdf'). Default: 'img'.
  - type: Image type for /img endpoint ('jpeg', 'png', 'webp'). Default: 'png'.
  - theme: Mermaid theme ('default', 'neutral', 'dark', 'forest'). Default: 'neutral'.
  - bgColor: Background color in hexadecimal (e.g., 'FFFFFF') or named format (e.g., '!white').
  - width: Width of the output image (integer).
  - height: Height of the output image (integer).
  - scale: Scaling factor (1–3). Only applicable if 'width' or 'height' is specified.
  - fit: Whether to fit the diagram size to the page (PDF only, boolean).
  - paper: Paper size for PDFs (e.g., 'a4', 'a3'). Ignored if 'fit' is true.
  - landscape: Landscape orientation for PDFs (boolean). Ignored if 'fit' is true.
- **timeout** (<code>int</code>) – Timeout in seconds for the request to the Mermaid server.
- **super_component_expansion** (<code>bool</code>) – If set to True and the pipeline contains SuperComponents the diagram will show the internal structure of
  super-components as if they were components part of the pipeline instead of a "black-box".
  Otherwise, only the super-component itself will be displayed.

**Raises:**

- <code>PipelineDrawingError</code> – If the function is called outside of a Jupyter notebook or if there is an issue with rendering.

#### `draw`

```python
draw(
    *,
    path: Path,
    server_url: str = "https://mermaid.ink",
    params: dict | None = None,
    timeout: int = 30,
    super_component_expansion: bool = False
) -> None
```

Save an image representing this `Pipeline` to the specified file path.

This function generates a diagram of the `Pipeline` using the Mermaid server and saves it to the provided path.

**Parameters:**

- **path** (<code>Path</code>) – The file path where the generated image will be saved.
- **server_url** (<code>str</code>) – The base URL of the Mermaid server used for rendering (default: 'https://mermaid.ink').
  See https://github.com/jihchi/mermaid.ink and https://github.com/mermaid-js/mermaid-live-editor for more
  info on how to set up your own Mermaid server.
- **params** (<code>dict | None</code>) – Dictionary of customization parameters to modify the output. Refer to Mermaid documentation for more details
  Supported keys:
  - format: Output format ('img', 'svg', or 'pdf'). Default: 'img'.
  - type: Image type for /img endpoint ('jpeg', 'png', 'webp'). Default: 'png'.
  - theme: Mermaid theme ('default', 'neutral', 'dark', 'forest'). Default: 'neutral'.
  - bgColor: Background color in hexadecimal (e.g., 'FFFFFF') or named format (e.g., '!white').
  - width: Width of the output image (integer).
  - height: Height of the output image (integer).
  - scale: Scaling factor (1–3). Only applicable if 'width' or 'height' is specified.
  - fit: Whether to fit the diagram size to the page (PDF only, boolean).
  - paper: Paper size for PDFs (e.g., 'a4', 'a3'). Ignored if 'fit' is true.
  - landscape: Landscape orientation for PDFs (boolean). Ignored if 'fit' is true.
- **timeout** (<code>int</code>) – Timeout in seconds for the request to the Mermaid server.
- **super_component_expansion** (<code>bool</code>) – If set to True and the pipeline contains SuperComponents the diagram will show the internal structure of
  super-components as if they were components part of the pipeline instead of a "black-box".
  Otherwise, only the super-component itself will be displayed.

**Raises:**

- <code>PipelineDrawingError</code> – If there is an issue with rendering or saving the image.

#### `walk`

```python
walk() -> Iterator[tuple[str, Component]]
```

Visits each component in the pipeline exactly once and yields its name and instance.

No guarantees are provided on the visiting order.

**Returns:**

- <code>Iterator\[tuple\[str, Component\]\]</code> – An iterator of tuples of component name and component instance.

#### `warm_up`

```python
warm_up() -> None
```

Make sure all nodes are warm.

It's the node's responsibility to make sure this method can be called at every `Pipeline.run()`
without re-initializing everything.

#### `validate_input`

```python
validate_input(data: dict[str, Any]) -> None
```

Validates pipeline input data.

Validates that data:

- Each Component name actually exists in the Pipeline
- Each Component is not missing any input
- Each Component has only one input per input socket, if not variadic
- Each Component doesn't receive inputs that are already sent by another Component

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – A dictionary of inputs for the pipeline's components. Each key is a component name.

**Raises:**

- <code>ValueError</code> – If inputs are invalid according to the above.

#### `from_template`

```python
from_template(
    predefined_pipeline: PredefinedPipeline,
    template_params: dict[str, Any] | None = None,
) -> PipelineBase
```

Create a Pipeline from a predefined template. See `PredefinedPipeline` for available options.

**Parameters:**

- **predefined_pipeline** (<code>PredefinedPipeline</code>) – The predefined pipeline to use.
- **template_params** (<code>dict\[str, Any\] | None</code>) – An optional dictionary of parameters to use when rendering the pipeline template.

**Returns:**

- <code>PipelineBase</code> – An instance of `Pipeline`.

#### `validate_pipeline`

```python
validate_pipeline(priority_queue: FIFOPriorityQueue) -> None
```

Validate the pipeline to check if it is blocked or has no valid entry point.

**Parameters:**

- **priority_queue** (<code>FIFOPriorityQueue</code>) – Priority queue of component names.

**Raises:**

- <code>PipelineRuntimeError</code> – If the pipeline is blocked or has no valid entry point.

## `haystack.core.pipeline.pipeline`

### `Pipeline`

Bases: <code>PipelineBase</code>

Synchronous version of the orchestration engine.

Orchestrates component execution according to the execution graph, one after the other.

#### `__init__`

```python
__init__(
    metadata: dict[str, Any] | None = None,
    max_runs_per_component: int = 100,
    connection_type_validation: bool = True,
)
```

Creates the Pipeline.

**Parameters:**

- **metadata** (<code>dict\[str, Any\] | None</code>) – Arbitrary dictionary to store metadata about this `Pipeline`. Make sure all the values contained in
  this dictionary can be serialized and deserialized if you wish to save this `Pipeline` to file.
- **max_runs_per_component** (<code>int</code>) – How many times the `Pipeline` can run the same Component.
  If this limit is reached a `PipelineMaxComponentRuns` exception is raised.
  If not set defaults to 100 runs per Component.
- **connection_type_validation** (<code>bool</code>) – Whether the pipeline will validate the types of the connections.
  Defaults to True.

#### `run`

```python
run(
    data: dict[str, Any],
    include_outputs_from: set[str] | None = None,
    *,
    break_point: Breakpoint | AgentBreakpoint | None = None,
    pipeline_snapshot: PipelineSnapshot | None = None,
    snapshot_callback: SnapshotCallback | None = None
) -> dict[str, Any]
```

Runs the Pipeline with given input data.

Usage:

```python
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder

# Write documents to InMemoryDocumentStore
document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome.")
])

prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
"""

retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt_template)
llm = OpenAIGenerator(api_key=Secret.from_token(api_key))

rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Ask a question
question = "Who lives in Paris?"
results = rag_pipeline.run(
    {
        "retriever": {"query": question},
        "prompt_builder": {"question": question},
    }
)

print(results["llm"]["replies"])
# Jean lives in Paris
```

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – A dictionary of inputs for the pipeline's components. Each key is a component name
  and its value is a dictionary of that component's input parameters:

```
data = {
    "comp1": {"input1": 1, "input2": 2},
}
```

For convenience, this format is also supported when input names are unique:

```
data = {
    "input1": 1, "input2": 2,
}
```

- **include_outputs_from** (<code>set\[str\] | None</code>) – Set of component names whose individual outputs are to be
  included in the pipeline's output. For components that are
  invoked multiple times (in a loop), only the last-produced
  output is included.
- **break_point** (<code>Breakpoint | AgentBreakpoint | None</code>) – A set of breakpoints that can be used to debug the pipeline execution.
- **pipeline_snapshot** (<code>PipelineSnapshot | None</code>) – A dictionary containing a snapshot of a previously saved pipeline execution.
- **snapshot_callback** (<code>SnapshotCallback | None</code>) – Optional callback function that is invoked when a pipeline snapshot is created.
  The callback receives a `PipelineSnapshot` object and can return an optional string
  (e.g., a file path or identifier).
  If provided, the callback is used instead of the default file-saving behavior,
  allowing custom handling of snapshots (e.g., saving to a database, sending to a remote service).
  If not provided, the default behavior saves snapshots to a JSON file.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary where each entry corresponds to a component name
  and its output. If `include_outputs_from` is `None`, this dictionary
  will only contain the outputs of leaf components, i.e., components
  without outgoing connections.

**Raises:**

- <code>ValueError</code> – If invalid inputs are provided to the pipeline.
- <code>PipelineRuntimeError</code> – If the Pipeline contains cycles with unsupported connections that would cause
  it to get stuck and fail running.
  Or if a Component fails or returns output in an unsupported type.
- <code>PipelineMaxComponentRuns</code> – If a Component reaches the maximum number of times it can be run in this Pipeline.
- <code>PipelineBreakpointException</code> – When a pipeline_breakpoint is triggered. Contains the component name, state, and partial results.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the pipeline to a dictionary.

This is meant to be an intermediate representation but it can be also used to save a pipeline to file.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(
    data: dict[str, Any],
    callbacks: DeserializationCallbacks | None = None,
    **kwargs: Any
) -> T
```

Deserializes the pipeline from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.
- **callbacks** (<code>DeserializationCallbacks | None</code>) – Callbacks to invoke during deserialization.
- **kwargs** (<code>Any</code>) – `components`: a dictionary of `{name: instance}` to reuse instances of components instead of creating new
  ones.

**Returns:**

- <code>T</code> – Deserialized component.

#### `dumps`

```python
dumps(marshaller: Marshaller = DEFAULT_MARSHALLER) -> str
```

Returns the string representation of this pipeline according to the format dictated by the `Marshaller` in use.

**Parameters:**

- **marshaller** (<code>Marshaller</code>) – The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.

**Returns:**

- <code>str</code> – A string representing the pipeline.

#### `dump`

```python
dump(fp: TextIO, marshaller: Marshaller = DEFAULT_MARSHALLER) -> None
```

Writes the string representation of this pipeline to the file-like object passed in the `fp` argument.

**Parameters:**

- **fp** (<code>TextIO</code>) – A file-like object ready to be written to.
- **marshaller** (<code>Marshaller</code>) – The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.

#### `loads`

```python
loads(
    data: str | bytes | bytearray,
    marshaller: Marshaller = DEFAULT_MARSHALLER,
    callbacks: DeserializationCallbacks | None = None,
) -> T
```

Creates a `Pipeline` object from the string representation passed in the `data` argument.

**Parameters:**

- **data** (<code>str | bytes | bytearray</code>) – The string representation of the pipeline, can be `str`, `bytes` or `bytearray`.
- **marshaller** (<code>Marshaller</code>) – The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
- **callbacks** (<code>DeserializationCallbacks | None</code>) – Callbacks to invoke during deserialization.

**Returns:**

- <code>T</code> – A `Pipeline` object.

**Raises:**

- <code>DeserializationError</code> – If an error occurs during deserialization.

#### `load`

```python
load(
    fp: TextIO,
    marshaller: Marshaller = DEFAULT_MARSHALLER,
    callbacks: DeserializationCallbacks | None = None,
) -> T
```

Creates a `Pipeline` object a string representation.

The string representation is read from the file-like object passed in the `fp` argument.

**Parameters:**

- **fp** (<code>TextIO</code>) – A file-like object ready to be read from.
- **marshaller** (<code>Marshaller</code>) – The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
- **callbacks** (<code>DeserializationCallbacks | None</code>) – Callbacks to invoke during deserialization.

**Returns:**

- <code>T</code> – A `Pipeline` object.

**Raises:**

- <code>DeserializationError</code> – If an error occurs during deserialization.

#### `add_component`

```python
add_component(name: str, instance: Component) -> None
```

Add the given component to the pipeline.

Components are not connected to anything by default: use `Pipeline.connect()` to connect components together.
Component names must be unique, but component instances can be reused if needed.

**Parameters:**

- **name** (<code>str</code>) – The name of the component to add.
- **instance** (<code>Component</code>) – The component instance to add.

**Raises:**

- <code>ValueError</code> – If a component with the same name already exists.
- <code>PipelineValidationError</code> – If the given instance is not a component.

#### `remove_component`

```python
remove_component(name: str) -> Component
```

Remove and returns component from the pipeline.

Remove an existing component from the pipeline by providing its name.
All edges that connect to the component will also be deleted.

**Parameters:**

- **name** (<code>str</code>) – The name of the component to remove.

**Returns:**

- <code>Component</code> – The removed Component instance.

**Raises:**

- <code>ValueError</code> – If there is no component with that name already in the Pipeline.

#### `connect`

```python
connect(sender: str, receiver: str) -> PipelineBase
```

Connects two components together.

All components to connect must exist in the pipeline.
If connecting to a component that has several output connections, specify the inputs and output names as
'component_name.connections_name'.

**Parameters:**

- **sender** (<code>str</code>) – The component that delivers the value. This can be either just a component name or can be
  in the format `component_name.connection_name` if the component has multiple outputs.
- **receiver** (<code>str</code>) – The component that receives the value. This can be either just a component name or can be
  in the format `component_name.connection_name` if the component has multiple inputs.

**Returns:**

- <code>PipelineBase</code> – The Pipeline instance.

**Raises:**

- <code>PipelineConnectError</code> – If the two components cannot be connected (for example if one of the components is
  not present in the pipeline, or the connections don't match by type, and so on).

#### `get_component`

```python
get_component(name: str) -> Component
```

Get the component with the specified name from the pipeline.

**Parameters:**

- **name** (<code>str</code>) – The name of the component.

**Returns:**

- <code>Component</code> – The instance of that component.

**Raises:**

- <code>ValueError</code> – If a component with that name is not present in the pipeline.

#### `get_component_name`

```python
get_component_name(instance: Component) -> str
```

Returns the name of the Component instance if it has been added to this Pipeline or an empty string otherwise.

**Parameters:**

- **instance** (<code>Component</code>) – The Component instance to look for.

**Returns:**

- <code>str</code> – The name of the Component instance.

#### `inputs`

```python
inputs(
    include_components_with_connected_inputs: bool = False,
) -> dict[str, dict[str, Any]]
```

Returns a dictionary containing the inputs of a pipeline.

Each key in the dictionary corresponds to a component name, and its value is another dictionary that describes
the input sockets of that component, including their types and whether they are optional.

**Parameters:**

- **include_components_with_connected_inputs** (<code>bool</code>) – If `False`, only components that have disconnected input edges are
  included in the output.

**Returns:**

- <code>dict\[str, dict\[str, Any\]\]</code> – A dictionary where each key is a pipeline component name and each value is a dictionary of
  inputs sockets of that component.

#### `outputs`

```python
outputs(
    include_components_with_connected_outputs: bool = False,
) -> dict[str, dict[str, Any]]
```

Returns a dictionary containing the outputs of a pipeline.

Each key in the dictionary corresponds to a component name, and its value is another dictionary that describes
the output sockets of that component.

**Parameters:**

- **include_components_with_connected_outputs** (<code>bool</code>) – If `False`, only components that have disconnected output edges are
  included in the output.

**Returns:**

- <code>dict\[str, dict\[str, Any\]\]</code> – A dictionary where each key is a pipeline component name and each value is a dictionary of
  output sockets of that component.

#### `show`

```python
show(
    *,
    server_url: str = "https://mermaid.ink",
    params: dict | None = None,
    timeout: int = 30,
    super_component_expansion: bool = False
) -> None
```

Display an image representing this `Pipeline` in a Jupyter notebook.

This function generates a diagram of the `Pipeline` using a Mermaid server and displays it directly in
the notebook.

**Parameters:**

- **server_url** (<code>str</code>) – The base URL of the Mermaid server used for rendering (default: 'https://mermaid.ink').
  See https://github.com/jihchi/mermaid.ink and https://github.com/mermaid-js/mermaid-live-editor for more
  info on how to set up your own Mermaid server.
- **params** (<code>dict | None</code>) – Dictionary of customization parameters to modify the output. Refer to Mermaid documentation for more details
  Supported keys:
  - format: Output format ('img', 'svg', or 'pdf'). Default: 'img'.
  - type: Image type for /img endpoint ('jpeg', 'png', 'webp'). Default: 'png'.
  - theme: Mermaid theme ('default', 'neutral', 'dark', 'forest'). Default: 'neutral'.
  - bgColor: Background color in hexadecimal (e.g., 'FFFFFF') or named format (e.g., '!white').
  - width: Width of the output image (integer).
  - height: Height of the output image (integer).
  - scale: Scaling factor (1–3). Only applicable if 'width' or 'height' is specified.
  - fit: Whether to fit the diagram size to the page (PDF only, boolean).
  - paper: Paper size for PDFs (e.g., 'a4', 'a3'). Ignored if 'fit' is true.
  - landscape: Landscape orientation for PDFs (boolean). Ignored if 'fit' is true.
- **timeout** (<code>int</code>) – Timeout in seconds for the request to the Mermaid server.
- **super_component_expansion** (<code>bool</code>) – If set to True and the pipeline contains SuperComponents the diagram will show the internal structure of
  super-components as if they were components part of the pipeline instead of a "black-box".
  Otherwise, only the super-component itself will be displayed.

**Raises:**

- <code>PipelineDrawingError</code> – If the function is called outside of a Jupyter notebook or if there is an issue with rendering.

#### `draw`

```python
draw(
    *,
    path: Path,
    server_url: str = "https://mermaid.ink",
    params: dict | None = None,
    timeout: int = 30,
    super_component_expansion: bool = False
) -> None
```

Save an image representing this `Pipeline` to the specified file path.

This function generates a diagram of the `Pipeline` using the Mermaid server and saves it to the provided path.

**Parameters:**

- **path** (<code>Path</code>) – The file path where the generated image will be saved.
- **server_url** (<code>str</code>) – The base URL of the Mermaid server used for rendering (default: 'https://mermaid.ink').
  See https://github.com/jihchi/mermaid.ink and https://github.com/mermaid-js/mermaid-live-editor for more
  info on how to set up your own Mermaid server.
- **params** (<code>dict | None</code>) – Dictionary of customization parameters to modify the output. Refer to Mermaid documentation for more details
  Supported keys:
  - format: Output format ('img', 'svg', or 'pdf'). Default: 'img'.
  - type: Image type for /img endpoint ('jpeg', 'png', 'webp'). Default: 'png'.
  - theme: Mermaid theme ('default', 'neutral', 'dark', 'forest'). Default: 'neutral'.
  - bgColor: Background color in hexadecimal (e.g., 'FFFFFF') or named format (e.g., '!white').
  - width: Width of the output image (integer).
  - height: Height of the output image (integer).
  - scale: Scaling factor (1–3). Only applicable if 'width' or 'height' is specified.
  - fit: Whether to fit the diagram size to the page (PDF only, boolean).
  - paper: Paper size for PDFs (e.g., 'a4', 'a3'). Ignored if 'fit' is true.
  - landscape: Landscape orientation for PDFs (boolean). Ignored if 'fit' is true.
- **timeout** (<code>int</code>) – Timeout in seconds for the request to the Mermaid server.
- **super_component_expansion** (<code>bool</code>) – If set to True and the pipeline contains SuperComponents the diagram will show the internal structure of
  super-components as if they were components part of the pipeline instead of a "black-box".
  Otherwise, only the super-component itself will be displayed.

**Raises:**

- <code>PipelineDrawingError</code> – If there is an issue with rendering or saving the image.

#### `walk`

```python
walk() -> Iterator[tuple[str, Component]]
```

Visits each component in the pipeline exactly once and yields its name and instance.

No guarantees are provided on the visiting order.

**Returns:**

- <code>Iterator\[tuple\[str, Component\]\]</code> – An iterator of tuples of component name and component instance.

#### `warm_up`

```python
warm_up() -> None
```

Make sure all nodes are warm.

It's the node's responsibility to make sure this method can be called at every `Pipeline.run()`
without re-initializing everything.

#### `validate_input`

```python
validate_input(data: dict[str, Any]) -> None
```

Validates pipeline input data.

Validates that data:

- Each Component name actually exists in the Pipeline
- Each Component is not missing any input
- Each Component has only one input per input socket, if not variadic
- Each Component doesn't receive inputs that are already sent by another Component

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – A dictionary of inputs for the pipeline's components. Each key is a component name.

**Raises:**

- <code>ValueError</code> – If inputs are invalid according to the above.

#### `from_template`

```python
from_template(
    predefined_pipeline: PredefinedPipeline,
    template_params: dict[str, Any] | None = None,
) -> PipelineBase
```

Create a Pipeline from a predefined template. See `PredefinedPipeline` for available options.

**Parameters:**

- **predefined_pipeline** (<code>PredefinedPipeline</code>) – The predefined pipeline to use.
- **template_params** (<code>dict\[str, Any\] | None</code>) – An optional dictionary of parameters to use when rendering the pipeline template.

**Returns:**

- <code>PipelineBase</code> – An instance of `Pipeline`.

#### `validate_pipeline`

```python
validate_pipeline(priority_queue: FIFOPriorityQueue) -> None
```

Validate the pipeline to check if it is blocked or has no valid entry point.

**Parameters:**

- **priority_queue** (<code>FIFOPriorityQueue</code>) – Priority queue of component names.

**Raises:**

- <code>PipelineRuntimeError</code> – If the pipeline is blocked or has no valid entry point.
