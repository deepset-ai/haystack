---
title: "Pipeline"
id: pipeline-api
description: "Arranges components and integrations in flow."
slug: "/pipeline-api"
---

<a id="async_pipeline"></a>

# Module async\_pipeline

<a id="async_pipeline.AsyncPipeline"></a>

## AsyncPipeline

Asynchronous version of the Pipeline orchestration engine.

Manages components in a pipeline allowing for concurrent processing when the pipeline's execution graph permits.
This enables efficient processing of components by minimizing idle time and maximizing resource utilization.

<a id="async_pipeline.AsyncPipeline.run_async_generator"></a>

#### AsyncPipeline.run\_async\_generator

```python
async def run_async_generator(
        data: dict[str, Any],
        include_outputs_from: Optional[set[str]] = None,
        concurrency_limit: int = 4) -> AsyncIterator[dict[str, Any]]
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

**Arguments**:

- `data`: Initial input data to the pipeline.
- `concurrency_limit`: The maximum number of components that are allowed to run concurrently.
- `include_outputs_from`: Set of component names whose individual outputs are to be
included in the pipeline's output. For components that are
invoked multiple times (in a loop), only the last-produced
output is included.

**Raises**:

- `ValueError`: If invalid inputs are provided to the pipeline.
- `PipelineMaxComponentRuns`: If a component exceeds the maximum number of allowed executions within the pipeline.
- `PipelineRuntimeError`: If the Pipeline contains cycles with unsupported connections that would cause
it to get stuck and fail running.
Or if a Component fails or returns output in an unsupported type.

**Returns**:

An async iterator containing partial (and final) outputs.

<a id="async_pipeline.AsyncPipeline.run_async"></a>

#### AsyncPipeline.run\_async

```python
async def run_async(data: dict[str, Any],
                    include_outputs_from: Optional[set[str]] = None,
                    concurrency_limit: int = 4) -> dict[str, Any]
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
# _name=None, _meta={'model': 'gpt-4o-mini-2024-07-18', 'index': 0, 'finish_reason': 'stop', 'usage':
# {'completion_tokens': 6, 'prompt_tokens': 69, 'total_tokens': 75,
# 'completion_tokens_details': CompletionTokensDetails(accepted_prediction_tokens=0,
# audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), 'prompt_tokens_details':
# PromptTokensDetails(audio_tokens=0, cached_tokens=0)}})]
```

**Arguments**:

- `data`: A dictionary of inputs for the pipeline's components. Each key is a component name
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
- `include_outputs_from`: Set of component names whose individual outputs are to be
included in the pipeline's output. For components that are
invoked multiple times (in a loop), only the last-produced
output is included.
- `concurrency_limit`: The maximum number of components that should be allowed to run concurrently.

**Raises**:

- `ValueError`: If invalid inputs are provided to the pipeline.
- `PipelineRuntimeError`: If the Pipeline contains cycles with unsupported connections that would cause
it to get stuck and fail running.
Or if a Component fails or returns output in an unsupported type.
- `PipelineMaxComponentRuns`: If a Component reaches the maximum number of times it can be run in this Pipeline.

**Returns**:

A dictionary where each entry corresponds to a component name
and its output. If `include_outputs_from` is `None`, this dictionary
will only contain the outputs of leaf components, i.e., components
without outgoing connections.

<a id="async_pipeline.AsyncPipeline.run"></a>

#### AsyncPipeline.run

```python
def run(data: dict[str, Any],
        include_outputs_from: Optional[set[str]] = None,
        concurrency_limit: int = 4) -> dict[str, Any]
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
# _name=None, _meta={'model': 'gpt-4o-mini-2024-07-18', 'index': 0, 'finish_reason': 'stop', 'usage':
# {'completion_tokens': 6, 'prompt_tokens': 69, 'total_tokens': 75, 'completion_tokens_details':
# CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0,
# rejected_prediction_tokens=0), 'prompt_tokens_details': PromptTokensDetails(audio_tokens=0,
# cached_tokens=0)}})]
```

**Arguments**:

- `data`: A dictionary of inputs for the pipeline's components. Each key is a component name
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
- `include_outputs_from`: Set of component names whose individual outputs are to be
included in the pipeline's output. For components that are
invoked multiple times (in a loop), only the last-produced
output is included.
- `concurrency_limit`: The maximum number of components that should be allowed to run concurrently.

**Raises**:

- `ValueError`: If invalid inputs are provided to the pipeline.
- `PipelineRuntimeError`: If the Pipeline contains cycles with unsupported connections that would cause
it to get stuck and fail running.
Or if a Component fails or returns output in an unsupported type.
- `PipelineMaxComponentRuns`: If a Component reaches the maximum number of times it can be run in this Pipeline.
- `RuntimeError`: If called from within an async context. Use `run_async` instead.

**Returns**:

A dictionary where each entry corresponds to a component name
and its output. If `include_outputs_from` is `None`, this dictionary
will only contain the outputs of leaf components, i.e., components
without outgoing connections.

<a id="async_pipeline.AsyncPipeline.__init__"></a>

#### AsyncPipeline.\_\_init\_\_

```python
def __init__(metadata: Optional[dict[str, Any]] = None,
             max_runs_per_component: int = 100,
             connection_type_validation: bool = True)
```

Creates the Pipeline.

**Arguments**:

- `metadata`: Arbitrary dictionary to store metadata about this `Pipeline`. Make sure all the values contained in
this dictionary can be serialized and deserialized if you wish to save this `Pipeline` to file.
- `max_runs_per_component`: How many times the `Pipeline` can run the same Component.
If this limit is reached a `PipelineMaxComponentRuns` exception is raised.
If not set defaults to 100 runs per Component.
- `connection_type_validation`: Whether the pipeline will validate the types of the connections.
Defaults to True.

<a id="async_pipeline.AsyncPipeline.__eq__"></a>

#### AsyncPipeline.\_\_eq\_\_

```python
def __eq__(other: object) -> bool
```

Pipeline equality is defined by their type and the equality of their serialized form.

Pipelines of the same type share every metadata, node and edge, but they're not required to use
the same node instances: this allows pipeline saved and then loaded back to be equal to themselves.

<a id="async_pipeline.AsyncPipeline.__repr__"></a>

#### AsyncPipeline.\_\_repr\_\_

```python
def __repr__() -> str
```

Returns a text representation of the Pipeline.

<a id="async_pipeline.AsyncPipeline.to_dict"></a>

#### AsyncPipeline.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the pipeline to a dictionary.

This is meant to be an intermediate representation but it can be also used to save a pipeline to file.

**Returns**:

Dictionary with serialized data.

<a id="async_pipeline.AsyncPipeline.from_dict"></a>

#### AsyncPipeline.from\_dict

```python
@classmethod
def from_dict(cls: type[T],
              data: dict[str, Any],
              callbacks: Optional[DeserializationCallbacks] = None,
              **kwargs: Any) -> T
```

Deserializes the pipeline from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.
- `callbacks`: Callbacks to invoke during deserialization.
- `kwargs`: `components`: a dictionary of `{name: instance}` to reuse instances of components instead of creating new
ones.

**Returns**:

Deserialized component.

<a id="async_pipeline.AsyncPipeline.dumps"></a>

#### AsyncPipeline.dumps

```python
def dumps(marshaller: Marshaller = DEFAULT_MARSHALLER) -> str
```

Returns the string representation of this pipeline according to the format dictated by the `Marshaller` in use.

**Arguments**:

- `marshaller`: The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.

**Returns**:

A string representing the pipeline.

<a id="async_pipeline.AsyncPipeline.dump"></a>

#### AsyncPipeline.dump

```python
def dump(fp: TextIO, marshaller: Marshaller = DEFAULT_MARSHALLER) -> None
```

Writes the string representation of this pipeline to the file-like object passed in the `fp` argument.

**Arguments**:

- `fp`: A file-like object ready to be written to.
- `marshaller`: The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.

<a id="async_pipeline.AsyncPipeline.loads"></a>

#### AsyncPipeline.loads

```python
@classmethod
def loads(cls: type[T],
          data: Union[str, bytes, bytearray],
          marshaller: Marshaller = DEFAULT_MARSHALLER,
          callbacks: Optional[DeserializationCallbacks] = None) -> T
```

Creates a `Pipeline` object from the string representation passed in the `data` argument.

**Arguments**:

- `data`: The string representation of the pipeline, can be `str`, `bytes` or `bytearray`.
- `marshaller`: The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
- `callbacks`: Callbacks to invoke during deserialization.

**Raises**:

- `DeserializationError`: If an error occurs during deserialization.

**Returns**:

A `Pipeline` object.

<a id="async_pipeline.AsyncPipeline.load"></a>

#### AsyncPipeline.load

```python
@classmethod
def load(cls: type[T],
         fp: TextIO,
         marshaller: Marshaller = DEFAULT_MARSHALLER,
         callbacks: Optional[DeserializationCallbacks] = None) -> T
```

Creates a `Pipeline` object a string representation.

The string representation is read from the file-like object passed in the `fp` argument.

**Arguments**:

- `fp`: A file-like object ready to be read from.
- `marshaller`: The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
- `callbacks`: Callbacks to invoke during deserialization.

**Raises**:

- `DeserializationError`: If an error occurs during deserialization.

**Returns**:

A `Pipeline` object.

<a id="async_pipeline.AsyncPipeline.add_component"></a>

#### AsyncPipeline.add\_component

```python
def add_component(name: str, instance: Component) -> None
```

Add the given component to the pipeline.

Components are not connected to anything by default: use `Pipeline.connect()` to connect components together.
Component names must be unique, but component instances can be reused if needed.

**Arguments**:

- `name`: The name of the component to add.
- `instance`: The component instance to add.

**Raises**:

- `ValueError`: If a component with the same name already exists.
- `PipelineValidationError`: If the given instance is not a component.

<a id="async_pipeline.AsyncPipeline.remove_component"></a>

#### AsyncPipeline.remove\_component

```python
def remove_component(name: str) -> Component
```

Remove and returns component from the pipeline.

Remove an existing component from the pipeline by providing its name.
All edges that connect to the component will also be deleted.

**Arguments**:

- `name`: The name of the component to remove.

**Raises**:

- `ValueError`: If there is no component with that name already in the Pipeline.

**Returns**:

The removed Component instance.

<a id="async_pipeline.AsyncPipeline.connect"></a>

#### AsyncPipeline.connect

```python
def connect(sender: str, receiver: str) -> "PipelineBase"
```

Connects two components together.

All components to connect must exist in the pipeline.
If connecting to a component that has several output connections, specify the inputs and output names as
'component_name.connections_name'.

**Arguments**:

- `sender`: The component that delivers the value. This can be either just a component name or can be
in the format `component_name.connection_name` if the component has multiple outputs.
- `receiver`: The component that receives the value. This can be either just a component name or can be
in the format `component_name.connection_name` if the component has multiple inputs.

**Raises**:

- `PipelineConnectError`: If the two components cannot be connected (for example if one of the components is
not present in the pipeline, or the connections don't match by type, and so on).

**Returns**:

The Pipeline instance.

<a id="async_pipeline.AsyncPipeline.get_component"></a>

#### AsyncPipeline.get\_component

```python
def get_component(name: str) -> Component
```

Get the component with the specified name from the pipeline.

**Arguments**:

- `name`: The name of the component.

**Raises**:

- `ValueError`: If a component with that name is not present in the pipeline.

**Returns**:

The instance of that component.

<a id="async_pipeline.AsyncPipeline.get_component_name"></a>

#### AsyncPipeline.get\_component\_name

```python
def get_component_name(instance: Component) -> str
```

Returns the name of the Component instance if it has been added to this Pipeline or an empty string otherwise.

**Arguments**:

- `instance`: The Component instance to look for.

**Returns**:

The name of the Component instance.

<a id="async_pipeline.AsyncPipeline.inputs"></a>

#### AsyncPipeline.inputs

```python
def inputs(
    include_components_with_connected_inputs: bool = False
) -> dict[str, dict[str, Any]]
```

Returns a dictionary containing the inputs of a pipeline.

Each key in the dictionary corresponds to a component name, and its value is another dictionary that describes
the input sockets of that component, including their types and whether they are optional.

**Arguments**:

- `include_components_with_connected_inputs`: If `False`, only components that have disconnected input edges are
included in the output.

**Returns**:

A dictionary where each key is a pipeline component name and each value is a dictionary of
inputs sockets of that component.

<a id="async_pipeline.AsyncPipeline.outputs"></a>

#### AsyncPipeline.outputs

```python
def outputs(
    include_components_with_connected_outputs: bool = False
) -> dict[str, dict[str, Any]]
```

Returns a dictionary containing the outputs of a pipeline.

Each key in the dictionary corresponds to a component name, and its value is another dictionary that describes
the output sockets of that component.

**Arguments**:

- `include_components_with_connected_outputs`: If `False`, only components that have disconnected output edges are
included in the output.

**Returns**:

A dictionary where each key is a pipeline component name and each value is a dictionary of
output sockets of that component.

<a id="async_pipeline.AsyncPipeline.show"></a>

#### AsyncPipeline.show

```python
def show(*,
         server_url: str = "https://mermaid.ink",
         params: Optional[dict] = None,
         timeout: int = 30,
         super_component_expansion: bool = False) -> None
```

Display an image representing this `Pipeline` in a Jupyter notebook.

This function generates a diagram of the `Pipeline` using a Mermaid server and displays it directly in
the notebook.

**Arguments**:

- `server_url`: The base URL of the Mermaid server used for rendering (default: 'https://mermaid.ink').
See https://github.com/jihchi/mermaid.ink and https://github.com/mermaid-js/mermaid-live-editor for more
info on how to set up your own Mermaid server.
- `params`: Dictionary of customization parameters to modify the output. Refer to Mermaid documentation for more details
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
- `timeout`: Timeout in seconds for the request to the Mermaid server.
- `super_component_expansion`: If set to True and the pipeline contains SuperComponents the diagram will show the internal structure of
super-components as if they were components part of the pipeline instead of a "black-box".
Otherwise, only the super-component itself will be displayed.

**Raises**:

- `PipelineDrawingError`: If the function is called outside of a Jupyter notebook or if there is an issue with rendering.

<a id="async_pipeline.AsyncPipeline.draw"></a>

#### AsyncPipeline.draw

```python
def draw(*,
         path: Path,
         server_url: str = "https://mermaid.ink",
         params: Optional[dict] = None,
         timeout: int = 30,
         super_component_expansion: bool = False) -> None
```

Save an image representing this `Pipeline` to the specified file path.

This function generates a diagram of the `Pipeline` using the Mermaid server and saves it to the provided path.

**Arguments**:

- `path`: The file path where the generated image will be saved.
- `server_url`: The base URL of the Mermaid server used for rendering (default: 'https://mermaid.ink').
See https://github.com/jihchi/mermaid.ink and https://github.com/mermaid-js/mermaid-live-editor for more
info on how to set up your own Mermaid server.
- `params`: Dictionary of customization parameters to modify the output. Refer to Mermaid documentation for more details
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
- `timeout`: Timeout in seconds for the request to the Mermaid server.
- `super_component_expansion`: If set to True and the pipeline contains SuperComponents the diagram will show the internal structure of
super-components as if they were components part of the pipeline instead of a "black-box".
Otherwise, only the super-component itself will be displayed.

**Raises**:

- `PipelineDrawingError`: If there is an issue with rendering or saving the image.

<a id="async_pipeline.AsyncPipeline.walk"></a>

#### AsyncPipeline.walk

```python
def walk() -> Iterator[tuple[str, Component]]
```

Visits each component in the pipeline exactly once and yields its name and instance.

No guarantees are provided on the visiting order.

**Returns**:

An iterator of tuples of component name and component instance.

<a id="async_pipeline.AsyncPipeline.warm_up"></a>

#### AsyncPipeline.warm\_up

```python
def warm_up() -> None
```

Make sure all nodes are warm.

It's the node's responsibility to make sure this method can be called at every `Pipeline.run()`
without re-initializing everything.

<a id="async_pipeline.AsyncPipeline.validate_input"></a>

#### AsyncPipeline.validate\_input

```python
def validate_input(data: dict[str, Any]) -> None
```

Validates pipeline input data.

Validates that data:
* Each Component name actually exists in the Pipeline
* Each Component is not missing any input
* Each Component has only one input per input socket, if not variadic
* Each Component doesn't receive inputs that are already sent by another Component

**Arguments**:

- `data`: A dictionary of inputs for the pipeline's components. Each key is a component name.

**Raises**:

- `ValueError`: If inputs are invalid according to the above.

<a id="async_pipeline.AsyncPipeline.from_template"></a>

#### AsyncPipeline.from\_template

```python
@classmethod
def from_template(
        cls,
        predefined_pipeline: PredefinedPipeline,
        template_params: Optional[dict[str, Any]] = None) -> "PipelineBase"
```

Create a Pipeline from a predefined template. See `PredefinedPipeline` for available options.

**Arguments**:

- `predefined_pipeline`: The predefined pipeline to use.
- `template_params`: An optional dictionary of parameters to use when rendering the pipeline template.

**Returns**:

An instance of `Pipeline`.

<a id="async_pipeline.AsyncPipeline.validate_pipeline"></a>

#### AsyncPipeline.validate\_pipeline

```python
@staticmethod
def validate_pipeline(priority_queue: FIFOPriorityQueue) -> None
```

Validate the pipeline to check if it is blocked or has no valid entry point.

**Arguments**:

- `priority_queue`: Priority queue of component names.

**Raises**:

- `PipelineRuntimeError`: If the pipeline is blocked or has no valid entry point.

<a id="pipeline"></a>

# Module pipeline

<a id="pipeline.Pipeline"></a>

## Pipeline

Synchronous version of the orchestration engine.

Orchestrates component execution according to the execution graph, one after the other.

<a id="pipeline.Pipeline.run"></a>

#### Pipeline.run

```python
def run(data: dict[str, Any],
        include_outputs_from: Optional[set[str]] = None,
        *,
        break_point: Optional[Union[Breakpoint, AgentBreakpoint]] = None,
        pipeline_snapshot: Optional[PipelineSnapshot] = None
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

**Arguments**:

- `data`: A dictionary of inputs for the pipeline's components. Each key is a component name
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
- `include_outputs_from`: Set of component names whose individual outputs are to be
included in the pipeline's output. For components that are
invoked multiple times (in a loop), only the last-produced
output is included.
- `break_point`: A set of breakpoints that can be used to debug the pipeline execution.
- `pipeline_snapshot`: A dictionary containing a snapshot of a previously saved pipeline execution.

**Raises**:

- `ValueError`: If invalid inputs are provided to the pipeline.
- `PipelineRuntimeError`: If the Pipeline contains cycles with unsupported connections that would cause
it to get stuck and fail running.
Or if a Component fails or returns output in an unsupported type.
- `PipelineMaxComponentRuns`: If a Component reaches the maximum number of times it can be run in this Pipeline.
- `PipelineBreakpointException`: When a pipeline_breakpoint is triggered. Contains the component name, state, and partial results.

**Returns**:

A dictionary where each entry corresponds to a component name
and its output. If `include_outputs_from` is `None`, this dictionary
will only contain the outputs of leaf components, i.e., components
without outgoing connections.

<a id="pipeline.Pipeline.__init__"></a>

#### Pipeline.\_\_init\_\_

```python
def __init__(metadata: Optional[dict[str, Any]] = None,
             max_runs_per_component: int = 100,
             connection_type_validation: bool = True)
```

Creates the Pipeline.

**Arguments**:

- `metadata`: Arbitrary dictionary to store metadata about this `Pipeline`. Make sure all the values contained in
this dictionary can be serialized and deserialized if you wish to save this `Pipeline` to file.
- `max_runs_per_component`: How many times the `Pipeline` can run the same Component.
If this limit is reached a `PipelineMaxComponentRuns` exception is raised.
If not set defaults to 100 runs per Component.
- `connection_type_validation`: Whether the pipeline will validate the types of the connections.
Defaults to True.

<a id="pipeline.Pipeline.__eq__"></a>

#### Pipeline.\_\_eq\_\_

```python
def __eq__(other: object) -> bool
```

Pipeline equality is defined by their type and the equality of their serialized form.

Pipelines of the same type share every metadata, node and edge, but they're not required to use
the same node instances: this allows pipeline saved and then loaded back to be equal to themselves.

<a id="pipeline.Pipeline.__repr__"></a>

#### Pipeline.\_\_repr\_\_

```python
def __repr__() -> str
```

Returns a text representation of the Pipeline.

<a id="pipeline.Pipeline.to_dict"></a>

#### Pipeline.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the pipeline to a dictionary.

This is meant to be an intermediate representation but it can be also used to save a pipeline to file.

**Returns**:

Dictionary with serialized data.

<a id="pipeline.Pipeline.from_dict"></a>

#### Pipeline.from\_dict

```python
@classmethod
def from_dict(cls: type[T],
              data: dict[str, Any],
              callbacks: Optional[DeserializationCallbacks] = None,
              **kwargs: Any) -> T
```

Deserializes the pipeline from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.
- `callbacks`: Callbacks to invoke during deserialization.
- `kwargs`: `components`: a dictionary of `{name: instance}` to reuse instances of components instead of creating new
ones.

**Returns**:

Deserialized component.

<a id="pipeline.Pipeline.dumps"></a>

#### Pipeline.dumps

```python
def dumps(marshaller: Marshaller = DEFAULT_MARSHALLER) -> str
```

Returns the string representation of this pipeline according to the format dictated by the `Marshaller` in use.

**Arguments**:

- `marshaller`: The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.

**Returns**:

A string representing the pipeline.

<a id="pipeline.Pipeline.dump"></a>

#### Pipeline.dump

```python
def dump(fp: TextIO, marshaller: Marshaller = DEFAULT_MARSHALLER) -> None
```

Writes the string representation of this pipeline to the file-like object passed in the `fp` argument.

**Arguments**:

- `fp`: A file-like object ready to be written to.
- `marshaller`: The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.

<a id="pipeline.Pipeline.loads"></a>

#### Pipeline.loads

```python
@classmethod
def loads(cls: type[T],
          data: Union[str, bytes, bytearray],
          marshaller: Marshaller = DEFAULT_MARSHALLER,
          callbacks: Optional[DeserializationCallbacks] = None) -> T
```

Creates a `Pipeline` object from the string representation passed in the `data` argument.

**Arguments**:

- `data`: The string representation of the pipeline, can be `str`, `bytes` or `bytearray`.
- `marshaller`: The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
- `callbacks`: Callbacks to invoke during deserialization.

**Raises**:

- `DeserializationError`: If an error occurs during deserialization.

**Returns**:

A `Pipeline` object.

<a id="pipeline.Pipeline.load"></a>

#### Pipeline.load

```python
@classmethod
def load(cls: type[T],
         fp: TextIO,
         marshaller: Marshaller = DEFAULT_MARSHALLER,
         callbacks: Optional[DeserializationCallbacks] = None) -> T
```

Creates a `Pipeline` object a string representation.

The string representation is read from the file-like object passed in the `fp` argument.

**Arguments**:

- `fp`: A file-like object ready to be read from.
- `marshaller`: The Marshaller used to create the string representation. Defaults to `YamlMarshaller`.
- `callbacks`: Callbacks to invoke during deserialization.

**Raises**:

- `DeserializationError`: If an error occurs during deserialization.

**Returns**:

A `Pipeline` object.

<a id="pipeline.Pipeline.add_component"></a>

#### Pipeline.add\_component

```python
def add_component(name: str, instance: Component) -> None
```

Add the given component to the pipeline.

Components are not connected to anything by default: use `Pipeline.connect()` to connect components together.
Component names must be unique, but component instances can be reused if needed.

**Arguments**:

- `name`: The name of the component to add.
- `instance`: The component instance to add.

**Raises**:

- `ValueError`: If a component with the same name already exists.
- `PipelineValidationError`: If the given instance is not a component.

<a id="pipeline.Pipeline.remove_component"></a>

#### Pipeline.remove\_component

```python
def remove_component(name: str) -> Component
```

Remove and returns component from the pipeline.

Remove an existing component from the pipeline by providing its name.
All edges that connect to the component will also be deleted.

**Arguments**:

- `name`: The name of the component to remove.

**Raises**:

- `ValueError`: If there is no component with that name already in the Pipeline.

**Returns**:

The removed Component instance.

<a id="pipeline.Pipeline.connect"></a>

#### Pipeline.connect

```python
def connect(sender: str, receiver: str) -> "PipelineBase"
```

Connects two components together.

All components to connect must exist in the pipeline.
If connecting to a component that has several output connections, specify the inputs and output names as
'component_name.connections_name'.

**Arguments**:

- `sender`: The component that delivers the value. This can be either just a component name or can be
in the format `component_name.connection_name` if the component has multiple outputs.
- `receiver`: The component that receives the value. This can be either just a component name or can be
in the format `component_name.connection_name` if the component has multiple inputs.

**Raises**:

- `PipelineConnectError`: If the two components cannot be connected (for example if one of the components is
not present in the pipeline, or the connections don't match by type, and so on).

**Returns**:

The Pipeline instance.

<a id="pipeline.Pipeline.get_component"></a>

#### Pipeline.get\_component

```python
def get_component(name: str) -> Component
```

Get the component with the specified name from the pipeline.

**Arguments**:

- `name`: The name of the component.

**Raises**:

- `ValueError`: If a component with that name is not present in the pipeline.

**Returns**:

The instance of that component.

<a id="pipeline.Pipeline.get_component_name"></a>

#### Pipeline.get\_component\_name

```python
def get_component_name(instance: Component) -> str
```

Returns the name of the Component instance if it has been added to this Pipeline or an empty string otherwise.

**Arguments**:

- `instance`: The Component instance to look for.

**Returns**:

The name of the Component instance.

<a id="pipeline.Pipeline.inputs"></a>

#### Pipeline.inputs

```python
def inputs(
    include_components_with_connected_inputs: bool = False
) -> dict[str, dict[str, Any]]
```

Returns a dictionary containing the inputs of a pipeline.

Each key in the dictionary corresponds to a component name, and its value is another dictionary that describes
the input sockets of that component, including their types and whether they are optional.

**Arguments**:

- `include_components_with_connected_inputs`: If `False`, only components that have disconnected input edges are
included in the output.

**Returns**:

A dictionary where each key is a pipeline component name and each value is a dictionary of
inputs sockets of that component.

<a id="pipeline.Pipeline.outputs"></a>

#### Pipeline.outputs

```python
def outputs(
    include_components_with_connected_outputs: bool = False
) -> dict[str, dict[str, Any]]
```

Returns a dictionary containing the outputs of a pipeline.

Each key in the dictionary corresponds to a component name, and its value is another dictionary that describes
the output sockets of that component.

**Arguments**:

- `include_components_with_connected_outputs`: If `False`, only components that have disconnected output edges are
included in the output.

**Returns**:

A dictionary where each key is a pipeline component name and each value is a dictionary of
output sockets of that component.

<a id="pipeline.Pipeline.show"></a>

#### Pipeline.show

```python
def show(*,
         server_url: str = "https://mermaid.ink",
         params: Optional[dict] = None,
         timeout: int = 30,
         super_component_expansion: bool = False) -> None
```

Display an image representing this `Pipeline` in a Jupyter notebook.

This function generates a diagram of the `Pipeline` using a Mermaid server and displays it directly in
the notebook.

**Arguments**:

- `server_url`: The base URL of the Mermaid server used for rendering (default: 'https://mermaid.ink').
See https://github.com/jihchi/mermaid.ink and https://github.com/mermaid-js/mermaid-live-editor for more
info on how to set up your own Mermaid server.
- `params`: Dictionary of customization parameters to modify the output. Refer to Mermaid documentation for more details
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
- `timeout`: Timeout in seconds for the request to the Mermaid server.
- `super_component_expansion`: If set to True and the pipeline contains SuperComponents the diagram will show the internal structure of
super-components as if they were components part of the pipeline instead of a "black-box".
Otherwise, only the super-component itself will be displayed.

**Raises**:

- `PipelineDrawingError`: If the function is called outside of a Jupyter notebook or if there is an issue with rendering.

<a id="pipeline.Pipeline.draw"></a>

#### Pipeline.draw

```python
def draw(*,
         path: Path,
         server_url: str = "https://mermaid.ink",
         params: Optional[dict] = None,
         timeout: int = 30,
         super_component_expansion: bool = False) -> None
```

Save an image representing this `Pipeline` to the specified file path.

This function generates a diagram of the `Pipeline` using the Mermaid server and saves it to the provided path.

**Arguments**:

- `path`: The file path where the generated image will be saved.
- `server_url`: The base URL of the Mermaid server used for rendering (default: 'https://mermaid.ink').
See https://github.com/jihchi/mermaid.ink and https://github.com/mermaid-js/mermaid-live-editor for more
info on how to set up your own Mermaid server.
- `params`: Dictionary of customization parameters to modify the output. Refer to Mermaid documentation for more details
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
- `timeout`: Timeout in seconds for the request to the Mermaid server.
- `super_component_expansion`: If set to True and the pipeline contains SuperComponents the diagram will show the internal structure of
super-components as if they were components part of the pipeline instead of a "black-box".
Otherwise, only the super-component itself will be displayed.

**Raises**:

- `PipelineDrawingError`: If there is an issue with rendering or saving the image.

<a id="pipeline.Pipeline.walk"></a>

#### Pipeline.walk

```python
def walk() -> Iterator[tuple[str, Component]]
```

Visits each component in the pipeline exactly once and yields its name and instance.

No guarantees are provided on the visiting order.

**Returns**:

An iterator of tuples of component name and component instance.

<a id="pipeline.Pipeline.warm_up"></a>

#### Pipeline.warm\_up

```python
def warm_up() -> None
```

Make sure all nodes are warm.

It's the node's responsibility to make sure this method can be called at every `Pipeline.run()`
without re-initializing everything.

<a id="pipeline.Pipeline.validate_input"></a>

#### Pipeline.validate\_input

```python
def validate_input(data: dict[str, Any]) -> None
```

Validates pipeline input data.

Validates that data:
* Each Component name actually exists in the Pipeline
* Each Component is not missing any input
* Each Component has only one input per input socket, if not variadic
* Each Component doesn't receive inputs that are already sent by another Component

**Arguments**:

- `data`: A dictionary of inputs for the pipeline's components. Each key is a component name.

**Raises**:

- `ValueError`: If inputs are invalid according to the above.

<a id="pipeline.Pipeline.from_template"></a>

#### Pipeline.from\_template

```python
@classmethod
def from_template(
        cls,
        predefined_pipeline: PredefinedPipeline,
        template_params: Optional[dict[str, Any]] = None) -> "PipelineBase"
```

Create a Pipeline from a predefined template. See `PredefinedPipeline` for available options.

**Arguments**:

- `predefined_pipeline`: The predefined pipeline to use.
- `template_params`: An optional dictionary of parameters to use when rendering the pipeline template.

**Returns**:

An instance of `Pipeline`.

<a id="pipeline.Pipeline.validate_pipeline"></a>

#### Pipeline.validate\_pipeline

```python
@staticmethod
def validate_pipeline(priority_queue: FIFOPriorityQueue) -> None
```

Validate the pipeline to check if it is blocked or has no valid entry point.

**Arguments**:

- `priority_queue`: Priority queue of component names.

**Raises**:

- `PipelineRuntimeError`: If the pipeline is blocked or has no valid entry point.

