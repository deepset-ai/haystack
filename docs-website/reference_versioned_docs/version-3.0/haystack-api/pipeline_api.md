---
title: "Pipeline"
id: pipeline-api
description: "Arranges components and integrations in flow."
slug: "/pipeline-api"
---


## pipeline

### PipelineStreamHandle

Handle returned by `Pipeline.stream()`.

Async-iterable over `StreamingChunk`s produced by streaming components in the pipeline. After iteration ends,
`result` holds the final pipeline output dict.

By default, iteration cleans up automatically: if the consumer abandons iteration, the underlying pipeline task is
cancelled. `aclose()` is also available for explicit cleanup.

#### result

```python
result: dict[str, Any]
```

Final pipeline output dict, available only after a successful, complete run.

Raises a `RuntimeError` if the pipeline has not finished or was cancelled. If the pipeline failed, re-raises the
original exception.

#### aclose

```python
aclose() -> None
```

Cancel the underlying pipeline task.

Bounded by `_CLEANUP_TIMEOUT_SECONDS` so that components cannot block cleanup indefinitely.

### Pipeline

Bases: <code>PipelineBase</code>

Orchestration engine that runs components according to the execution graph.

Supports both a synchronous run path (`run`) and an asynchronous run path
(`run_async`, `run_async_generator`, `stream`).

#### run

```python
run(
    data: dict[str, Any],
    include_outputs_from: set[str] | None = None,
    *,
    break_point: Breakpoint | None = None,
    pipeline_snapshot: PipelineSnapshot | None = None,
    snapshot_callback: SnapshotCallback | None = None
) -> dict[str, Any]
```

Runs the Pipeline with given input data.

`run` executes synchronously and blocks the calling thread until the run completes. In an async context,
use `run_async` instead.

Usage:

```python
from haystack import Pipeline, Document
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret

# Write documents to InMemoryDocumentStore
document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome.")
])

retriever = InMemoryBM25Retriever(document_store=document_store)

prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
"""

template = [ChatMessage.from_user(prompt_template)]
prompt_builder = ChatPromptBuilder(
    template=template,
    required_variables=["question", "documents"],
    variables=["question", "documents"]
)

llm = OpenAIChatGenerator()
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

question = "Who lives in Paris?"
results = rag_pipeline.run(
    {
        "retriever": {"query": question},
        "prompt_builder": {"question": question},
    }
)

print(results["llm"]["replies"][0].text)
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
- **break_point** (<code>Breakpoint | None</code>) – A breakpoint that pauses execution before the specified component runs by raising a
  `BreakpointException` carrying a `PipelineSnapshot` of the current pipeline state.
- **pipeline_snapshot** (<code>PipelineSnapshot | None</code>) – A snapshot of a previously interrupted pipeline execution to resume from. Can be combined with
  `break_point` to step through a pipeline: resume from the snapshot and pause again at the next
  breakpoint. The `break_point` must target a different component or visit count than the one the
  snapshot was created at, otherwise it would trigger again before any progress is made.
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

#### run_async_generator

```python
run_async_generator(
    data: dict[str, Any],
    include_outputs_from: set[str] | None = None,
    concurrency_limit: int = 4,
) -> AsyncGenerator[dict[str, Any], None]
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
from haystack import Pipeline
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

rag_pipeline = Pipeline()
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

- <code>AsyncGenerator\[dict\[str, Any\], None\]</code> – An async iterator containing partial (and final) outputs.

**Raises:**

- <code>ValueError</code> – If invalid inputs are provided to the pipeline, or if `concurrency_limit` is less than 1.
- <code>PipelineMaxComponentRuns</code> – If a component exceeds the maximum number of allowed executions within the pipeline.
- <code>PipelineRuntimeError</code> – If the Pipeline contains cycles with unsupported connections that would cause
  it to get stuck and fail running.
  Or if a Component fails or returns output in an unsupported type.

#### run_async

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
from haystack import Pipeline
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

rag_pipeline = Pipeline()
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

- <code>ValueError</code> – If invalid inputs are provided to the pipeline, or if `concurrency_limit` is less than 1.
- <code>PipelineRuntimeError</code> – If the Pipeline contains cycles with unsupported connections that would cause
  it to get stuck and fail running.
  Or if a Component fails or returns output in an unsupported type.
- <code>PipelineMaxComponentRuns</code> – If a Component reaches the maximum number of times it can be run in this Pipeline.

#### stream

```python
stream(
    data: dict[str, Any],
    *,
    streaming_components: list[str] | None = None,
    include_outputs_from: set[str] | None = None,
    concurrency_limit: int = 4,
    cancel_on_abandon: bool = True
) -> PipelineStreamHandle
```

Run the pipeline and return a handle that streams `StreamingChunk`s as they arrive.

Iterate the handle with `async for` to consume chunks; after iteration ends, `handle.result` holds the final
pipeline output dict (same as `run_async`). By default, if iteration is abandoned, the underlying pipeline task
is cancelled automatically. Pass `cancel_on_abandon=False` to instead let the pipeline run to completion.

For every async-capable component that exposes a `streaming_callback` input socket, a forwarder is injected at
runtime that pushes chunks onto the handle's queue. If a `streaming_callback` is provided at component init or
at runtime (inside `data`, e.g. `data={"llm": {"streaming_callback": cb}}`), it is also invoked for each chunk.
Async callbacks are preferred; a sync callback is accepted but will run synchronously on the event loop and
may block it.

Usage:

```python
import asyncio

from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack import Pipeline
from haystack.dataclasses import ChatMessage

pipe = Pipeline()
pipe.add_component(
    "prompt_builder",
    ChatPromptBuilder(template=[ChatMessage.from_user("Tell me about {{topic}}")]),
)
pipe.add_component("llm", OpenAIChatGenerator())
pipe.connect("prompt_builder.prompt", "llm.messages")

async def main():
    handle = pipe.stream(data={"prompt_builder": {"topic": "Italy"}})
    async for chunk in handle:
        print(chunk.content, end="", flush=True)
    return handle.result

result = asyncio.run(main())
print(result["llm"]["replies"])
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

- **streaming_components** (<code>list\[str\] | None</code>) – Names of components to stream from. If `None` (default), every streaming-capable
  component is forwarded. If a list, only the listed components are forwarded; unknown names or names of
  components that do not support streaming raise `ValueError`.
- **include_outputs_from** (<code>set\[str\] | None</code>) – Set of component names whose individual outputs are to be
  included in the pipeline's output. For components that are
  invoked multiple times (in a loop), only the last-produced
  output is included.
- **concurrency_limit** (<code>int</code>) – The maximum number of components that should be allowed to run concurrently.
- **cancel_on_abandon** (<code>bool</code>) – If `True` (default), the underlying pipeline task is cancelled when iteration is
  abandoned. If `False`, the pipeline runs to completion even when the consumer stops reading.

**Returns:**

- <code>PipelineStreamHandle</code> – A `PipelineStreamHandle` that is async-iterable over `StreamingChunk`s. After iteration ends,
  `handle.result` holds the final pipeline output dict (same shape as `run_async`).

**Raises:**

- <code>ValueError</code> – If `streaming_components` contains unknown component names or components that do not support streaming,
  or if invalid inputs are provided to the pipeline, or if `concurrency_limit` is less than 1.
- <code>PipelineRuntimeError</code> – Surfaced during iteration. If the Pipeline contains cycles with unsupported connections that would cause
  it to get stuck and fail running, or if a Component fails or returns output in an unsupported type.
- <code>PipelineMaxComponentRuns</code> – Surfaced during iteration. If a Component reaches the maximum number of times it can be run in this
  Pipeline.
