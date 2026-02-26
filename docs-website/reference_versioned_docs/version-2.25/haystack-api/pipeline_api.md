---
title: "Pipeline"
id: pipeline-api
description: "Arranges components and integrations in flow."
slug: "/pipeline-api"
---


## async_pipeline

### AsyncPipeline

Bases: <code>PipelineBase</code>

Asynchronous version of the Pipeline orchestration engine.

Manages components in a pipeline allowing for concurrent processing when the pipeline's execution graph permits.
This enables efficient processing of components by minimizing idle time and maximizing resource utilization.

#### run_async_generator

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

#### run

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

## pipeline

### Pipeline

Bases: <code>PipelineBase</code>

Synchronous version of the orchestration engine.

Orchestrates component execution according to the execution graph, one after the other.

#### run

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
