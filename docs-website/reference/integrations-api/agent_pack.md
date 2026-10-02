---
title: "Agent Pack"
id: integrations-agent-pack
description: "Agent Pack integration for Haystack"
slug: "/integrations-agent-pack"
---


## haystack_integrations.agent_pack.advanced_rag.agent

### create_advanced_rag_agent

```python
create_advanced_rag_agent(
    *,
    document_store: DocumentStore,
    retriever: TextRetriever | Pipeline | None = None,
    retrieval_pipeline_input_mapping: dict[str, list[str]] | None = None,
    retrieval_pipeline_output_mapping: dict[str, str] | None = None,
    llm: ChatGenerator | None = None,
    backup_answer_llm: ChatGenerator | None = None,
    system_prompt: str | None = None,
    max_agent_steps: int = 20,
    max_fetched_docs: int = 10
) -> Agent
```

Create the advanced RAG agent.

The agent answers questions from documents it retrieves out of the document store. Instead of guessing which
metadata fields exist, it can inspect the store (fields, values, ranges) and construct a Haystack filter to
narrow its retrieval when metadata helps — plain, unfiltered retrieval remains available when it doesn't. The
answer cites the retrieved documents.

The required `retriever` becomes the `search_documents` tool; `document_store` additionally feeds the three
metadata inspection tools and must implement the metadata introspection methods (`get_metadata_fields_info`,
`get_metadata_field_unique_values`, `get_metadata_field_min_max`).

To customize the returned agent further (e.g. add tools or hooks), use `Agent.clone`:
`agent = agent.clone(tools=[*agent.tools, my_tool])`.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – The document store the metadata inspection tools and the `fetch_documents_by_filter` tool
  run against.
- **retriever** (<code>TextRetriever | Pipeline | None</code>) – What retrieves for the `search_documents` tool (required). Either a standalone retriever
  component following the `TextRetriever` protocol, i.e. its `run` method accepts `query` and `filters`
  (e.g. `InMemoryBM25Retriever`, or an embedding retriever wrapped in `TextEmbeddingRetriever`), or a custom
  retrieval `Pipeline` (e.g. embedder -> retriever, or hybrid retrieval) — a pipeline additionally requires
  `retrieval_pipeline_input_mapping`. It should retrieve by relevance scoring (keyword or embedding-based) —
  direct, unscored fetching is already covered by the built-in `fetch_documents_by_filter` tool.
- **retrieval_pipeline_input_mapping** (<code>dict\[str, list\[str\]\] | None</code>) – Required when `retriever` is a `Pipeline`: maps the tool inputs to
  pipeline input sockets; must have exactly the keys "query" and "filters",
  e.g. `{"query": ["embedder.text"], "filters": ["retriever.filters"]}`.
- **retrieval_pipeline_output_mapping** (<code>dict\[str, str\] | None</code>) – Optional when `retriever` is a `Pipeline`: maps pipeline output sockets
  to tool outputs, e.g. `{"retriever.documents": "documents"}`.
- **llm** (<code>ChatGenerator | None</code>) – LLM that drives the agent loop. Defaults to `OpenAIResponsesChatGenerator("gpt-5.4")` with low
  reasoning effort.
- **backup_answer_llm** (<code>ChatGenerator | None</code>) – LLM the built-in `BackupAnswerHook` uses to write a best-effort answer when the run
  is cut off by `max_agent_steps`. Defaults to a separate `OpenAIResponsesChatGenerator("gpt-5.4")` with low
  reasoning effort.
- **system_prompt** (<code>str | None</code>) – Overrides the pre-made system prompt.
- **max_agent_steps** (<code>int</code>) – Maximum steps for the agent loop. If the loop is cut off by this limit before writing an
  answer, an `after_run` hook (`BackupAnswerHook`) makes one extra LLM call to produce a best-effort answer from
  the evidence gathered so far, so `last_message` always carries a text answer.
- **max_fetched_docs** (<code>int</code>) – Maximum number of documents `fetch_documents_by_filter` shows per fetch. A filter fetch is
  not bounded by a retriever's `top_k`, so this caps the tool result instead; the scored `search_documents` tool
  is bounded by the `top_k` configured on your retrieval components.

**Returns:**

- <code>Agent</code> – The advanced RAG `Agent`. Call it with the question as a user message,
  `agent.run(messages=[ChatMessage.from_user(question)])`; the answer is in `last_message` (a `ChatMessage`) and
  `documents` carries every document the agent retrieved during the run (deduplicated by id, in first-retrieved
  order) — the answer cites them by the first 8 characters of their id, e.g. `[doc a1b2c3d4]`. The standard Agent
  outputs `messages`, `step_count`, `token_usage` and `tool_call_counts` are also returned.

## haystack_integrations.agent_pack.advanced_rag.hooks

### BackupAnswerHook

Produce a final answer when the agent run ends without one. Runs as an `after_run` hook.

When the agent exhausts `max_agent_steps` mid-investigation, the run ends on a tool call or tool result instead of
an assistant text answer (and only `after_run` hooks run in this situation). This hook detects that case and makes
one LLM call over the conversation so far to produce a best-effort answer from the already-gathered evidence.

#### __init__

```python
__init__(chat_generator: ChatGenerator) -> None
```

Create the hook.

**Parameters:**

- **chat_generator** (<code>ChatGenerator</code>) – LLM that writes the backup answer from the gathered evidence.

#### warm_up

```python
warm_up() -> None
```

Prepare the hook's generator for use; called from the Agent's `warm_up`.

#### close

```python
close() -> None
```

Release the hook's generator resources; called from the Agent's `close`.

#### to_dict

```python
to_dict() -> dict
```

Serialize the hook to a dictionary.

**Returns:**

- <code>dict</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict) -> BackupAnswerHook
```

Deserialize the hook from a dictionary.

**Parameters:**

- **data** (<code>dict</code>) – Dictionary to deserialize from.

**Returns:**

- <code>BackupAnswerHook</code> – Deserialized hook.

#### run

```python
run(state: State) -> None
```

Append a best-effort final answer when the run ended without one (e.g. step exhaustion).

**Parameters:**

- **state** (<code>State</code>) – The agent run's state.

## haystack_integrations.agent_pack.advanced_rag.tools

### ListMetadataFieldsTool

Bases: <code>Tool</code>

Tool that lists all metadata fields and their types from a document store.

#### __init__

```python
__init__(document_store: DocumentStore) -> None
```

Create the tool.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – The document store to inspect. Must implement `get_metadata_fields_info`.

**Raises:**

- <code>ValueError</code> – If the store does not implement `get_metadata_fields_info`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the tool to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ListMetadataFieldsTool
```

Deserialize the tool from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary produced by `to_dict`.

**Returns:**

- <code>ListMetadataFieldsTool</code> – The deserialized tool.

### GetMetadataFieldValuesTool

Bases: <code>Tool</code>

Tool that returns the distinct values of a metadata field from a document store.

#### __init__

```python
__init__(document_store: DocumentStore) -> None
```

Create the tool.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – The document store to inspect. Must implement `get_metadata_field_unique_values`.

**Raises:**

- <code>ValueError</code> – If the store does not implement `get_metadata_field_unique_values`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the tool to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> GetMetadataFieldValuesTool
```

Deserialize the tool from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary produced by `to_dict`.

**Returns:**

- <code>GetMetadataFieldValuesTool</code> – The deserialized tool.

### GetMetadataFieldRangeTool

Bases: <code>Tool</code>

Tool that returns the minimum and maximum values of a metadata field from a document store.

#### __init__

```python
__init__(document_store: DocumentStore) -> None
```

Create the tool.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – The document store to inspect. Must implement `get_metadata_field_min_max`.

**Raises:**

- <code>ValueError</code> – If the store does not implement `get_metadata_field_min_max`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the tool to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> GetMetadataFieldRangeTool
```

Deserialize the tool from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary produced by `to_dict`.

**Returns:**

- <code>GetMetadataFieldRangeTool</code> – The deserialized tool.

### FetchDocumentsByFilterTool

Bases: <code>Tool</code>

Tool that fetches documents directly from a document store by metadata filter.

Unlike a scored retrieval tool, this fetches without any relevance ranking, so an agent can grab specific documents
(e.g. a known title or source file) without going through a relevance search. The fetched documents are put into
reading order first: grouped by their parent file (`file_name`/`file_path`/`source_id`) and sorted by their
position within it (`split_id`/`split_idx_start`/`page_number`), using whichever of those metadata fields the
documents carry. Match sets larger than `max_docs` are paged: each call returns one page plus the total match
count, and the tool's `offset` input continues where the previous page ended.

#### __init__

```python
__init__(
    document_store: DocumentStore,
    max_docs: int = 10,
    max_fetch_factor: int = 10,
) -> None
```

Create the tool.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – The document store to fetch documents from.
- **max_docs** (<code>int</code>) – Ceiling on the number of documents shown to the agent per fetch. Unlike scored retrieval, a
  filter fetch is not bounded by a retriever's `top_k`, so this caps the tool result instead. The LLM can
  request fewer via the tool's optional `max_docs` input, but never more.
- **max_fetch_factor** (<code>int</code>) – How many times the `max_docs` ceiling a filter may match before the fetch is
  refused outright (when the store supports `count_documents_by_filter`) — the refusal is surfaced to the
  LLM as an error it can recover from by narrowing the filter.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the tool to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> FetchDocumentsByFilterTool
```

Deserialize the tool from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary produced by `to_dict`.

**Returns:**

- <code>FetchDocumentsByFilterTool</code> – The deserialized tool.

### DocumentStoreToolset

Bases: <code>Toolset</code>

All document-store-backed tools as one unit.

Bundles the three metadata inspection tools (`ListMetadataFieldsTool`, `GetMetadataFieldValuesTool`,
`GetMetadataFieldRangeTool`) and the direct `FetchDocumentsByFilterTool`, so they can be handed to an `Agent`
(or combined with a retrieval tool) as a single object.

#### __init__

```python
__init__(document_store: DocumentStore, max_fetched_docs: int = 10) -> None
```

Create the toolset.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – The document store all tools run against. Must implement the metadata introspection
  methods (`get_metadata_fields_info`, `get_metadata_field_unique_values`, `get_metadata_field_min_max`).
- **max_fetched_docs** (<code>int</code>) – Maximum number of documents `fetch_documents_by_filter` shows per fetch (see
  `FetchDocumentsByFilterTool.max_docs`).

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the toolset to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> DocumentStoreToolset
```

Deserialize the toolset from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary produced by `to_dict`.

**Returns:**

- <code>DocumentStoreToolset</code> – The deserialized toolset.

## haystack_integrations.agent_pack.deep_research.agent

### create_deep_research_agent

```python
create_deep_research_agent(
    *,
    llm: ChatGenerator | None = None,
    system_prompt: str | None = None,
    max_agent_steps: int = 8,
    brief_llm: ChatGenerator | None = None,
    researcher_llm: ChatGenerator | None = None,
    search_tool: Tool | None = None,
    page_summary_llm: ChatGenerator | None = None,
    report_llm: ChatGenerator | None = None,
    max_researcher_steps: int = 20,
    max_concurrent_researchers: int = 5,
    max_subtopics: int = 5,
    max_page_chars: int = 50000
) -> Agent
```

Create the deep research agent.

To customize the returned agent further (e.g. add tools or hooks), use `Agent.clone`:
`agent = agent.clone(tools=[*agent.tools, my_tool])`.

**Parameters:**

- **llm** (<code>ChatGenerator | None</code>) – LLM that plans the investigation and delegates the sub-questions.
  Defaults to `OpenAIResponsesChatGenerator("gpt-5.4")`.
- **system_prompt** (<code>str | None</code>) – Overrides the pre-made system prompt. `{{ max_subtopics }}` is replaced with the value of
  `max_subtopics`.
- **max_agent_steps** (<code>int</code>) – Maximum steps for the agent loop (reflect -> delegate rounds).
- **brief_llm** (<code>ChatGenerator | None</code>) – LLM that rewrites the user query into a focused research brief.
  Defaults to `OpenAIResponsesChatGenerator("gpt-5.4")`.
- **researcher_llm** (<code>ChatGenerator | None</code>) – LLM that drives each sub-researcher's search/read/think loop.
  Defaults to `OpenAIResponsesChatGenerator("gpt-5.4-mini")`.
- **search_tool** (<code>Tool | None</code>) – Web search tool used by each sub-researcher. Defaults to `TavilyWebSearchTool(top_k=10)`,
  which requires `tavily-haystack`. The pre-made researcher prompt refers to the tool as `web_search`.
- **page_summary_llm** (<code>ChatGenerator | None</code>) – LLM used inside the `read_url` tool to summarize a fetched page toward
  the question. Defaults to `OpenAIResponsesChatGenerator("gpt-5.4-mini")`.
- **report_llm** (<code>ChatGenerator | None</code>) – LLM that turns the brief plus collected notes into the final report.
  Defaults to `OpenAIResponsesChatGenerator("gpt-5.4")`.
- **max_researcher_steps** (<code>int</code>) – Maximum steps for each sub-researcher's agent loop.
- **max_concurrent_researchers** (<code>int</code>) – Maximum number of sub-researchers that run at the same time.
- **max_subtopics** (<code>int</code>) – Maximum number of sub-questions the agent may delegate (breadth).
- **max_page_chars** (<code>int</code>) – Maximum raw page characters fed to the summarizer, before summarization.

**Returns:**

- <code>Agent</code> – The deep research `Agent`. Call it with the question as a user message,
  `agent.run(messages=[ChatMessage.from_user(question)])`; it returns a dict whose main output is
  `report` (the final markdown report, a `str`). The dict also carries the intermediate `brief`
  (`str`) and `notes` (`list[str]`), plus the standard Agent outputs `messages`, `last_message`,
  `step_count`, `token_usage` and `tool_call_counts`.
