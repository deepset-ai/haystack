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
__init__(
    document_store: DocumentStore,
    *,
    name: str = "list_metadata_fields",
    description: str = prompts.LIST_METADATA_FIELDS_TOOL_DESCRIPTION
) -> None
```

Create the tool.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – The document store to inspect. Must implement `get_metadata_fields_info`.
- **name** (<code>str</code>) – Overrides the tool name the LLM selects by.
- **description** (<code>str</code>) – Overrides the tool description the LLM reads when choosing between tools.

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
__init__(
    document_store: DocumentStore,
    *,
    name: str = "get_metadata_field_values",
    description: str = prompts.GET_METADATA_FIELD_VALUES_TOOL_DESCRIPTION
) -> None
```

Create the tool.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – The document store to inspect. Must implement `get_metadata_field_unique_values`.
- **name** (<code>str</code>) – Overrides the tool name the LLM selects by.
- **description** (<code>str</code>) – Overrides the tool description the LLM reads when choosing between tools.

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
__init__(
    document_store: DocumentStore,
    *,
    name: str = "get_metadata_field_range",
    description: str = prompts.GET_METADATA_FIELD_RANGE_TOOL_DESCRIPTION
) -> None
```

Create the tool.

**Parameters:**

- **document_store** (<code>DocumentStore</code>) – The document store to inspect. Must implement `get_metadata_field_min_max`.
- **name** (<code>str</code>) – Overrides the tool name the LLM selects by.
- **description** (<code>str</code>) – Overrides the tool description the LLM reads when choosing between tools.

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
    *,
    name: str = "fetch_documents_by_filter",
    description: str = prompts.FILTER_RETRIEVER_TOOL_DESCRIPTION
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
- **name** (<code>str</code>) – Overrides the tool name the LLM selects by.
- **description** (<code>str</code>) – Overrides the tool description the LLM reads when choosing between tools.

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

## haystack_integrations.agent_pack.evaluation.dataclasses

### RetrievalEvalCase

One labelled question and the documents needed to answer it.

**Parameters:**

- **question** (<code>str</code>) – The question to put to whatever is under evaluation.
- **evidence** (<code>dict\[str, str\]</code>) – Ground truth, as `{document id: the quote found in that document}`. The keys are the
  documents recall is measured against, and the values say what each one was needed for, for example
  `{"a1b2c3...": "Tyreek Hill now needs to ...", "d4e5f6...": "The Dolphins went on to ..."}`.
  A harness that knows which documents are needed but not what they were needed for leaves the values empty.

#### expected_document_ids

```python
expected_document_ids: frozenset[str]
```

The documents needed to answer a question based on the provided evidence

#### found_at

```python
found_at(document_ids: list[str], k: int | None = None) -> frozenset[str]
```

Return which of the needed documents a run returned within its first `k`.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – What the run returned, in the order it ranked them.
- **k** (<code>int | None</code>) – Rank cutoff, or `None` to count everything returned.

**Returns:**

- <code>frozenset\[str\]</code> – The needed documents that were found.

#### recall_at

```python
recall_at(document_ids: list[str], k: int | None = None) -> float
```

Return the share of the needed documents a run found within its first `k`.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – What the run returned, in the order it ranked them.
- **k** (<code>int | None</code>) – Rank cutoff, or `None` to count everything returned.

**Returns:**

- <code>float</code> – Recall@k.

#### precision_at

```python
precision_at(document_ids: list[str], k: int | None = None) -> float
```

Return the share of a run's first `k` documents that were needed.

**Parameters:**

- **document_ids** (<code>list\[str\]</code>) – What the run returned, in the order it ranked them.
- **k** (<code>int | None</code>) – Rank cutoff, or `None` to count everything returned.

**Returns:**

- <code>float</code> – Precision@k, or 0.0 when the run returned nothing.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Convert the eval case into a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with one key per field, JSON compatible and with the evidence in a stable order,
  so an evaluation set is identified the same way whichever order it was built in.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> RetrievalEvalCase
```

Create a new eval case from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to build the eval case from.

**Returns:**

- <code>RetrievalEvalCase</code> – The created object.

### ModelTokenUsage

Raw token usage attributable to one model deployment.

**Parameters:**

- **input_tokens** (<code>int</code>) – Number of input tokens consumed by the model.
- **output_tokens** (<code>int</code>) – Number of output tokens generated by the model.

### ModelPrice

Informational token prices for one model deployment.

**Parameters:**

- **input_cost_per_million** (<code>float</code>) – Price of one million input tokens.
- **output_cost_per_million** (<code>float</code>) – Price of one million output tokens.

### cost_of_model_usage

```python
cost_of_model_usage(
    model_usage: dict[str, ModelTokenUsage], prices: dict[str, ModelPrice]
) -> float | None
```

Calculate what raw token usage costs at known prices.

Every input token is charged at full price. A provider that discounts tokens served from its prompt cache
charges less than this, so the result is an upper bound wherever caching is in play.

**Parameters:**

- **model_usage** (<code>dict\[str, ModelTokenUsage\]</code>) – Raw token usage keyed by model identifier.
- **prices** (<code>dict\[str, ModelPrice\]</code>) – Token prices keyed by model identifier.

**Returns:**

- <code>float | None</code> – The total cost, or `None` when any model in the usage has no known price.

### EvalMetrics

Metrics produced by a harness evaluator for one target configuration.

**Parameters:**

- **durations** (<code>list\[float\]</code>) – How long each run of the target took, in seconds, one per eval case in eval case order.
- **all_tokens_reported** (<code>bool</code>) – Whether every LLM call the evaluation made reported its token counts. False means
  `model_usage` understates what the evaluation actually spent.
- **model_usage** (<code>dict\[str, ModelTokenUsage\]</code>) – Raw token usage keyed by model identifier.
- **eval_cases** (<code>list\[dict\[str, Any\]\]</code>) – One record per eval case measured, each in whatever shape its evaluator scores. Empty for
  an evaluator that reports only an aggregate.
- **details** (<code>dict\[str, Any\]</code>) – What the harness evaluator scored and the diagnostics it collected. Each harness evaluator
  documents its own keys.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Return a JSON-compatible representation.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> EvalMetrics
```

Restore metrics from a serialized representation.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Serialized eval metrics.

**Returns:**

- <code>EvalMetrics</code> – The restored eval metrics.

## haystack_integrations.agent_pack.evaluation.harness_evaluator

### HarnessEvaluator

Bases: <code>Protocol</code>

Measure a target configuration over labelled eval cases.

#### evaluate

```python
evaluate(target: Any, eval_cases: list[Any]) -> EvalMetrics
```

Measure a target configuration over the supplied eval cases.

**Parameters:**

- **target** (<code>Any</code>) – Target configuration to evaluate.
- **eval_cases** (<code>list\[Any\]</code>) – The labelled expectations to score it against.

**Returns:**

- <code>EvalMetrics</code> – Returns EvalMetrics, which includes the score and other relevant metrics.

#### evaluate_async

```python
evaluate_async(target: Any, eval_cases: list[Any]) -> EvalMetrics
```

Measure a materialized configuration over the supplied eval cases.

**Parameters:**

- **target** (<code>Any</code>) – Materialized configuration to evaluate, of whatever kind this evaluator measures.
- **eval_cases** (<code>list\[Any\]</code>) – The labelled expectations to score it against.

**Returns:**

- <code>EvalMetrics</code> – Returns EvalMetrics, which includes the score and other relevant metrics.

#### validate

```python
validate(target: Any) -> None
```

Validate a target configuration to catch errors before harness runs another evaluation.

**Parameters:**

- **target** (<code>Any</code>) – Candidate configuration deserialized from YAML.

**Raises:**

- <code>ValueError</code> – If the candidate is missing something the evaluator requires.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the evaluator.

**Returns:**

- <code>dict\[str, Any\]</code> – The evaluator's `type` and the settings `from_dict` restores it from.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> HarnessEvaluator
```

Restore an evaluator from what `to_dict` returned.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The serialized evaluator.

**Returns:**

- <code>HarnessEvaluator</code> – The restored evaluator.

## haystack_integrations.agent_pack.evaluation.retrieval_harness_evaluator

### RetrievalEvalCaseMetrics

Score for one retrieval eval case.

**Parameters:**

- **question** (<code>str</code>) – The question that was posed.
- **passed** (<code>bool</code>) – Whether the run met every expectation, which is true exactly when `failures` is empty.
- **component_output_sizes** (<code>dict\[str, dict\[str, int\]\]</code>) – How many items each component emitted, by component name and output socket.
- **component_output_samples** (<code>dict\[str, dict\[str, list\[str\]\]\]</code>) – A capped sample of whatever each component emitted as text, by component name and
  output socket. For a pipeline that rewrites the question, this is what it actually asked the store,
  which is usually what explains a recall failure rather than how many queries there were.
- **failures** (<code>tuple\[str, ...\]</code>) – Every expectation the run missed, named.
- **recall_at_k** (<code>float</code>) – The share of the expected documents found within the first `k` returned.
- **precision_at_k** (<code>float</code>) – The share of the first `k` returned that were expected.
- **retrieved** (<code>int</code>) – Everything the run returned, which is separate from how deep it was scored: returning more
  than `k` is not a fault, it simply earns nothing for the documents past the cutoff.
- **missed_document_ids** (<code>tuple\[str, ...\]</code>) – The expected documents the run did not return, sorted.
- **duration** (<code>float</code>) – How long the run took, in seconds.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Return a JSON-compatible representation.

### RetrievalHarnessEvaluator

Pose every eval case's question to a retrieval pipeline and score what came back.

#### __init__

```python
__init__(
    *,
    k: int | None = None,
    min_recall: float = 1.0,
    min_precision: float = 0.0,
    max_concurrent_eval_cases: int = 1
) -> None
```

Create an evaluator.

**Parameters:**

- **k** (<code>int | None</code>) – Rank cutoff every eval case is scored at, giving recall@k and precision@k. Only the first
  `k` documents a run returns count, in the order it ranked them, so a pipeline is measured on what
  it put at the top rather than on how much it returned. `None` scores everything returned.
- **min_recall** (<code>float</code>) – Share of an eval case's needed documents a run must find for that eval case to pass.
  The mean recall is reported either way; this only decides which eval cases are reported as failures.
- **min_precision** (<code>float</code>) – Share of what a run returned that must be needed for its eval case to pass. Left at 0
  by default, because returning more than was asked for is not itself a fault.
- **max_concurrent_eval_cases** (<code>int</code>) – How many eval cases to measure at once. Eval cases are independent and
  each spends its time waiting on a model, so this decides wall-clock time rather than cost. Leave it at
  1 when comparing configurations on `durations`, since concurrent runs contend for the same rate
  limits and each run's duration then reflects that contention rather than the configuration.

**Raises:**

- <code>ValueError</code> – If `max_concurrent_eval_cases` is below one.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the evaluator.

**Returns:**

- <code>dict\[str, Any\]</code> – The evaluator's `type` and its settings.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> RetrievalHarnessEvaluator
```

Restore an evaluator from what `to_dict` returned.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The serialized evaluator.

**Returns:**

- <code>RetrievalHarnessEvaluator</code> – The restored evaluator.

#### validate

```python
validate(target: Pipeline) -> None
```

Validate the candidate pipeline to expose at least one `query` input and exactly one `documents` output.

Every other mandatory input must be connected, since the harness feeds the pipeline nothing but the question.

**Parameters:**

- **target** (<code>Pipeline</code>) – Candidate pipeline deserialized from YAML.

**Raises:**

- <code>ValueError</code> – If the pipeline exposes no `query` input, leaves another mandatory input unconnected,
  or has not exactly one `documents` output.

#### evaluate

```python
evaluate(target: Pipeline, eval_cases: list[RetrievalEvalCase]) -> EvalMetrics
```

Pose every eval case to the pipeline and return raw experiment metrics, from synchronous code.

**Parameters:**

- **target** (<code>Pipeline</code>) – The materialized candidate pipeline to score.
- **eval_cases** (<code>list\[RetrievalEvalCase\]</code>) – The labelled expectations to score it against.

**Returns:**

- <code>EvalMetrics</code> – What `evaluate_async` measured.

**Raises:**

- <code>RuntimeError</code> – If an event loop is already running; await `evaluate_async` from inside one.

#### evaluate_async

```python
evaluate_async(
    target: Pipeline, eval_cases: list[RetrievalEvalCase]
) -> EvalMetrics
```

Pose every eval case to the pipeline and return raw experiment metrics.

**Parameters:**

- **target** (<code>Pipeline</code>) – The materialized candidate pipeline to score.
- **eval_cases** (<code>list\[RetrievalEvalCase\]</code>) – The labelled expectations to score it against.

**Returns:**

- <code>EvalMetrics</code> – Per-eval-case durations and raw model usage, with `details` holding `mean_recall_at_k`,
  `mean_precision_at_k`, `mean_retrieved`, `component_output_sizes` and `warnings`, and one record per
  eval case in `eval_cases`.

**Raises:**

- <code>ValueError</code> – If no eval cases were supplied, leaving nothing to score.

## haystack_integrations.agent_pack.optimization.agent

### create_harness_optimizer_agent

```python
create_harness_optimizer_agent(
    *,
    evaluator: HarnessEvaluator,
    loader: Callable[[str], Agent | Pipeline] = load_agent,
    llm: ChatGenerator | None = None,
    system_prompt: str | None = None,
    max_agent_steps: int = 24,
    additional_instructions: str | None = None,
    documentation_tools: bool = False
) -> Agent
```

Create the harness optimizer agent.

The agent reads a configuration's YAML together with its evaluation results and edits it into the next
candidate to measure.

**Parameters:**

- **evaluator** (<code>HarnessEvaluator</code>) – The evaluator the candidates are measured with. `validate_config` runs its `validate` on every
  draft, and it has to implement `to_dict` and `from_dict` so the agent can be serialized.
- **loader** (<code>Callable\\[[str\], Agent | Pipeline\]</code>) – Builds a configuration from YAML, and decides what shape a candidate must keep. Defaults to the
  Agent contract; pass `load_pipeline` to optimize a Pipeline that is not a single Agent.
- **llm** (<code>ChatGenerator | None</code>) – LLM that reasons about the evaluation results and edits the configuration. Defaults to
  `OpenAIResponsesChatGenerator("gpt-5.6-terra")` with low reasoning effort.
- **system_prompt** (<code>str | None</code>) – Overrides the pre-made system prompt. A description of the installed Haystack version
  and integrations is appended either way.
- **max_agent_steps** (<code>int</code>) – Maximum steps for the agent loop while producing one candidate.
- **additional_instructions** (<code>str | None</code>) – Guidance appended to the system prompt about what a good configuration looks
  like for one specific harness.
- **documentation_tools** (<code>bool</code>) – Give the agent read-only search over the public Haystack documentation, so it can
  look up a component's parameters and serialized shape. Requires `mcp-haystack>=1.2.0`, and reaches the
  documentation server over the network.

**Returns:**

- <code>Agent</code> – The harness optimizer `Agent`. Its editing tools work on the `draft` and `known` state passed to `run`,
  and a run ends when the optimizer calls `submit_candidate` or `finish`. `propose_candidate` runs one turn
  with that state and the prompt built from the experiment so far.

### propose_candidate

```python
propose_candidate(
    *,
    optimizer_agent: Agent,
    reference: Agent | Pipeline,
    reference_yaml: str,
    prices: dict[str, ModelPrice],
    objectives: OptimizationObjectives,
    baseline: EvalMetrics,
    history: list[CandidateOutcome],
    history_digest_window: int = 1,
    remaining_evaluations: int | None = None,
    candidates: dict[str, str] | None = None,
    base_id: str | None = None
) -> ProposalResult
```

Let the optimizer edit, validate and submit one YAML candidate.

**Parameters:**

- **optimizer_agent** (<code>Agent</code>) – The agent from `create_harness_optimizer_agent`.
- **reference** (<code>Agent | Pipeline</code>) – Reference configuration supplying tool specifications, when it has any.
- **reference_yaml** (<code>str</code>) – The reference serialized as YAML, which the optimizer can restore as `"reference"`.
- **prices** (<code>dict\[str, ModelPrice\]</code>) – Known token prices keyed by model identifier.
- **objectives** (<code>OptimizationObjectives</code>) – Quality gates and ranking objective.
- **baseline** (<code>EvalMetrics</code>) – Reference measurement.
- **history** (<code>list\[CandidateOutcome\]</code>) – Prior candidate outcomes, oldest first.
- **history_digest_window** (<code>int</code>) – How many of the most recent outcomes are shown in full.
- **remaining_evaluations** (<code>int | None</code>) – How many candidates, including this one, the experiment can still measure. Shown
  to the optimizer as its budget, or as unknown when None.
- **candidates** (<code>dict\[str, str\] | None</code>) – The YAML of every candidate submitted on earlier turns, keyed by candidate ID. The optimizer
  can restore any of them, and submitting one of them again is refused.
- **base_id** (<code>str | None</code>) – The candidate this turn's edits start from, or `None` to start from the reference.

**Returns:**

- <code>ProposalResult</code> – The submitted candidate, or `None` in its place when the optimizer finished or ran out of steps, along
  with why it finished and the drafts that failed validation.

**Raises:**

- <code>ValueError</code> – If `base_id` is not one of `candidates`.

## haystack_integrations.agent_pack.optimization.dataclasses

### OptimizationObjectives

Which measurement the experiment treats as quality, the gates a candidate must clear, and what it is ranked on.

**Parameters:**

- **quality_metric** (<code>str</code>) – The key in `EvalMetrics.details` treated as quality, where higher is better. Which keys
  exist depends on the harness evaluator measuring the candidates.
- **max_quality_loss** (<code>float</code>) – How far below the reference's quality metric a candidate may fall and still clear the
  gates.
- **primary** (<code>Literal['cost', 'duration', 'quality']</code>) – What candidates are ranked by. "cost" and "duration" are minimized among candidates that clear
  the quality gates, where "duration" is the median of `EvalMetrics.durations`. "quality" is maximized
  directly, with cost breaking ties.

#### get_quality

```python
get_quality(metrics: EvalMetrics) -> float
```

Read the quality metric from one measurement.

**Parameters:**

- **metrics** (<code>EvalMetrics</code>) – A harness evaluator's measurement.

**Returns:**

- <code>float</code> – The value the harness reported under `quality_metric`.

**Raises:**

- <code>ValueError</code> – If the harness reported no such key.

#### find_failed_gates

```python
find_failed_gates(
    metrics: EvalMetrics, baseline: EvalMetrics
) -> tuple[str, ...]
```

Check a candidate's measurement against the hard gates.

**Parameters:**

- **metrics** (<code>EvalMetrics</code>) – The candidate's measurement.
- **baseline** (<code>EvalMetrics</code>) – The reference's measurement.

**Returns:**

- <code>tuple\[str, ...\]</code> – The gates the candidate failed, for example `("quality_below_floor:0.8000",)`. An empty tuple is
  returned when it cleared them all.

#### sort_key

```python
sort_key(metrics: EvalMetrics, cost: float | None) -> tuple[float, float]
```

Build the sort key for one measurement, lower being better.

**Parameters:**

- **metrics** (<code>EvalMetrics</code>) – A measurement.
- **cost** (<code>float | None</code>) – Its cost, or `None` when it could not be priced.

**Returns:**

- <code>tuple\[float, float\]</code> – `(cost, median duration)` for "cost", `(median duration, cost)` for "duration", and
  `(-quality, cost)` for "quality". A missing cost or duration sorts as infinity.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Return a JSON-compatible representation.

### CandidateConfiguration

One configuration the optimizer submitted with `submit_candidate`.

**Parameters:**

- **candidate_id** (<code>str</code>) – Digest of the parsed YAML, which ignores formatting.
- **parent_id** (<code>str</code>) – Identifier of the snapshot the edits started from, the reference's ID when that is the
  reference.
- **yaml** (<code>str</code>) – The submitted configuration as Pipeline YAML.
- **rationale** (<code>str</code>) – The hypothesis the optimizer gave when submitting it.
- **diff** (<code>str</code>) – Unified diff of the parent snapshot against the submitted YAML.

### CandidateOutcome

One measured candidate, as the optimizer is shown it on later turns.

**Parameters:**

- **candidate_id** (<code>str</code>) – Identifier of the candidate configuration.
- **parent_id** (<code>str | None</code>) – Identifier of the configuration it was edited from, or None when that is the reference.
- **rationale** (<code>str</code>) – Why the optimizer proposed it.
- **diff** (<code>str</code>) – Its change against the parent, as a unified diff.
- **metrics** (<code>EvalMetrics | None</code>) – Its measurement, or None when it could not be measured.
- **cost** (<code>float | None</code>) – Its cost, or None when it could not be priced.
- **failure** (<code>str | None</code>) – Why it could not be measured.
- **gate_failures** (<code>tuple\[str, ...\]</code>) – The gates it failed, for example `("usage_incomplete",)`. Empty when it cleared them all.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Return a JSON-compatible representation.

### ConfigurationDraft

The YAML the optimizer is editing during one proposal turn.

**Parameters:**

- **yaml** (<code>str</code>) – The current YAML.
- **parent_id** (<code>str</code>) – The configuration the edits started from.
- **validated_revision** (<code>str | None</code>) – The revision that last passed `validate_config`, or `None` when the current YAML has
  not.

#### revision

```python
revision: str
```

The current YAML's revision, which every edit has to name.

### KnownConfigurations

Every configuration measured before a proposal turn, which the optimizer can restore and may not submit again.

**Parameters:**

- **reference_id** (<code>str</code>) – The reference's ID, which the optimizer restores as `"reference"`.
- **yaml_by_id** (<code>dict\[str, str\]</code>) – The YAML of the reference and of every earlier candidate, keyed by ID.

### ProposalResult

What one optimizer turn produced.

**Parameters:**

- **candidate** (<code>CandidateConfiguration | None</code>) – The submitted configuration, or `None` when the optimizer finished or ran out of steps.
- **finish_reason** (<code>str | None</code>) – Why the optimizer ended the search, when it called `finish`.
- **validation_failures** (<code>list\[dict\[str, str\]\]</code>) – Every draft that failed `validate_config` during the turn, as `{revision, error}`.

## haystack_integrations.agent_pack.optimization.tools

### inspect_component

```python
inspect_component(
    type_name: Annotated[
        str,
        "Fully qualified class name, as written in a configuration's 'type' field, for example 'haystack.components.rankers.llm_ranker.LLMRanker'. A path that does not import is retried where the class is installed, and the answer reports the path to use.",
    ],
) -> dict[str, str]
```

Inspect an installed class using the same namespace allowlist as deserialization.

### read_config

```python
read_config(draft: ConfigurationDraft) -> dict[str, str]
```

Read the entire editable YAML and its revision for subsequent edits.

### edit_config

```python
edit_config(
    old: Annotated[
        str,
        "Nonempty text to replace, matched literally and occurring exactly once in the current YAML. Include enough surrounding lines to be unique: a bare 'top_k: 2' or a type line repeated across components matches more than once and is rejected. Pass the entire YAML to rewrite the whole file.",
    ],
    new: Annotated[
        str, "Text replacing that block verbatim, or empty text to delete it."
    ],
    expected_revision: Annotated[
        str,
        "The revision returned by read_config or by the preceding edit, which must still be current.",
    ],
    draft: ConfigurationDraft,
) -> dict[str, Any]
```

Replace one exact text block. Use the entire current YAML as old for a full rewrite.

### ValidateConfig

Bases: <code>Tool</code>

The `validate_config` tool, holding the loader and the evaluator whose check a candidate has to pass.

#### __init__

```python
__init__(
    *,
    evaluator: HarnessEvaluator,
    loader: Callable[[str], Agent | Pipeline] = load_agent
) -> None
```

Create the tool.

**Parameters:**

- **evaluator** (<code>HarnessEvaluator</code>) – The evaluator the candidates are measured with. Its `validate` runs on every loaded
  configuration, and its `to_dict` and `from_dict` let this tool, and the agent holding it, be serialized.
- **loader** (<code>Callable\\[[str\], Agent | Pipeline\]</code>) – Builds the configuration from YAML, and decides what shape a candidate must keep. Defaults
  to the Agent contract; pass `load_pipeline` to optimize a Pipeline that is not a single Agent.

**Raises:**

- <code>TypeError</code> – If the evaluator is missing `validate`, `to_dict` or `from_dict`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the tool.

**Returns:**

- <code>dict\[str, Any\]</code> – The evaluator as its own `to_dict` describes it, and the loader as an import path.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ValidateConfig
```

Deserialize the tool.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – What `to_dict` returned.

**Returns:**

- <code>ValidateConfig</code> – The restored tool.

### submit_candidate

```python
submit_candidate(
    expected_revision: Annotated[
        str,
        "The revision returned by a successful validate_config, which must still be current.",
    ],
    rationale: Annotated[
        str,
        "The hypothesis this candidate tests: what was changed and what it is expected to move. Read back alongside the score, so name the change rather than restating the goal.",
    ],
    draft: ConfigurationDraft,
    known: KnownConfigurations,
) -> dict[str, Any]
```

Submit this validated revision for evaluation and end the proposal turn.

### restore_candidate

```python
restore_candidate(
    candidate_id: Annotated[
        str,
        "A candidate ID from the outcomes so far, or 'reference' for the original configuration.",
    ],
    expected_revision: Annotated[
        str, "The current revision, from read_config or the last edit."
    ],
    draft: ConfigurationDraft,
    known: KnownConfigurations,
) -> dict[str, Any]
```

Restore a submitted candidate or the reference as the base for further edits.

### finish

```python
finish(
    reason: Annotated[
        str,
        "What was considered and why none of it is worth measuring. This ends the experiment with the remaining evaluations unspent, and is the only record of why.",
    ],
) -> dict[str, str]
```

End optimization when no hypothesis worth measuring remains.
