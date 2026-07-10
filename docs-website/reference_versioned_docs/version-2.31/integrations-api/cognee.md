---
title: "Cognee"
id: integrations-cognee
description: "Cognee integration for Haystack"
slug: "/integrations-cognee"
---


## haystack_integrations.components.retrievers.cognee.memory_retriever

### CogneeRetriever

Retrieves memories from a `CogneeMemoryStore` as `ChatMessage` instances.

Configuration (`search_type`, `top_k`, `dataset_name`, `session_id`) lives on
the store; this retriever is a thin pipeline adapter over `search_memories`.

#### __init__

```python
__init__(*, memory_store: CogneeMemoryStore, top_k: int | None = None) -> None
```

Initialize the retriever.

**Parameters:**

- **memory_store** (<code>CogneeMemoryStore</code>) – Backing `CogneeMemoryStore` to query.
- **top_k** (<code>int | None</code>) – Default max results; falls back to the store's `top_k` when `None`.

#### run

```python
run(
    query: str, top_k: int | None = None, user_id: str | None = None
) -> dict[str, list[ChatMessage]]
```

Search the attached store and return matching memories as ChatMessages.

**Parameters:**

- **query** (<code>str</code>) – Natural-language query.
- **top_k** (<code>int | None</code>) – Per-call override; falls back to init `top_k`, then the store's default.
- **user_id** (<code>str | None</code>) – Cognee user UUID; scopes the search to that user.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> CogneeRetriever
```

Deserialize a component from a dictionary.

## haystack_integrations.components.writers.cognee.memory_writer

### CogneeWriter

Persists `ChatMessage`s into a `CogneeMemoryStore`.

Use without `session_id` to write to the permanent graph; pass `session_id` to
target cognee's session cache for that writer's writes. The writer's
`session_id` overrides the store's own `session_id` per call, so one store can
back multiple writers writing to different tiers.

#### __init__

```python
__init__(
    *, memory_store: CogneeMemoryStore, session_id: str | None = None
) -> None
```

Initialize the writer.

**Parameters:**

- **memory_store** (<code>CogneeMemoryStore</code>) – Backing `CogneeMemoryStore` to write into.
- **session_id** (<code>str | None</code>) – Overrides the store's `session_id` for this writer's writes.

#### run

```python
run(
    messages: list[ChatMessage], user_id: str | None = None
) -> dict[str, list[ChatMessage]]
```

Store `messages` in Cognee memory and pass them through unchanged.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – Messages to persist.
- **user_id** (<code>str | None</code>) – Cognee user UUID; scopes the write to that user.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> CogneeWriter
```

Deserialize a component from a dictionary.

## haystack_integrations.memory_stores.cognee.memory_store

### CogneeMemoryStore

Memory backend backed by Cognee, implementing the haystack-experimental `MemoryStore` protocol.

Wraps cognee's V2 memory API: `add_memories` -> `cognee.remember`,
`search_memories` -> `cognee.recall`, `improve` -> `cognee.improve`,
`delete_all_memories` -> `cognee.forget`.

`session_id` selects the tier — set it to use cognee's session cache (cheap,
no LLM extraction, session-aware recall); leave `None` for the permanent
graph.

`self_improvement` is forwarded to `cognee.remember` and defaults to `True`
(same as cognee). On the permanent tier it awaits `improve` inline; on the
session tier it schedules `improve` as a fire-and-forget background task.
Set to `False` when you want `improve()` to be the only improve trigger
— otherwise an explicit `improve()` runs improve twice and produces
near-duplicate graph nodes.

`timeout` (seconds) caps how long any single cognee call may run before
raising `concurrent.futures.TimeoutError`. The default of 300s covers
single-message agent-memory writes comfortably; bulk ingestion of long
documents may need a larger value.

#### __init__

```python
__init__(
    *,
    search_type: CogneeSearchType = "GRAPH_COMPLETION",
    top_k: int = 5,
    dataset_name: str = "haystack_memory",
    session_id: str | None = None,
    self_improvement: bool = True,
    timeout: float = 300
) -> None
```

Initialize the store.

**Parameters:**

- **search_type** (<code>CogneeSearchType</code>) – Cognee search strategy used by `search_memories`.
- **top_k** (<code>int</code>) – Default max results for `search_memories`.
- **dataset_name** (<code>str</code>) – Cognee dataset backing this store.
- **session_id** (<code>str | None</code>) – When set, use the session-cache tier; otherwise the permanent graph.
- **self_improvement** (<code>bool</code>) – Forwarded to `cognee.remember` (default `True`, matches cognee).
  Set to `False` when `improve()` should be the only improve trigger.
- **timeout** (<code>float</code>) – Per-call timeout in seconds for any cognee operation.
  Raise this for bulk ingestion workloads that legitimately need >300s.

#### add_memories

```python
add_memories(
    *,
    messages: list[ChatMessage],
    user_id: str | None = None,
    session_id: str | None = None
) -> None
```

Persist messages via `cognee.remember`.

Permanent tier batches all texts into one call; session tier writes one
entry per message (matches cognee's session example). Empty messages
are skipped.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – Messages to store.
- **user_id** (<code>str | None</code>) – Cognee user UUID; `None` uses cognee's default user.
- **session_id** (<code>str | None</code>) – Per-call override of the store's `session_id`.

#### search_memories

```python
search_memories(
    *,
    query: str | None = None,
    top_k: int | None = None,
    user_id: str | None = None
) -> list[ChatMessage]
```

Search via `cognee.recall` and wrap each hit in a system `ChatMessage`.

**Parameters:**

- **query** (<code>str | None</code>) – Natural-language query. Empty/`None` returns `[]`.
- **top_k** (<code>int | None</code>) – Per-call override of the store's default.
- **user_id** (<code>str | None</code>) – Cognee user UUID; `None` uses cognee's default user.

#### improve

```python
improve(*, session_id: str | None = None, user_id: str | None = None) -> None
```

Promote session-cache content into the permanent graph via `cognee.improve`.

Without any session_id this is a plain graph-enrichment pass.

**Parameters:**

- **session_id** (<code>str | None</code>) – Session to promote; defaults to the store's `session_id`.
- **user_id** (<code>str | None</code>) – Cognee user UUID; `None` uses cognee's default user.

#### delete_all_memories

```python
delete_all_memories(*, user_id: str | None = None) -> None
```

Delete this dataset via `cognee.forget(dataset=...)`.

Session cache survives (sessions aren't dataset-scoped) — use
`cognee.forget(everything=True)` for a full wipe.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this store for pipeline persistence.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> CogneeMemoryStore
```

Deserialize a store from a dict produced by `to_dict`.
