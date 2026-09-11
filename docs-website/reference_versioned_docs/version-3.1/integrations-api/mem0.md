---
title: "Mem0"
id: integrations-mem0
description: "Mem0 integration for Haystack"
slug: "/integrations-mem0"
---


## haystack_integrations.components.retrievers.mem0.retriever

### Mem0MemoryRetriever

Retrieves memories from a Mem0MemoryStore as a list of ChatMessage objects.

Use this component in a Haystack Pipeline to fetch relevant memories before passing
context to a language model or Agent. The returned memories are system messages.

Provide either `filters` or at least one Mem0 entity ID (`user_id`, `run_id`, `agent_id`, or `app_id`)
when running the component. If both are provided, the filters and entity IDs are combined.

### Usage example

```python
from haystack_integrations.components.retrievers.mem0 import Mem0MemoryRetriever
from haystack_integrations.memory_stores.mem0 import Mem0MemoryStore

store = Mem0MemoryStore()
retriever = Mem0MemoryRetriever(memory_store=store, top_k=3)

result = retriever.run(query="What does Alice like?", user_id="alice")
memories = result["memories"]
print([message.text for message in memories])

# Pass query=None to retrieve all memories in scope.
all_memories = retriever.run(query=None, user_id="alice")["memories"]
```

#### __init__

```python
__init__(*, memory_store: Mem0MemoryStore, top_k: int = 5) -> None
```

Initialize the Mem0MemoryRetriever.

**Parameters:**

- **memory_store** (<code>Mem0MemoryStore</code>) – The Mem0MemoryStore instance to retrieve memories from.
- **top_k** (<code>int</code>) – Default maximum number of memories to return per query.

#### run

```python
run(
    query: str | None,
    *,
    user_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    app_id: str | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None
) -> dict[str, list[ChatMessage]]
```

Retrieve memories matching the query from Mem0.

**Parameters:**

- **query** (<code>str | None</code>) – Text query used to search for relevant memories. Pass `None` to retrieve all memories matching
  the scope.
- **user_id** (<code>str | None</code>) – User ID to scope the search.
- **run_id** (<code>str | None</code>) – Run ID to scope the search.
- **agent_id** (<code>str | None</code>) – Agent ID to scope the search.
- **app_id** (<code>str | None</code>) – App ID to scope the search.
- **filters** (<code>dict\[str, Any\] | None</code>) – Haystack-style filters to apply. When provided with ID parameters, they are combined.
  Mem0 requires entity IDs inside filters and supports a fixed set of native fields and operators:
  [Search Memories API](https://docs.mem0.ai/api-reference/memory/search-memories) and
  [Memory Filters](https://docs.mem0.ai/platform/features/v2-memory-filters). Fields that are not native
  Mem0 filter fields are treated as Mem0 metadata fields.
- **top_k** (<code>int | None</code>) – Maximum number of memories to return. Overrides the init-time default.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – Dictionary with key `memories` containing a list of ChatMessage objects. User-provided
  Mem0 metadata is included in each message's meta. Mem0 retrieval fields such as `memory_id`, `user_id`,
  `score`, and timestamps are included under `meta["mem0"]`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> Mem0MemoryRetriever
```

Deserialize this component from a dictionary.

## haystack_integrations.components.writers.mem0.writer

### Mem0MemoryWriter

Writes ChatMessage objects as memories to a Mem0MemoryStore.

Use this component in a Haystack Pipeline to persist conversation messages.
Scoping IDs (`user_id`, `run_id`, `agent_id`, `app_id`) are runtime parameters so the
same pipeline instance can serve multiple users or agents. The `infer` setting controls whether
Mem0 extracts memories from messages or stores message text as-is.

### Usage example

```python
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.writers.mem0 import Mem0MemoryWriter
from haystack_integrations.memory_stores.mem0 import Mem0MemoryStore

store = Mem0MemoryStore()
writer = Mem0MemoryWriter(memory_store=store, infer=False)

result = writer.run(
    messages=[ChatMessage.from_user("Alice prefers concise Python examples.")],
    user_id="alice",
)
print(result["memories_written"])
```

#### __init__

```python
__init__(*, memory_store: Mem0MemoryStore, infer: bool = True) -> None
```

Initialize the Mem0MemoryWriter.

**Parameters:**

- **memory_store** (<code>Mem0MemoryStore</code>) – The Mem0MemoryStore instance to write memories to.
- **infer** (<code>bool</code>) – If True, Mem0 extracts memories from messages. If False, Mem0 stores message text as-is.

#### run

```python
run(
    messages: list[ChatMessage],
    *,
    user_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    app_id: str | None = None
) -> dict[str, int]
```

Write messages as memories to the Mem0 store.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – List of ChatMessage objects to store.
- **user_id** (<code>str | None</code>) – User ID to scope the stored memories.
- **run_id** (<code>str | None</code>) – Run ID to scope the stored memories.
- **agent_id** (<code>str | None</code>) – Agent ID to scope the stored memories.
- **app_id** (<code>str | None</code>) – App ID to scope the stored memories.

**Returns:**

- <code>dict\[str, int\]</code> – Dictionary with key `memories_written` containing the count of stored memory items.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> Mem0MemoryWriter
```

Deserialize this component from a dictionary.

## haystack_integrations.memory_stores.mem0.errors

### Mem0MemoryStoreError

Bases: <code>RuntimeError</code>

Raised when a Mem0 API operation fails.

## haystack_integrations.memory_stores.mem0.memory_store

### Mem0MemoryStore

A memory store backed by the Mem0 cloud API.

Stores and retrieves ChatMessage-based memories scoped by user_id, run_id, agent_id, or app_id.
The Mem0 client is created lazily on first use (or explicitly via warm_up()).
Requires a Mem0 API key set via the MEM0_API_KEY environment variable or passed explicitly.

#### __init__

```python
__init__(*, api_key: Secret = Secret.from_env_var('MEM0_API_KEY')) -> None
```

Initialize the Mem0 memory store.

The Mem0 client is not created until warm_up() is called (or the first method that
needs the client is invoked).

**Parameters:**

- **api_key** (<code>Secret</code>) – The Mem0 API key. Defaults to the MEM0_API_KEY environment variable.

#### warm_up

```python
warm_up() -> None
```

Create the Mem0 client. Called automatically on first use if not called explicitly.

Calling this method explicitly is useful when you want to validate the API key
or pre-connect before the first pipeline run.

#### client

```python
client: MemoryClient
```

Return the initialized client, calling warm_up() if necessary.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the store configuration to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> Mem0MemoryStore
```

Deserialize the store from a dictionary.

#### add_memories

```python
add_memories(
    *,
    messages: list[ChatMessage],
    user_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    app_id: str | None = None,
    infer: bool = True,
    **kwargs: Any
) -> list[dict[str, Any]]
```

Add ChatMessage memories to Mem0.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – List of ChatMessage objects to store as memories.
- **user_id** (<code>str | None</code>) – User ID to scope these memories.
- **run_id** (<code>str | None</code>) – Run ID to scope these memories.
- **agent_id** (<code>str | None</code>) – Agent ID to scope these memories. Required for Mem0 to store assistant messages.
- **app_id** (<code>str | None</code>) – App ID to scope these memories.
- **infer** (<code>bool</code>) – If True, Mem0 extracts memories from messages. If False, Mem0 stores message text as-is.
- **kwargs** (<code>Any</code>) – Additional keyword arguments forwarded to the Mem0 client add method.
  Note: ChatMessage.meta is ignored because Mem0 doesn't support per-message metadata.
  Pass `metadata` as a kwarg to attach metadata to the whole batch instead.

**Returns:**

- <code>list\[dict\[str, Any\]\]</code> – List of objects with `memory_id` and `memory` text for each stored memory.

**Raises:**

- <code>Mem0MemoryStoreError</code> – If the Mem0 API call fails.

#### search_memories

```python
search_memories(
    *,
    query: str | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
    user_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    app_id: str | None = None,
    **kwargs: Any
) -> list[ChatMessage]
```

Search for memories in Mem0.

Either `filters` or at least one of `user_id`, `run_id`, `agent_id`, or `app_id` must be provided.
When both `filters` and IDs are provided, they are combined with an `AND` condition.

**Parameters:**

- **query** (<code>str | None</code>) – Text query to search. If omitted, returns all memories matching the scope.
- **filters** (<code>dict\[str, Any\] | None</code>) – Haystack-style filters to apply. See
  [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).
  Mem0 requires entity IDs inside filters and supports a fixed set of native fields and operators:
  [Search Memories API](https://docs.mem0.ai/api-reference/memory/search-memories) and
  [Memory Filters](https://docs.mem0.ai/platform/features/v2-memory-filters).
  Fields that are not native Mem0 filter fields are treated as Mem0 metadata fields.
- **top_k** (<code>int</code>) – Maximum number of results to return.
- **user_id** (<code>str | None</code>) – User ID to scope the search.
- **run_id** (<code>str | None</code>) – Run ID to scope the search.
- **agent_id** (<code>str | None</code>) – Agent ID to scope the search.
- **app_id** (<code>str | None</code>) – App ID to scope the search.
- **kwargs** (<code>Any</code>) – Additional keyword arguments forwarded to the Mem0 client.

**Returns:**

- <code>list\[ChatMessage\]</code> – List of ChatMessage (system role) objects containing the retrieved memories. User-provided
  Mem0 metadata is included in each message's meta. Mem0 retrieval fields such as `memory_id`, `user_id`,
  `score`, and timestamps are included under `meta["mem0"]`.

**Raises:**

- <code>Mem0MemoryStoreError</code> – If the Mem0 API call fails.

## haystack_integrations.tools.mem0.retriever_tool

### Mem0MemoryRetrieverTool

Bases: <code>Tool</code>

A tool that searches a Mem0MemoryStore for memories.

The `user_id` is injected at runtime from Agent State via `inputs_from_state`,
so a single tool instance can serve many users. The LLM only sees `query` and `top_k` by default.
If the LLM omits `query` or passes `None`, Mem0 returns all memories matching the injected scope.
Pass a custom `inputs_from_state` mapping to inject other supported Mem0 entity IDs such as
`run_id`, `agent_id`, or `app_id`. The mapping keys are Agent State keys and the values are this
tool's parameter names. For example, use
`inputs_from_state={"user_id": "user_id", "session_id": "run_id"}` to pass `state["session_id"]`
to the tool's `run_id` parameter at runtime.

### Usage example

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.memory_stores.mem0 import Mem0MemoryStore
from haystack_integrations.tools.mem0 import Mem0MemoryRetrieverTool

store = Mem0MemoryStore()
retrieve_memories = Mem0MemoryRetrieverTool(memory_store=store, top_k=5)

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=[retrieve_memories],
    state_schema={"user_id": {"type": str}, "session_id": {"type": str}},
)

# The Agent can call retrieve_memories with a query for targeted recall,
# or without a query when it needs all scoped memories.
result = agent.run(
    messages=[ChatMessage.from_user("What do you remember about me?")],
    user_id="alice",
    session_id="chat-42",
)
print(result["last_message"].text)
```

#### __init__

```python
__init__(
    *,
    memory_store: Mem0MemoryStore,
    top_k: int = 5,
    name: str = "retrieve_memories",
    description: str = _DEFAULT_DESCRIPTION,
    parameters: dict[str, Any] = _PARAMETERS,
    inputs_from_state: dict[str, str] = _DEFAULT_INPUTS_FROM_STATE
) -> None
```

Initialize the Mem0MemoryRetrieverTool.

**Parameters:**

- **memory_store** (<code>Mem0MemoryStore</code>) – The Mem0MemoryStore instance to query.
- **top_k** (<code>int</code>) – Default maximum number of memories to return. The LLM may override this.
- **name** (<code>str</code>) – Tool name exposed to the LLM.
- **description** (<code>str</code>) – Tool description exposed to the LLM.
- **parameters** (<code>dict\[str, Any\]</code>) – JSON schema for the parameters exposed to the LLM. Defaults to optional `query` and `top_k`.
- **inputs_from_state** (<code>dict\[str, str\]</code>) – Mapping from Agent State keys to this tool's parameter names.
  Defaults to `{"user_id": "user_id"}`, which injects `state["user_id"]` into the `user_id`
  parameter. To pass more Mem0 IDs at runtime, add the state fields to the Agent's
  `state_schema` and map them to the corresponding tool parameters, for example
  `{"user_id": "user_id", "session_id": "run_id", "agent_name": "agent_id", "app_name": "app_id"}`.

#### warm_up

```python
warm_up() -> None
```

Initialize the Mem0 client. Subsequent calls are no-ops.

#### retrieve

```python
retrieve(
    query: str | None = None,
    *,
    top_k: int | None = None,
    user_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    app_id: str | None = None
) -> str
```

Retrieve memories relevant to a query, or all memories when no query is provided.

**Parameters:**

- **query** (<code>str | None</code>) – Text query used to search for relevant memories. If omitted or `None`, all memories matching
  the scope are returned.
- **top_k** (<code>int | None</code>) – Maximum number of memories to return for query searches. Overrides the tool default.
- **user_id** (<code>str | None</code>) – User ID to scope the search. Injected from Agent State by default.
- **run_id** (<code>str | None</code>) – Run ID to scope the search. Can be injected with a custom `inputs_from_state` mapping.
- **agent_id** (<code>str | None</code>) – Agent ID to scope the search. Can be injected with a custom `inputs_from_state` mapping.
- **app_id** (<code>str | None</code>) – App ID to scope the search. Can be injected with a custom `inputs_from_state` mapping.

**Returns:**

- <code>str</code> – Retrieved memories formatted for the Agent, or a message when no memories were found.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this tool to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> Mem0MemoryRetrieverTool
```

Deserialize this tool from a dictionary.

## haystack_integrations.tools.mem0.writer_tool

### Mem0MemoryWriterTool

Bases: <code>Tool</code>

A tool that writes a memory to a Mem0MemoryStore.

The `user_id` is injected at runtime from Agent State via `inputs_from_state`,
so a single tool instance can serve many users. The LLM only sees `text` and `infer`.
Pass a custom `inputs_from_state` mapping to inject other supported Mem0 entity IDs such as
`run_id`, `agent_id`, or `app_id`. The mapping keys are Agent State keys and the values are this
tool's parameter names. For example, use
`inputs_from_state={"user_id": "user_id", "session_id": "run_id"}` to pass `state["session_id"]`
to the tool's `run_id` parameter at runtime.

### Usage example

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.memory_stores.mem0 import Mem0MemoryStore
from haystack_integrations.tools.mem0 import Mem0MemoryWriterTool

store = Mem0MemoryStore()
store_memory = Mem0MemoryWriterTool(memory_store=store)

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=[store_memory],
    state_schema={"user_id": {"type": str}, "session_id": {"type": str}},
)

result = agent.run(
    messages=[ChatMessage.from_user("Remember that I prefer concise Python examples.")],
    user_id="alice",
    session_id="chat-42",
)
print(result["last_message"].text)
```

#### __init__

```python
__init__(
    *,
    memory_store: Mem0MemoryStore,
    name: str = "store_memory",
    description: str = _DEFAULT_DESCRIPTION,
    parameters: dict[str, Any] = _PARAMETERS,
    inputs_from_state: dict[str, str] = _DEFAULT_INPUTS_FROM_STATE
) -> None
```

Initialize the Mem0MemoryWriterTool.

**Parameters:**

- **memory_store** (<code>Mem0MemoryStore</code>) – The Mem0MemoryStore instance to write to.
- **name** (<code>str</code>) – Tool name exposed to the LLM.
- **description** (<code>str</code>) – Tool description exposed to the LLM.
- **parameters** (<code>dict\[str, Any\]</code>) – JSON schema for the parameters exposed to the LLM. Defaults to `text` and `infer`.
- **inputs_from_state** (<code>dict\[str, str\]</code>) – Mapping from Agent State keys to this tool's parameter names.
  Defaults to `{"user_id": "user_id"}`, which injects `state["user_id"]` into the `user_id`
  parameter. To pass more Mem0 IDs at runtime, add the state fields to the Agent's
  `state_schema` and map them to the corresponding tool parameters, for example
  `{"user_id": "user_id", "session_id": "run_id", "agent_name": "agent_id", "app_name": "app_id"}`.

#### warm_up

```python
warm_up() -> None
```

Initialize the Mem0 client. Subsequent calls are no-ops.

#### store

```python
store(
    text: str,
    *,
    infer: bool = False,
    user_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    app_id: str | None = None
) -> str
```

Store text as a memory.

**Parameters:**

- **text** (<code>str</code>) – The information to store as a memory.
- **infer** (<code>bool</code>) – If True, Mem0 extracts memories from the text. If False, Mem0 stores the text as-is.
- **user_id** (<code>str | None</code>) – User ID to scope the stored memory. Injected from Agent State by default.
- **run_id** (<code>str | None</code>) – Run ID to scope the stored memory. Can be injected with a custom `inputs_from_state` mapping.
- **agent_id** (<code>str | None</code>) – Agent ID to scope the stored memory. Can be injected with a custom `inputs_from_state` mapping.
- **app_id** (<code>str | None</code>) – App ID to scope the stored memory. Can be injected with a custom `inputs_from_state` mapping.

**Returns:**

- <code>str</code> – A string indicating how many memory items were stored.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this tool to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> Mem0MemoryWriterTool
```

Deserialize this tool from a dictionary.
