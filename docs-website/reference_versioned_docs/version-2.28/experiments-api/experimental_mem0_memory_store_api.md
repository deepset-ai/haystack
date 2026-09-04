---
title: "Mem0 Memory Store"
id: experimental-mem0-memory-store-api
description: "Storage for the memories using Mem0 as the backend."
slug: "/experimental-mem0-memory-store-api"
---

<a id="haystack_experimental.memory_stores.mem0.memory_store"></a>

## Module haystack\_experimental.memory\_stores.mem0.memory\_store

<a id="haystack_experimental.memory_stores.mem0.memory_store.Mem0MemoryStore"></a>

### Mem0MemoryStore

A memory store implementation using Mem0 as the backend.

<a id="haystack_experimental.memory_stores.mem0.memory_store.Mem0MemoryStore.__init__"></a>

#### Mem0MemoryStore.\_\_init\_\_

```python
def __init__(*, api_key: Secret = Secret.from_env_var("MEM0_API_KEY"))
```

Initialize the Mem0 memory store.

**Arguments**:

- `api_key`: The Mem0 API key. You can also set it using `MEM0_API_KEY` environment variable.

<a id="haystack_experimental.memory_stores.mem0.memory_store.Mem0MemoryStore.to_dict"></a>

#### Mem0MemoryStore.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the store configuration to a dictionary.

<a id="haystack_experimental.memory_stores.mem0.memory_store.Mem0MemoryStore.from_dict"></a>

#### Mem0MemoryStore.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "Mem0MemoryStore"
```

Deserialize the store from a dictionary.

<a id="haystack_experimental.memory_stores.mem0.memory_store.Mem0MemoryStore.add_memories"></a>

#### Mem0MemoryStore.add\_memories

```python
def add_memories(*,
                 messages: list[ChatMessage],
                 infer: bool = True,
                 user_id: str | None = None,
                 run_id: str | None = None,
                 agent_id: str | None = None,
                 async_mode: bool = False,
                 **kwargs: Any) -> list[dict[str, Any]]
```

Add ChatMessage memories to Mem0.

**Arguments**:

- `messages`: List of ChatMessage objects with memory metadata
- `infer`: Whether to infer facts from the messages. If False, the whole message will
be added as a memory.
- `user_id`: The user ID to to store and retrieve memories from the memory store.
- `run_id`: The run ID to to store and retrieve memories from the memory store.
- `agent_id`: The agent ID to to store and retrieve memories from the memory store.
If you want Mem0 to store chat messages from the assistant, you need to set the agent_id.
- `async_mode`: Whether to add memories asynchronously.
If True, the method will return immediately and the memories will be added in the background.
- `kwargs`: Additional keyword arguments to pass to the Mem0 client.add method.
Note: ChatMessage.meta in the list of messages will be ignored because Mem0 doesn't allow
passing metadata for each message in the list. You can pass metadata for the whole memory
by passing the `metadata` keyword argument to the method.

**Returns**:

List of objects with the memory_id and the memory

<a id="haystack_experimental.memory_stores.mem0.memory_store.Mem0MemoryStore.search_memories"></a>

#### Mem0MemoryStore.search\_memories

```python
def search_memories(*,
                    query: str | None = None,
                    filters: dict[str, Any] | None = None,
                    top_k: int = 5,
                    user_id: str | None = None,
                    run_id: str | None = None,
                    agent_id: str | None = None,
                    include_memory_metadata: bool = False,
                    **kwargs: Any) -> list[ChatMessage]
```

Search for memories in Mem0.

If filters are not provided, at least one of user_id, run_id, or agent_id must be set.
If filters are provided, the search will be scoped to the provided filters and the other ids will be ignored.

**Arguments**:

- `query`: Text query to search for. If not provided, all memories will be returned.
- `filters`: Haystack filters to apply on search. For more details on Haystack filters, see https://docs.haystack.deepset.ai/docs/metadata-filtering
- `top_k`: Maximum number of results to return
- `user_id`: The user ID to to store and retrieve memories from the memory store.
- `run_id`: The run ID to to store and retrieve memories from the memory store.
- `agent_id`: The agent ID to to store and retrieve memories from the memory store.
If you want Mem0 to store chat messages from the assistant, you need to set the agent_id.
- `include_memory_metadata`: Whether to include the mem0 related metadata for the
retrieved memory in the ChatMessage.
If True, the metadata will include the mem0 related metadata i.e. memory_id, score, etc.
in the `mem0_memory_metadata` key.
If False, the `ChatMessage.meta` will only contain the user defined metadata.
- `kwargs`: Additional keyword arguments to pass to the Mem0 client.
If query is passed, the kwargs will be passed to the Mem0 client.search method.
If query is not passed, the kwargs will be passed to the Mem0 client.get_all method.

**Returns**:

List of ChatMessage memories matching the criteria

<a id="haystack_experimental.memory_stores.mem0.memory_store.Mem0MemoryStore.search_memories_as_single_message"></a>

#### Mem0MemoryStore.search\_memories\_as\_single\_message

```python
def search_memories_as_single_message(*,
                                      query: str | None = None,
                                      filters: dict[str, Any] | None = None,
                                      top_k: int = 5,
                                      user_id: str | None = None,
                                      run_id: str | None = None,
                                      agent_id: str | None = None,
                                      **kwargs: Any) -> ChatMessage
```

Search for memories in Mem0 and return a single ChatMessage object.

If filters are not provided, at least one of user_id, run_id, or agent_id must be set.
If filters are provided, the search will be scoped to the provided filters and the other ids will be ignored.

**Arguments**:

- `query`: Text query to search for. If not provided, all memories will be returned.
- `filters`: Additional filters to apply on search. For more details on mem0 filters, see https://mem0.ai/docs/search/
- `top_k`: Maximum number of results to return
- `user_id`: The user ID to to store and retrieve memories from the memory store.
- `run_id`: The run ID to to store and retrieve memories from the memory store.
- `agent_id`: The agent ID to to store and retrieve memories from the memory store.
If you want Mem0 to store chat messages from the assistant, you need to set the agent_id.
- `kwargs`: Additional keyword arguments to pass to the Mem0 client.
If query is passed, the kwargs will be passed to the Mem0 client.search method.
If query is not passed, the kwargs will be passed to the Mem0 client.get_all method.

**Returns**:

A single ChatMessage object with the memories matching the criteria

<a id="haystack_experimental.memory_stores.mem0.memory_store.Mem0MemoryStore.delete_all_memories"></a>

#### Mem0MemoryStore.delete\_all\_memories

```python
def delete_all_memories(*,
                        user_id: str | None = None,
                        run_id: str | None = None,
                        agent_id: str | None = None,
                        **kwargs: Any) -> None
```

Delete memory records from Mem0.

At least one of user_id, run_id, or agent_id must be set.

**Arguments**:

- `user_id`: The user ID to delete memories from.
- `run_id`: The run ID to delete memories from.
- `agent_id`: The agent ID to delete memories from.
- `kwargs`: Additional keyword arguments to pass to the Mem0 client.delete_all method.

<a id="haystack_experimental.memory_stores.mem0.memory_store.Mem0MemoryStore.delete_memory"></a>

#### Mem0MemoryStore.delete\_memory

```python
def delete_memory(memory_id: str, **kwargs: Any) -> None
```

Delete memory from Mem0.

**Arguments**:

- `memory_id`: The ID of the memory to delete.
- `kwargs`: Additional keyword arguments to pass to the Mem0 client.delete method.

<a id="haystack_experimental.memory_stores.mem0.memory_store.Mem0MemoryStore.normalize_filters"></a>

#### Mem0MemoryStore.normalize\_filters

```python
@staticmethod
def normalize_filters(filters: dict[str, Any]) -> dict[str, Any]
```

Convert Haystack filters to Mem0 filters.

