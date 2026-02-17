---
title: "Retrievers"
id: experimental-retrievers-api
description: "Sweep through Document Stores and return a set of candidate documents that are relevant to the query."
slug: "/experimental-retrievers-api"
---


## `haystack-experimental.haystack_experimental.components.retrievers.chat_message_retriever`

### `ChatMessageRetriever`

Retrieves chat messages from the underlying ChatMessageStore.

Usage example:

```python
from haystack.dataclasses import ChatMessage
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore

messages = [
    ChatMessage.from_assistant("Hello, how can I help you?"),
    ChatMessage.from_user("Hi, I have a question about Python. What is a Protocol?"),
]

message_store = InMemoryChatMessageStore()
message_store.write_messages(chat_history_id="user_456_session_123", messages=messages)
retriever = ChatMessageRetriever(message_store)

result = retriever.run(chat_history_id="user_456_session_123")

print(result["messages"])
```

#### `__init__`

```python
__init__(chat_message_store: ChatMessageStore, last_k: int | None = 10)
```

Create the ChatMessageRetriever component.

**Parameters:**

- **chat_message_store** (<code>ChatMessageStore</code>) – An instance of a ChatMessageStore.
- **last_k** (<code>int | None</code>) – The number of last messages to retrieve. Defaults to 10 messages if not specified.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ChatMessageRetriever
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>ChatMessageRetriever</code> – The deserialized component.

#### `run`

```python
run(
    chat_history_id: str,
    *,
    last_k: int | None = None,
    current_messages: list[ChatMessage] | None = None
) -> dict[str, list[ChatMessage]]
```

Run the ChatMessageRetriever

**Parameters:**

- **chat_history_id** (<code>str</code>) – A unique identifier for the chat session or conversation whose messages should be retrieved.
  Each `chat_history_id` corresponds to a distinct chat history stored in the underlying ChatMessageStore.
  For example, use a session ID or conversation ID to isolate messages from different chat sessions.
- **last_k** (<code>int | None</code>) – The number of last messages to retrieve. This parameter takes precedence over the last_k
  parameter passed to the ChatMessageRetriever constructor. If unspecified, the last_k parameter passed
  to the constructor will be used.
- **current_messages** (<code>list\[ChatMessage\] | None</code>) – A list of incoming chat messages to combine with the retrieved messages. System messages from this list
  are prepended before the retrieved history, while all other messages (e.g., user messages) are appended
  after. This is useful for including new conversational context alongside stored history so the output
  can be directly used as input to a ChatGenerator or an Agent. If not provided, only the stored messages
  will be returned.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following key:
- `messages` - The retrieved chat messages combined with any provided current messages.

**Raises:**

- <code>ValueError</code> – If last_k is not None and is less than 0.
