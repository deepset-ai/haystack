---
title: "Writers"
id: experimental-writers-api
description: "Writers for Haystack."
slug: "/experimental-writers-api"
---


## `haystack-experimental.haystack_experimental.components.writers.chat_message_writer`

### `ChatMessageWriter`

Writes chat messages to an underlying ChatMessageStore.

Usage example:

```python
from haystack.dataclasses import ChatMessage
from haystack_experimental.components.writers import ChatMessageWriter
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore

messages = [
    ChatMessage.from_assistant("Hello, how can I help you?"),
    ChatMessage.from_user("I have a question about Python."),
]
message_store = InMemoryChatMessageStore()
writer = ChatMessageWriter(message_store)
writer.run(chat_history_id="user_456_session_123", messages=messages)
```

#### `__init__`

```python
__init__(chat_message_store: ChatMessageStore) -> None
```

Create a ChatMessageWriter component.

**Parameters:**

- **chat_message_store** (<code>ChatMessageStore</code>) – The ChatMessageStore where the chat messages are to be written.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ChatMessageWriter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>ChatMessageWriter</code> – The deserialized component.

**Raises:**

- <code>DeserializationError</code> – If the message store is not properly specified in the serialization data or its type cannot be imported.

#### `run`

```python
run(chat_history_id: str, messages: list[ChatMessage]) -> dict[str, int]
```

Run the ChatMessageWriter on the given input data.

**Parameters:**

- **chat_history_id** (<code>str</code>) – A unique identifier for the chat session or conversation whose messages should be retrieved.
  Each `chat_history_id` corresponds to a distinct chat history stored in the underlying ChatMessageStore.
  For example, use a session ID or conversation ID to isolate messages from different chat sessions.
- **messages** (<code>list\[ChatMessage\]</code>) – A list of chat messages to write to the store.

**Returns:**

- <code>dict\[str, int\]</code> – - `messages_written`: Number of messages written to the ChatMessageStore.
