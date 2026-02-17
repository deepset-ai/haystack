---
title: "ChatMessage Store"
id: experimental-chatmessage-store-api
description: "Storage for the chat messages."
slug: "/experimental-chatmessage-store-api"
---


## `haystack_experimental.chat_message_stores.in_memory`

### `InMemoryChatMessageStore`

Stores chat messages in-memory.

The `chat_history_id` parameter is used as a unique identifier for each conversation or chat session.
It acts as a namespace that isolates messages from different sessions. Each `chat_history_id` value corresponds to a
separate list of `ChatMessage` objects stored in memory.

Typical usage involves providing a unique `chat_history_id` (for example, a session ID or conversation ID)
whenever you write, read, or delete messages. This ensures that chat messages from different
conversations do not overlap.

Usage example:

```python
from haystack.dataclasses import ChatMessage
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore

message_store = InMemoryChatMessageStore()

messages = [
    ChatMessage.from_assistant("Hello, how can I help you?"),
    ChatMessage.from_user("Hi, I have a question about Python. What is a Protocol?"),
]
message_store.write_messages(chat_history_id="user_456_session_123", messages=messages)
retrieved_messages = message_store.retrieve_messages(chat_history_id="user_456_session_123")

print(retrieved_messages)
```

#### `__init__`

```python
__init__(skip_system_messages: bool = True, last_k: int | None = 10) -> None
```

Create an InMemoryChatMessageStore.

**Parameters:**

- **skip_system_messages** (<code>bool</code>) – Whether to skip storing system messages. Defaults to True.
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
from_dict(data: dict[str, Any]) -> InMemoryChatMessageStore
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>InMemoryChatMessageStore</code> – The deserialized component.

#### `count_messages`

```python
count_messages(chat_history_id: str) -> int
```

Returns the number of chat messages stored in this store.

**Parameters:**

- **chat_history_id** (<code>str</code>) – The chat history id for which to count messages.

**Returns:**

- <code>int</code> – The number of messages.

#### `write_messages`

```python
write_messages(chat_history_id: str, messages: list[ChatMessage]) -> int
```

Writes chat messages to the ChatMessageStore.

**Parameters:**

- **chat_history_id** (<code>str</code>) – The chat history id under which to store the messages.
- **messages** (<code>list\[ChatMessage\]</code>) – A list of ChatMessages to write.

**Returns:**

- <code>int</code> – The number of messages written.

**Raises:**

- <code>ValueError</code> – If messages is not a list of ChatMessages.

#### `retrieve_messages`

```python
retrieve_messages(
    chat_history_id: str, last_k: int | None = None
) -> list[ChatMessage]
```

Retrieves all stored chat messages.

**Parameters:**

- **chat_history_id** (<code>str</code>) – The chat history id from which to retrieve messages.
- **last_k** (<code>int | None</code>) – The number of last messages to retrieve. If unspecified, the last_k parameter passed
  to the constructor will be used.

**Returns:**

- <code>list\[ChatMessage\]</code> – A list of chat messages.

**Raises:**

- <code>ValueError</code> – If last_k is not None and is less than 0.

#### `delete_messages`

```python
delete_messages(chat_history_id: str) -> None
```

Deletes all stored chat messages.

**Parameters:**

- **chat_history_id** (<code>str</code>) – The chat history id from which to delete messages.

#### `delete_all_messages`

```python
delete_all_messages() -> None
```

Deletes all stored chat messages from all chat history ids.
