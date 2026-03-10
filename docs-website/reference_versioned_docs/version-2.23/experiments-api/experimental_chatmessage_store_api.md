---
title: "ChatMessage Store"
id: experimental-chatmessage-store-api
description: "Storage for the chat messages."
slug: "/experimental-chatmessage-store-api"
---

<a id="haystack_experimental.chat_message_stores.in_memory"></a>

## Module haystack\_experimental.chat\_message\_stores.in\_memory

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore"></a>

### InMemoryChatMessageStore

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

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.__init__"></a>

#### InMemoryChatMessageStore.\_\_init\_\_

```python
def __init__(skip_system_messages: bool = True,
             last_k: int | None = 10) -> None
```

Create an InMemoryChatMessageStore.

**Arguments**:

- `skip_system_messages`: Whether to skip storing system messages. Defaults to True.
- `last_k`: The number of last messages to retrieve. Defaults to 10 messages if not specified.

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.to_dict"></a>

#### InMemoryChatMessageStore.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.from_dict"></a>

#### InMemoryChatMessageStore.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "InMemoryChatMessageStore"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.count_messages"></a>

#### InMemoryChatMessageStore.count\_messages

```python
def count_messages(chat_history_id: str) -> int
```

Returns the number of chat messages stored in this store.

**Arguments**:

- `chat_history_id`: The chat history id for which to count messages.

**Returns**:

The number of messages.

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.write_messages"></a>

#### InMemoryChatMessageStore.write\_messages

```python
def write_messages(chat_history_id: str, messages: list[ChatMessage]) -> int
```

Writes chat messages to the ChatMessageStore.

**Arguments**:

- `chat_history_id`: The chat history id under which to store the messages.
- `messages`: A list of ChatMessages to write.

**Raises**:

- `ValueError`: If messages is not a list of ChatMessages.

**Returns**:

The number of messages written.

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.retrieve_messages"></a>

#### InMemoryChatMessageStore.retrieve\_messages

```python
def retrieve_messages(chat_history_id: str,
                      last_k: int | None = None) -> list[ChatMessage]
```

Retrieves all stored chat messages.

**Arguments**:

- `chat_history_id`: The chat history id from which to retrieve messages.
- `last_k`: The number of last messages to retrieve. If unspecified, the last_k parameter passed
to the constructor will be used.

**Raises**:

- `ValueError`: If last_k is not None and is less than 0.

**Returns**:

A list of chat messages.

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.delete_messages"></a>

#### InMemoryChatMessageStore.delete\_messages

```python
def delete_messages(chat_history_id: str) -> None
```

Deletes all stored chat messages.

**Arguments**:

- `chat_history_id`: The chat history id from which to delete messages.

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.delete_all_messages"></a>

#### InMemoryChatMessageStore.delete\_all\_messages

```python
def delete_all_messages() -> None
```

Deletes all stored chat messages from all chat history ids.

