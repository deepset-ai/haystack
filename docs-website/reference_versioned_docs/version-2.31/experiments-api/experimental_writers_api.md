---
title: "Writers"
id: experimental-writers-api
description: "Writers for Haystack."
slug: "/experimental-writers-api"
---

<a id="haystack_experimental.components.writers.chat_message_writer"></a>

## Module haystack\_experimental.components.writers.chat\_message\_writer

<a id="haystack_experimental.components.writers.chat_message_writer.ChatMessageWriter"></a>

### ChatMessageWriter

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

<a id="haystack_experimental.components.writers.chat_message_writer.ChatMessageWriter.__init__"></a>

#### ChatMessageWriter.\_\_init\_\_

```python
def __init__(chat_message_store: ChatMessageStore) -> None
```

Create a ChatMessageWriter component.

**Arguments**:

- `chat_message_store`: The ChatMessageStore where the chat messages are to be written.

<a id="haystack_experimental.components.writers.chat_message_writer.ChatMessageWriter.to_dict"></a>

#### ChatMessageWriter.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_experimental.components.writers.chat_message_writer.ChatMessageWriter.from_dict"></a>

#### ChatMessageWriter.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ChatMessageWriter"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Raises**:

- `DeserializationError`: If the message store is not properly specified in the serialization data or its type cannot be imported.

**Returns**:

The deserialized component.

<a id="haystack_experimental.components.writers.chat_message_writer.ChatMessageWriter.run"></a>

#### ChatMessageWriter.run

```python
@component.output_types(messages_written=int)
def run(chat_history_id: str, messages: list[ChatMessage]) -> dict[str, int]
```

Run the ChatMessageWriter on the given input data.

**Arguments**:

- `chat_history_id`: A unique identifier for the chat session or conversation whose messages should be retrieved.
Each `chat_history_id` corresponds to a distinct chat history stored in the underlying ChatMessageStore.
For example, use a session ID or conversation ID to isolate messages from different chat sessions.
- `messages`: A list of chat messages to write to the store.

**Returns**:

- `messages_written`: Number of messages written to the ChatMessageStore.

