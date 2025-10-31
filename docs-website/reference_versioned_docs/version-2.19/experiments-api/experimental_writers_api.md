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
writer.run(messages)
```

<a id="haystack_experimental.components.writers.chat_message_writer.ChatMessageWriter.__init__"></a>

#### ChatMessageWriter.\_\_init\_\_

```python
def __init__(message_store: ChatMessageStore)
```

Create a ChatMessageWriter component.

**Arguments**:

- `message_store`: The ChatMessageStore where the chat messages are to be written.

<a id="haystack_experimental.components.writers.chat_message_writer.ChatMessageWriter.to_dict"></a>

#### ChatMessageWriter.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_experimental.components.writers.chat_message_writer.ChatMessageWriter.from_dict"></a>

#### ChatMessageWriter.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "ChatMessageWriter"
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
def run(messages: List[ChatMessage]) -> Dict[str, int]
```

Run the ChatMessageWriter on the given input data.

**Arguments**:

- `messages`: A list of chat messages to write to the store.

**Raises**:

- `ValueError`: If the specified message store is not found.

**Returns**:

- `messages_written`: Number of messages written to the ChatMessageStore.

