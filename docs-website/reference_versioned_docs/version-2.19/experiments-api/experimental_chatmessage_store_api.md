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

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.__init__"></a>

#### InMemoryChatMessageStore.\_\_init\_\_

```python
def __init__()
```

Initializes the InMemoryChatMessageStore.

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.to_dict"></a>

#### InMemoryChatMessageStore.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.from_dict"></a>

#### InMemoryChatMessageStore.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "InMemoryChatMessageStore"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.count_messages"></a>

#### InMemoryChatMessageStore.count\_messages

```python
def count_messages() -> int
```

Returns the number of chat messages stored.

**Returns**:

The number of messages.

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.write_messages"></a>

#### InMemoryChatMessageStore.write\_messages

```python
def write_messages(messages: List[ChatMessage]) -> int
```

Writes chat messages to the ChatMessageStore.

**Arguments**:

- `messages`: A list of ChatMessages to write.

**Raises**:

- `ValueError`: If messages is not a list of ChatMessages.

**Returns**:

The number of messages written.

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.delete_messages"></a>

#### InMemoryChatMessageStore.delete\_messages

```python
def delete_messages() -> None
```

Deletes all stored chat messages.

<a id="haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore.retrieve"></a>

#### InMemoryChatMessageStore.retrieve

```python
def retrieve() -> List[ChatMessage]
```

Retrieves all stored chat messages.

**Returns**:

A list of chat messages.
