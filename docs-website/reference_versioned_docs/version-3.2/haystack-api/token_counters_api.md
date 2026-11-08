---
title: "Token Counters"
id: token-counters-api
description: "Estimate how many tokens a conversation occupies, for features that need a size before sending it to a model."
slug: "/token-counters-api"
---


## approximate_counter

### ApproximateTokenCounter

Bases: <code>TokenCounter</code>

Estimates tokens from text length using a flat ratio of characters to tokens.

## Usage Example:

```python
from haystack.dataclasses import ChatMessage
from haystack.token_counters import ApproximateTokenCounter

counter = ApproximateTokenCounter(chars_per_token=4.0)
messages = [
    ChatMessage.from_user("Hello, how are you?"),
    ChatMessage.from_assistant("I'm good, thank you! How can I assist you today?")
]
token_count = counter.count(messages)
print(f"Estimated token count: {token_count}")
```

#### __init__

```python
__init__(
    chars_per_token: float = 4.0,
    tokens_per_image: int = 85,
    tokens_per_file: int = 1000,
) -> None
```

Initialize the counter.

**Parameters:**

- **chars_per_token** (<code>float</code>) – How many characters to treat as one token.
- **tokens_per_image** (<code>int</code>) – Tokens to charge per image, which has no text to measure. The default is what
  OpenAI charges for a small image; raise it if you send large ones.
- **tokens_per_file** (<code>int</code>) – Tokens to charge per file. A rough stand-in for a short document, since the real
  cost depends on the page count; raise it if you send long ones.

**Raises:**

- <code>ValueError</code> – If `chars_per_token` is not positive.

#### count

```python
count(messages: list[ChatMessage], tools: ToolsType | None = None) -> int
```

Return the estimated number of tokens the given messages occupy.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – The messages to measure.
- **tools** (<code>ToolsType | None</code>) – Tools whose schemas are sent alongside the messages, and so consume tokens too.

**Returns:**

- <code>int</code> – The estimated token count, or `0` when there is nothing to measure.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the counter.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary representation of the counter.

## openai_counter

### OpenAITokenCounter

Bases: <code>TokenCounter</code>

Counts tokens with OpenAI's input token counting API.

Unlike local token counters, this counter sends the input to OpenAI's
`POST /v1/responses/input_tokens` endpoint. The returned count includes the model-specific formatting used for
messages and tool schemas, as well as supported non-text content such as images and files.

## Usage Example:

```python
from haystack.dataclasses import ChatMessage
from haystack.token_counters import OpenAITokenCounter

counter = OpenAITokenCounter("gpt-5-mini")
messages = [ChatMessage.from_user("Hello, how are you?")]
token_count = counter.count(messages)
print(f"Token count: {token_count}")
```

#### __init__

```python
__init__(
    model: str,
    *,
    api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
    api_base_url: str | None = None,
    organization: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Initialize the counter.

**Parameters:**

- **model** (<code>str</code>) – The model whose tokenization should be used.
- **api_key** (<code>Secret</code>) – The OpenAI API key. You can set it with the `OPENAI_API_KEY` environment variable or pass it
  explicitly.
- **api_base_url** (<code>str | None</code>) – An optional base URL for the OpenAI API.
- **organization** (<code>str | None</code>) – Your OpenAI organization ID.
- **timeout** (<code>float | None</code>) – Timeout for OpenAI client calls. If unset, uses `OPENAI_TIMEOUT` or 30 seconds.
- **max_retries** (<code>int | None</code>) – Maximum retries for OpenAI client calls. If unset, uses `OPENAI_MAX_RETRIES` or 5.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – Keyword arguments used to configure the underlying HTTPX client.

#### warm_up

```python
warm_up() -> None
```

Initialize the OpenAI client.

#### count

```python
count(messages: list[ChatMessage], tools: ToolsType | None = None) -> int
```

Return the exact number of input tokens OpenAI will use for the given messages and tools.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – The messages to measure.
- **tools** (<code>ToolsType | None</code>) – Tools whose schemas are sent alongside the messages, and so consume tokens too.

**Returns:**

- <code>int</code> – The token count, or `0` when there is nothing to measure.

#### close

```python
close() -> None
```

Close the OpenAI client and its underlying HTTP resources.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the counter.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary representation of the counter.

## tiktoken_counter

### TiktokenCounter

Bases: <code>TokenCounter</code>

Counts tokens locally with `tiktoken`, OpenAI's byte-pair encoder.

Counting is an estimate, and two limits are worth knowing before relying on it:

- **It is text-only**, so images and files get the flat `tokens_per_image` / `tokens_per_file` estimate rather
  than a real count.
- **It is OpenAI's encoder.** Other providers tokenize differently, so expect the count to drift on them.

## Usage Example:

```python
from haystack.dataclasses import ChatMessage
from haystack.token_counters import TiktokenCounter

counter = TiktokenCounter(encoding="o200k_base")
messages = [
    ChatMessage.from_user("Hello, how are you?"),
    ChatMessage.from_assistant("I'm good, thank you! How can I assist you today?")
]
token_count = counter.count(messages)
print(f"Token count: {token_count}")
```

#### __init__

```python
__init__(
    encoding: str = "o200k_base",
    tokens_per_image: int = 85,
    tokens_per_file: int = 1000,
) -> None
```

Initialize the counter.

**Parameters:**

- **encoding** (<code>str</code>) – The `tiktoken` encoding to count with. The default, `o200k_base`, is what current OpenAI
  models use.
- **tokens_per_image** (<code>int</code>) – Tokens to charge per image, which the tokenizer cannot measure. The default is what
  OpenAI charges for a small image; raise it if you send large ones.
- **tokens_per_file** (<code>int</code>) – Tokens to charge per file. A rough stand-in for a short document, since the real
  cost depends on the page count; raise it if you send long ones.

**Raises:**

- <code>ImportError</code> – If `tiktoken` is not installed.

#### warm_up

```python
warm_up() -> None
```

Load the encoder, downloading its vocabulary if it is not already cached.

#### count

```python
count(messages: list[ChatMessage], tools: ToolsType | None = None) -> int
```

Return the estimated number of tokens used by the given messages.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – The messages to measure.
- **tools** (<code>ToolsType | None</code>) – Tools whose schemas are sent alongside the messages, and so consume tokens too.

**Returns:**

- <code>int</code> – The estimated token count, or `0` when there is nothing to measure.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the counter.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary representation of the counter.

## types/protocol

### TokenCounter

Bases: <code>Protocol</code>

Estimates the number tokens used by a list of messages.

Implement `to_dict` so the counter's settings survive serialization. The default `from_dict` passes them straight
back to the constructor, which is enough for plain values; override it when `to_dict` emitted something that has to
be rebuilt first, such as a `Secret` or a nested component.

#### count

```python
count(messages: list[ChatMessage], tools: ToolsType | None = None) -> int
```

Return the estimated number of tokens in the given messages.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – The messages to measure.
- **tools** (<code>ToolsType | None</code>) – Tools whose schemas are sent alongside the messages, and so consume tokens too. Pass them to have
  them counted; leave as None to measure the messages alone.

**Returns:**

- <code>int</code> – The estimated token count.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the counter to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> TokenCounter
```

Deserialize the counter from a dictionary.
