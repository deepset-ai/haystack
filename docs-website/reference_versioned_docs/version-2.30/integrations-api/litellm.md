---
title: "LiteLLM"
id: integrations-litellm
description: "LiteLLM integration for Haystack"
slug: "/integrations-litellm"
---


## haystack_integrations.components.generators.litellm.chat.chat_generator

### LiteLLMChatGenerator

Completes chats using any of 100+ LLM providers via LiteLLM.

LiteLLM routes to OpenAI, Anthropic, Google, AWS Bedrock, Azure, Cohere,
Mistral, Groq, and many more through a single unified interface.

Model names use LiteLLM format: `provider/model-name`, e.g.
`anthropic/claude-sonnet-4-20250514`, `openai/gpt-4o`,
`bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0`.

See https://docs.litellm.ai/docs/providers for the full list.

Usage example:

```python
from haystack_integrations.components.generators.litellm import LiteLLMChatGenerator
from haystack.dataclasses import ChatMessage

generator = LiteLLMChatGenerator(
    model="anthropic/claude-sonnet-4-20250514",
    generation_kwargs={"max_tokens": 1024, "temperature": 0.7},
)

messages = [
    ChatMessage.from_system("You are a helpful assistant"),
    ChatMessage.from_user("What's Natural Language Processing?"),
]
result = generator.run(messages=messages)
print(result["replies"][0].text)
```

#### __init__

```python
__init__(
    *,
    api_key: Secret | None = None,
    model: str = "openai/gpt-4o",
    streaming_callback: StreamingCallbackT | None = None,
    api_base_url: str | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None
) -> None
```

Create a LiteLLMChatGenerator instance.

**Parameters:**

- **api_key** (<code>Secret | None</code>) – The API key for the provider. Optional: when not set, LiteLLM resolves
  credentials itself from the provider's standard environment variable
  (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). Pass a `Secret` only
  when you want Haystack to manage and serialize the key explicitly.
- **model** (<code>str</code>) – The model name in LiteLLM format (provider/model-name).
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function invoked with each new StreamingChunk.
- **api_base_url** (<code>str | None</code>) – Custom API base URL (e.g. for a self-hosted LiteLLM proxy).
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional parameters passed to litellm.completion().
  See https://docs.litellm.ai/docs/completion/input for details.
- **tools** (<code>ToolsType | None</code>) – A list of Tool / Toolset objects the model can prepare calls for.

#### run

```python
run(
    messages: list[ChatMessage],
    streaming_callback: StreamingCallbackT | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    *,
    tools: ToolsType | None = None
) -> dict[str, list[ChatMessage]]
```

Invoke chat completion via LiteLLM.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – Input messages as ChatMessage instances.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – Override the streaming callback for this call.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Override generation parameters for this call.
- **tools** (<code>ToolsType | None</code>) – Override tools for this call.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dict with key `replies` containing ChatMessage instances.

#### run_async

```python
run_async(
    messages: list[ChatMessage],
    streaming_callback: StreamingCallbackT | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    *,
    tools: ToolsType | None = None
) -> dict[str, list[ChatMessage]]
```

Async version of run().

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> LiteLLMChatGenerator
```

Deserialize a component from a dictionary.
