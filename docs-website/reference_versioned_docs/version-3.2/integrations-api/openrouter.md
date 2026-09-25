---
title: "OpenRouter"
id: integrations-openrouter
description: "OpenRouter integration for Haystack"
slug: "/integrations-openrouter"
---


## haystack_integrations.components.generators.openrouter.chat.chat_generator

### OpenRouterChatGenerator

Bases: <code>OpenAIChatGenerator</code>

Enables text generation using OpenRouter generative models.

For supported models, see [OpenRouter docs](https://openrouter.ai/models).

Users can pass any text generation parameters valid for the OpenRouter chat completion API
directly to this component using the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
parameter in `run` method.

Key Features and Compatibility:

- **Primary Compatibility**: Compatible with the OpenRouter chat completion endpoint.
- **Streaming Support**: Supports streaming responses from the OpenRouter chat completion endpoint.
- **Customizability**: Supports all parameters supported by the OpenRouter chat completion endpoint.
- **Reasoning Support**: Extracts reasoning/thinking content from models that support it
  (e.g., DeepSeek R1, Claude with extended thinking) and stores it in the `ReasoningContent`
  field on `ChatMessage`. Reasoning content is only captured for non-streaming requests.

This component uses the ChatMessage format for structuring both input and output,
ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
Details on the ChatMessage format can be found in the
[Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

For more details on the parameters supported by the OpenRouter API, refer to the
[OpenRouter API Docs](https://openrouter.ai/docs/quickstart).

Usage example:

```python
from haystack_integrations.components.generators.openrouter import (
    OpenRouterChatGenerator,
)
from haystack.dataclasses import ChatMessage

messages = [ChatMessage.from_user("What's Natural Language Processing?")]

client = OpenRouterChatGenerator(
    model="deepseek/deepseek-r1",
    generation_kwargs={"reasoning": {"effort": "high"}},
)
response = client.run(messages)
print(response["replies"][0].reasoning)  # Access reasoning content
print(response["replies"][0].text)  # Access final answer
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("OPENROUTER_API_KEY"),
    model: str = "openai/gpt-5-mini",
    streaming_callback: StreamingCallbackT | None = None,
    api_base_url: str | None = "https://openrouter.ai/api/v1",
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None,
    timeout: float | None = None,
    extra_headers: dict[str, Any] | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates an instance of OpenRouterChatGenerator.

**Parameters:**

- **api_key** (<code>Secret</code>) – The OpenRouter API key.
- **model** (<code>str</code>) – The name of the OpenRouter chat completion model to use.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  The callback function accepts StreamingChunk as an argument.
- **api_base_url** (<code>str | None</code>) – The OpenRouter API Base url.
  For more details, see OpenRouter [docs](https://openrouter.ai/docs/quickstart).
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Other parameters to use for the model. These parameters are all sent directly to
  the OpenRouter endpoint. See [OpenRouter API docs](https://openrouter.ai/docs/quickstart) for more details.
  Some of the supported parameters:
- `max_tokens`: The maximum number of tokens the output text can have.
- `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
  Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
- `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
  considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
  comprising the top 10% probability mass are considered.
- `stream`: Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent
  events as they become available, with the stream terminated by a data: [DONE] message.
- `safe_prompt`: Whether to inject a safety prompt before all conversations.
- `random_seed`: The seed to use for random sampling.
- `reasoning`: A dict to configure reasoning/thinking tokens for models that support it.
  Example: `{"effort": "high"}` or `{"max_tokens": 2000}`.
  Reasoning content is only captured for non-streaming requests.
  See [OpenRouter reasoning docs](https://openrouter.ai/docs/use-cases/reasoning-tokens).
- `response_format`: A JSON schema or a Pydantic model that enforces the structure of the model's response.
- **tools** (<code>ToolsType | None</code>) – A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
  list of `Tool` objects or a `Toolset` instance.
- **timeout** (<code>float | None</code>) – The timeout for the OpenRouter API call.
- **extra_headers** (<code>dict\[str, Any\] | None</code>) – Additional HTTP headers to include in requests to the OpenRouter API.
  This can be useful for adding site URL or title for rankings on openrouter.ai
  For more details, see OpenRouter [docs](https://openrouter.ai/docs/quickstart).
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact OpenAI after an internal error.
  If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

#### run

```python
run(
    messages: list[ChatMessage] | str,
    streaming_callback: StreamingCallbackT | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    *,
    tools: ToolsType | None = None,
    tools_strict: bool | None = None
) -> dict[str, list[ChatMessage]]
```

Invokes chat completion on the OpenRouter API.

**Parameters:**

- **messages** (<code>list\[ChatMessage\] | str</code>) – A list of ChatMessage instances representing the input messages.
  If a string is provided, it is converted to a list containing a ChatMessage with user role.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for text generation. These parameters will
  override the parameters passed during component initialization.
  For details on OpenRouter API parameters, see
  [OpenRouter docs](https://openrouter.ai/docs/quickstart).
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  If set, it will override the `tools` parameter provided during initialization.
- **tools_strict** (<code>bool | None</code>) – Whether to enable strict schema adherence for tool calls.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances.

#### run_async

```python
run_async(
    messages: list[ChatMessage] | str,
    streaming_callback: StreamingCallbackT | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    *,
    tools: ToolsType | None = None,
    tools_strict: bool | None = None
) -> dict[str, list[ChatMessage]]
```

Asynchronously invokes chat completion on the OpenRouter API.

**Parameters:**

- **messages** (<code>list\[ChatMessage\] | str</code>) – A list of ChatMessage instances representing the input messages.
  If a string is provided, it is converted to a list containing a ChatMessage with user role.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  Must be a coroutine.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for text generation.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset.
- **tools_strict** (<code>bool | None</code>) – Whether to enable strict schema adherence for tool calls.

**Returns:**

- <code>dict\[str, list\[ChatMessage\]\]</code> – A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances.
