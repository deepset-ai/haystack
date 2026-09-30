---
title: "Hetzner"
id: integrations-hetzner
description: "Hetzner integration for Haystack"
slug: "/integrations-hetzner"
---


## haystack_integrations.components.generators.hetzner.chat.chat_generator

### HetznerChatGenerator

Bases: <code>OpenAIChatGenerator</code>

Enables text generation using the models served by the Hetzner Inference API.

For the list of available models, see the
[Hetzner Inference API docs](https://docs.hetzner.com/general/company-and-policy/experiments/inference/) or query
the `/v1/models` endpoint of the API, whose response is definitive.

You can pass any text generation parameters valid for the Hetzner chat completion API directly to this component
using the `generation_kwargs` parameter in `__init__` or in the `run` method.

The served models accept images alongside text, so
[`ImageContent`](https://docs.haystack.deepset.ai/docs/imagecontent) parts can be included in the
[`ChatMessage`](https://docs.haystack.deepset.ai/docs/chatmessage)s passed to `run`.

Usage example:

```python
from haystack_integrations.components.generators.hetzner import HetznerChatGenerator
from haystack.dataclasses import ChatMessage

messages = [ChatMessage.from_user("What's Natural Language Processing?")]

client = HetznerChatGenerator()
response = client.run(messages)
print(response)

>>{'replies': [ChatMessage(_content='Natural Language Processing (NLP) is a branch of artificial intelligence
>>that focuses on enabling computers to understand, interpret, and generate human language in a way that is
>>meaningful and useful.', _role=<ChatRole.ASSISTANT: 'assistant'>, _name=None,
>>_meta={'model': 'Qwen/Qwen3.6-35B-A3B-FP8', 'index': 0, 'finish_reason': 'stop',
>>'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}
```

#### SUPPORTED_MODELS

```python
SUPPORTED_MODELS: list[str] = ['Qwen/Qwen3.6-35B-A3B-FP8', 'Qwen3.8-27B']
```

The models supported by this component while the Hetzner Inference API is in experimental status.
The selection changes over time: query the `/v1/models` endpoint of the API for the definitive list.
Models outside this list are not rejected and are passed on to the API as-is.

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("HETZNER_API_KEY"),
    model: str = "Qwen/Qwen3.6-35B-A3B-FP8",
    streaming_callback: StreamingCallbackT | None = None,
    api_base_url: str | None = "https://inference.hetzner.com/api/v1",
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates an instance of HetznerChatGenerator.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Hetzner Inference API token.
- **model** (<code>str</code>) – The name of the Hetzner chat completion model to use. See `SUPPORTED_MODELS`.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  The callback function accepts StreamingChunk as an argument.
- **api_base_url** (<code>str | None</code>) – The Hetzner Inference API base url.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Other parameters to use for the model. These parameters are all sent directly to
  the Hetzner endpoint.
  Some of the supported parameters:
- `max_tokens`: The maximum number of tokens the output text can have.
- `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
  Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
- `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
  considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
  comprising the top 10% probability mass are considered.
- `stream`: Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent
  events as they become available, with the stream terminated by a data: [DONE] message.
- `response_format`: A JSON schema or a Pydantic model that enforces the structure of the model's response.
  If provided, the output will always be validated against this
  format (unless the model returns a tool call).
  For details, see the [OpenAI Structured Outputs documentation](https://platform.openai.com/docs/guides/structured-outputs).
  Notes:
  - For structured outputs with streaming,
    the `response_format` must be a JSON schema and not a Pydantic model.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  Each tool should have a unique name.
- **timeout** (<code>float | None</code>) – The timeout for the Hetzner API call.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact Hetzner after an internal error.
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
