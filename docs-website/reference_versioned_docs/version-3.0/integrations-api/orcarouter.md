---
title: "OrcaRouter"
id: integrations-orcarouter
description: "OrcaRouter integration for Haystack"
slug: "/integrations-orcarouter"
---


## haystack_integrations.components.generators.orcarouter.chat.chat_generator

### OrcaRouterChatGenerator

Bases: <code>OpenAIChatGenerator</code>

Enables text generation using OrcaRouter generative models.

OrcaRouter is an OpenAI-compatible model routing gateway that exposes 100+ chat models from providers such as
OpenAI, Anthropic, Google, DeepSeek, and Qwen behind a single endpoint and API key. Models are addressed with a
`provider/model` namespace (for example `openai/gpt-4o-mini` or `anthropic/claude-opus-4.8`). The special
`orcarouter/auto` router selects an upstream model per request according to the routing policy configured in your
OrcaRouter console.

For the list of supported models, see the [OrcaRouter model catalog](https://www.orcarouter.ai/models).

This component supports streaming, tool-calling, and structured outputs.
It uses the ChatMessage format for both input and output; see the
[Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage) for details.

Usage example:

```python
from haystack_integrations.components.generators.orcarouter import OrcaRouterChatGenerator
from haystack.dataclasses import ChatMessage

messages = [ChatMessage.from_user("What's Natural Language Processing?")]

client = OrcaRouterChatGenerator(model="openai/gpt-4o-mini")
response = client.run(messages)
print(response)
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("ORCAROUTER_API_KEY"),
    model: str = "openai/gpt-4o-mini",
    streaming_callback: StreamingCallbackT | None = None,
    api_base_url: str | None = "https://api.orcarouter.ai/v1",
    organization: str | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None,
    tools_strict: bool = False,
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates an instance of OrcaRouterChatGenerator.

Unless specified otherwise, the default model is `openai/gpt-4o-mini`.

**Parameters:**

- **api_key** (<code>Secret</code>) – The OrcaRouter API key.
- **model** (<code>str</code>) – The name of the OrcaRouter chat completion model to use. Models use a `provider/model` namespace
  (for example `openai/gpt-4o-mini`). Use `orcarouter/auto` to let OrcaRouter route the request.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  The callback function accepts StreamingChunk as an argument.
- **api_base_url** (<code>str | None</code>) – The OrcaRouter API base URL. For more details, see the OrcaRouter
  [documentation](https://docs.orcarouter.ai).
- **organization** (<code>str | None</code>) – Your OrcaRouter organization ID, if any.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Other parameters to use for the model. These parameters are sent directly to the OrcaRouter endpoint.
  See OrcaRouter [API docs](https://docs.orcarouter.ai) for more details. Some of the supported parameters:
- `max_tokens`: The maximum number of tokens the output text can have.
- `temperature`: The sampling temperature to use. Higher values mean the model takes more risks.
- `top_p`: The nucleus sampling value to use.
- `stream`: Whether to stream back partial progress.
- `extra_body`: A dictionary of OrcaRouter-specific routing preferences (such as a fallback list of
  models) that is passed straight through to the gateway.
- **tools** (<code>ToolsType | None</code>) – A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
  list of `Tool` objects or a `Toolset` instance.
- **tools_strict** (<code>bool</code>) – Whether to enable strict schema adherence for tool calls. If set to `True`, the model follows exactly
  the schema provided in the `parameters` field of the tool definition.
- **timeout** (<code>float | None</code>) – The timeout for the OrcaRouter API call.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact OrcaRouter after an internal error.
  If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
