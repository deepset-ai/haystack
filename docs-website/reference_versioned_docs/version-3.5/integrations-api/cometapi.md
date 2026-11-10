---
title: "Comet API"
id: integrations-cometapi
description: "Comet API integration for Haystack"
slug: "/integrations-cometapi"
---


## haystack_integrations.components.generators.cometapi.chat.chat_generator

### CometAPIChatGenerator

Bases: <code>OpenAIChatGenerator</code>

A chat generator that uses the CometAPI for generating chat responses.

This class extends Haystack's OpenAIChatGenerator to specifically interact with the CometAPI.
It sets the `api_base_url` to the CometAPI endpoint and allows for all the
standard configurations available in the OpenAIChatGenerator.

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("COMET_API_KEY"),
    model: str = "gpt-5-mini",
    streaming_callback: StreamingCallbackT | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    tools: list[Tool | Toolset] | Toolset | None = None,
    tools_strict: bool = False,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates a `CometAPIChatGenerator` instance.

**Parameters:**

- **api_key** (<code>Secret</code>) – The API key for authenticating with the CometAPI.
- **model** (<code>str</code>) – The name of the model to use for chat generation (e.g., `"gpt-5-mini"`, `"grok-3-mini"`).
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – An optional callable invoked with each chunk of a streaming response.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional keyword arguments passed to the underlying generation API call.
- **timeout** (<code>int | None</code>) – The maximum time in seconds to wait for a response from the API.
- **max_retries** (<code>int | None</code>) – The maximum number of times to retry a failed API request.
- **tools** (<code>list\[Tool | Toolset\] | Toolset | None</code>) – An optional list of tools the model can use.
- **tools_strict** (<code>bool</code>) – If `True`, the model is forced to use one of the provided tools.
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional keyword arguments passed to the HTTP client.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.
