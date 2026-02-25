---
title: "Comet API"
id: integrations-cometapi
description: "Comet API integration for Haystack"
slug: "/integrations-cometapi"
---

<a id="haystack_integrations.components.generators.cometapi.chat.chat_generator"></a>

## Module haystack\_integrations.components.generators.cometapi.chat.chat\_generator

<a id="haystack_integrations.components.generators.cometapi.chat.chat_generator.CometAPIChatGenerator"></a>

### CometAPIChatGenerator

A chat generator that uses the CometAPI for generating chat responses.

This class extends Haystack's OpenAIChatGenerator to specifically interact with the CometAPI.
It sets the `api_base_url` to the CometAPI endpoint and allows for all the
standard configurations available in the OpenAIChatGenerator.

**Arguments**:

- `api_key`: The API key for authenticating with the CometAPI. Defaults to
loading from the "COMET_API_KEY" environment variable.
- `model`: The name of the model to use for chat generation (e.g., "gpt-5-mini", "grok-3-mini").
Defaults to "gpt-5-mini".
- `streaming_callback`: An optional callable that will be called with each chunk of
a streaming response.
- `generation_kwargs`: Optional keyword arguments to pass to the underlying generation
API call.
- `timeout`: The maximum time in seconds to wait for a response from the API.
- `max_retries`: The maximum number of times to retry a failed API request.
- `tools`: An optional list of tool definitions that the model can use.
- `tools_strict`: If True, the model is forced to use one of the provided tools if a tool call is made.
- `http_client_kwargs`: Optional keyword arguments to pass to the HTTP client.

