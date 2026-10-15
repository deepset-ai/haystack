---
title: "Together AI"
id: integrations-togetherai
description: "Together AI integration for Haystack"
slug: "/integrations-togetherai"
---


## haystack_integrations.components.generators.togetherai.chat.chat_generator

### TogetherAIChatGenerator

Bases: <code>OpenAIChatGenerator</code>

Enables text generation using Together AI generative models.

For supported models, see [Together AI docs](https://docs.together.ai/docs).

Users can pass any text generation parameters valid for the Together AI chat completion API
directly to this component using the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
parameter in `run` method.

Key Features and Compatibility:

- **Primary Compatibility**: Designed to work seamlessly with the Together AI chat completion endpoint.
- **Streaming Support**: Supports streaming responses from the Together AI chat completion endpoint.
- **Customizability**: Supports all parameters supported by the Together AI chat completion endpoint.

This component uses the ChatMessage format for structuring both input and output,
ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
Details on the ChatMessage format can be found in the
[Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

For more details on the parameters supported by the Together AI API, refer to the
[Together AI API Docs](https://docs.together.ai/reference/chat-completions-1).

Usage example:

```python
from haystack_integrations.components.generators.togetherai import TogetherAIChatGenerator
from haystack.dataclasses import ChatMessage

messages = [ChatMessage.from_user("What's Natural Language Processing?")]

client = TogetherAIChatGenerator()
response = client.run(messages)
print(response)

>>{'replies': [ChatMessage(_content='Natural Language Processing (NLP) is a branch of artificial intelligence
>>that focuses on enabling computers to understand, interpret, and generate human language in a way that is
>>meaningful and useful.', _role=<ChatRole.ASSISTANT: 'assistant'>, _name=None,
>>_meta={'model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo', 'index': 0, 'finish_reason': 'stop',
>>'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("TOGETHER_API_KEY"),
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    streaming_callback: StreamingCallbackT | None = None,
    api_base_url: str | None = "https://api.together.xyz/v1",
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Creates an instance of TogetherAIChatGenerator.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Together API key.
- **model** (<code>str</code>) – The name of the Together AI chat completion model to use.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function that is called when a new token is received from the stream.
  The callback function accepts StreamingChunk as an argument.
- **api_base_url** (<code>str | None</code>) – The Together AI API Base url.
  For more details, see Together AI [docs](https://docs.together.ai/docs/openai-api-compatibility).
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Other parameters to use for the model. These parameters are all sent directly to
  the Together AI endpoint. See [Together AI API docs](https://docs.together.ai/reference/chat-completions-1)
  for more details.
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
- `response_format`: A JSON schema or a Pydantic model that enforces the structure of the model's response.
  If provided, the output will always be validated against this
  format (unless the model returns a tool call).
  For details, see the [OpenAI Structured Outputs documentation](https://platform.openai.com/docs/guides/structured-outputs).
  Notes:
  - For structured outputs with streaming,
    the `response_format` must be a JSON schema and not a Pydantic model.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
  Each tool should have a unique name.
- **timeout** (<code>float | None</code>) – The timeout for the Together AI API call.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact Together AI after an internal error.
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
