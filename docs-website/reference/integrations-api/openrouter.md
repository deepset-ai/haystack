---
title: "OpenRouter"
id: integrations-openrouter
description: "OpenRouter integration for Haystack"
slug: "/integrations-openrouter"
---

<a id="haystack_integrations.components.generators.openrouter.chat.chat_generator"></a>

# Module haystack\_integrations.components.generators.openrouter.chat.chat\_generator

<a id="haystack_integrations.components.generators.openrouter.chat.chat_generator.OpenRouterChatGenerator"></a>

## OpenRouterChatGenerator

Enables text generation using OpenRouter generative models.
For supported models, see [OpenRouter docs](https://openrouter.ai/models).

Users can pass any text generation parameters valid for the OpenRouter chat completion API
directly to this component using the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
parameter in `run` method.

Key Features and Compatibility:
- **Primary Compatibility**: Designed to work seamlessly with the OpenRouter chat completion endpoint.
- **Streaming Support**: Supports streaming responses from the OpenRouter chat completion endpoint.
- **Customizability**: Supports all parameters supported by the OpenRouter chat completion endpoint.

This component uses the ChatMessage format for structuring both input and output,
ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
Details on the ChatMessage format can be found in the
[Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

For more details on the parameters supported by the OpenRouter API, refer to the
[OpenRouter API Docs](https://openrouter.ai/docs/quickstart).

Usage example:
```python
from haystack_integrations.components.generators.openrouter import OpenRouterChatGenerator
from haystack.dataclasses import ChatMessage

messages = [ChatMessage.from_user("What's Natural Language Processing?")]

client = OpenRouterChatGenerator()
response = client.run(messages)
print(response)

>>{'replies': [ChatMessage(_content='Natural Language Processing (NLP) is a branch of artificial intelligence
>>that focuses on enabling computers to understand, interpret, and generate human language in a way that is
>>meaningful and useful.', _role=<ChatRole.ASSISTANT: 'assistant'>, _name=None,
>>_meta={'model': 'openai/gpt-4o-mini', 'index': 0, 'finish_reason': 'stop',
>>'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}
```

<a id="haystack_integrations.components.generators.openrouter.chat.chat_generator.OpenRouterChatGenerator.__init__"></a>

#### OpenRouterChatGenerator.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var("OPENROUTER_API_KEY"),
             model: str = "openai/gpt-4o-mini",
             streaming_callback: Optional[StreamingCallbackT] = None,
             api_base_url: Optional[str] = "https://openrouter.ai/api/v1",
             generation_kwargs: Optional[Dict[str, Any]] = None,
             tools: Optional[ToolsType] = None,
             timeout: Optional[float] = None,
             extra_headers: Optional[Dict[str, Any]] = None,
             max_retries: Optional[int] = None,
             http_client_kwargs: Optional[Dict[str, Any]] = None)
```

Creates an instance of OpenRouterChatGenerator. Unless specified otherwise,

the default model is `openai/gpt-4o-mini`.

**Arguments**:

- `api_key`: The OpenRouter API key.
- `model`: The name of the OpenRouter chat completion model to use.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts StreamingChunk as an argument.
- `api_base_url`: The OpenRouter API Base url.
For more details, see OpenRouter [docs](https://openrouter.ai/docs/quickstart).
- `generation_kwargs`: Other parameters to use for the model. These parameters are all sent directly to
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
- `response_format`: A JSON schema or a Pydantic model that enforces the structure of the model's response.
    If provided, the output will always be validated against this
    format (unless the model returns a tool call).
    For details, see the [OpenAI Structured Outputs documentation](https://platform.openai.com/docs/guides/structured-outputs).
    Notes:
    - This parameter accepts Pydantic models and JSON schemas for latest models starting from GPT-4o.
    - For structured outputs with streaming,
      the `response_format` must be a JSON schema and not a Pydantic model.
- `tools`: A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
list of `Tool` objects or a `Toolset` instance.
- `timeout`: The timeout for the OpenRouter API call.
- `extra_headers`: Additional HTTP headers to include in requests to the OpenRouter API.
This can be useful for adding site URL or title for rankings on openrouter.ai
For more details, see OpenRouter [docs](https://openrouter.ai/docs/quickstart).
- `max_retries`: Maximum number of retries to contact OpenAI after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="haystack_integrations.components.generators.openrouter.chat.chat_generator.OpenRouterChatGenerator.to_dict"></a>

#### OpenRouterChatGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.generators.openrouter.chat.chat_generator.OpenRouterChatGenerator.from_dict"></a>

#### OpenRouterChatGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenAIChatGenerator"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="haystack_integrations.components.generators.openrouter.chat.chat_generator.OpenRouterChatGenerator.run"></a>

#### OpenRouterChatGenerator.run

```python
@component.output_types(replies=list[ChatMessage])
def run(messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        *,
        tools: Optional[ToolsType] = None,
        tools_strict: Optional[bool] = None)
```

Invokes chat completion based on the provided messages and generation parameters.

**Arguments**:

- `messages`: A list of ChatMessage instances representing the input messages.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
- `generation_kwargs`: Additional keyword arguments for text generation. These parameters will
override the parameters passed during component initialization.
For details on OpenAI API parameters, see [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
- `tools`: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
If set, it will override the `tools` parameter provided during initialization.
- `tools_strict`: Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
the schema provided in the `parameters` field of the tool definition, but this may increase latency.
If set, it will override the `tools_strict` parameter set during component initialization.

**Returns**:

A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances.

<a id="haystack_integrations.components.generators.openrouter.chat.chat_generator.OpenRouterChatGenerator.run_async"></a>

#### OpenRouterChatGenerator.run\_async

```python
@component.output_types(replies=list[ChatMessage])
async def run_async(messages: list[ChatMessage],
                    streaming_callback: Optional[StreamingCallbackT] = None,
                    generation_kwargs: Optional[dict[str, Any]] = None,
                    *,
                    tools: Optional[ToolsType] = None,
                    tools_strict: Optional[bool] = None)
```

Asynchronously invokes chat completion based on the provided messages and generation parameters.

This is the asynchronous version of the `run` method. It has the same parameters and return values
but can be used with `await` in async code.

**Arguments**:

- `messages`: A list of ChatMessage instances representing the input messages.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
Must be a coroutine.
- `generation_kwargs`: Additional keyword arguments for text generation. These parameters will
override the parameters passed during component initialization.
For details on OpenAI API parameters, see [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
- `tools`: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
If set, it will override the `tools` parameter provided during initialization.
- `tools_strict`: Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
the schema provided in the `parameters` field of the tool definition, but this may increase latency.
If set, it will override the `tools_strict` parameter set during component initialization.

**Returns**:

A dictionary with the following key:
- `replies`: A list containing the generated responses as ChatMessage instances.
