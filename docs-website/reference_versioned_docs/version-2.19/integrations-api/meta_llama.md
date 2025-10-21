---
title: "Meta Llama API"
id: integrations-meta-llama
description: "Meta Llama API integration for Haystack"
slug: "/integrations-meta-llama"
---

<a id="haystack_integrations.components.generators.meta_llama.chat.chat_generator"></a>

# Module haystack\_integrations.components.generators.meta\_llama.chat.chat\_generator

<a id="haystack_integrations.components.generators.meta_llama.chat.chat_generator.MetaLlamaChatGenerator"></a>

## MetaLlamaChatGenerator

Enables text generation using Llama generative models.
For supported models, see [Llama API Docs](https://llama.developer.meta.com/docs/).

Users can pass any text generation parameters valid for the Llama Chat Completion API
directly to this component via the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
parameter in `run` method.

Key Features and Compatibility:
- **Primary Compatibility**: Designed to work seamlessly with the Llama API Chat Completion endpoint.
- **Streaming Support**: Supports streaming responses from the Llama API Chat Completion endpoint.
- **Customizability**: Supports parameters supported by the Llama API Chat Completion endpoint.
- **Response Format**: Currently only supports json_schema response format.

This component uses the ChatMessage format for structuring both input and output,
ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
Details on the ChatMessage format can be found in the
[Haystack docs](https://docs.haystack.deepset.ai/docs/data-classes#chatmessage)

For more details on the parameters supported by the Llama API, refer to the
[Llama API Docs](https://llama.developer.meta.com/docs/).

Usage example:
```python
from haystack_integrations.components.generators.llama import LlamaChatGenerator
from haystack.dataclasses import ChatMessage

messages = [ChatMessage.from_user("What's Natural Language Processing?")]

client = LlamaChatGenerator()
response = client.run(messages)
print(response)
```

<a id="haystack_integrations.components.generators.meta_llama.chat.chat_generator.MetaLlamaChatGenerator.__init__"></a>

#### MetaLlamaChatGenerator.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var("LLAMA_API_KEY"),
             model: str = "Llama-4-Scout-17B-16E-Instruct-FP8",
             streaming_callback: Optional[StreamingCallbackT] = None,
             api_base_url: Optional[str] = "https://api.llama.com/compat/v1/",
             generation_kwargs: Optional[Dict[str, Any]] = None,
             tools: Optional[Union[List[Tool], Toolset]] = None)
```

Creates an instance of LlamaChatGenerator. Unless specified otherwise in the `model`, this is for Llama's

`Llama-4-Scout-17B-16E-Instruct-FP8` model.

**Arguments**:

- `api_key`: The Llama API key.
- `model`: The name of the Llama chat completion model to use.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts StreamingChunk as an argument.
- `api_base_url`: The Llama API Base url.
For more details, see LlamaAPI [docs](https://llama.developer.meta.com/docs/features/compatibility/).
- `generation_kwargs`: Other parameters to use for the model. These parameters are all sent directly to
the Llama API endpoint. See [Llama API docs](https://llama.developer.meta.com/docs/features/compatibility/)
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
- `tools`: A list of tools for which the model can prepare calls.

<a id="haystack_integrations.components.generators.meta_llama.chat.chat_generator.MetaLlamaChatGenerator.to_dict"></a>

#### MetaLlamaChatGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.generators.meta_llama.chat.chat_generator.MetaLlamaChatGenerator.from_dict"></a>

#### MetaLlamaChatGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenAIChatGenerator"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="haystack_integrations.components.generators.meta_llama.chat.chat_generator.MetaLlamaChatGenerator.run"></a>

#### MetaLlamaChatGenerator.run

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

<a id="haystack_integrations.components.generators.meta_llama.chat.chat_generator.MetaLlamaChatGenerator.run_async"></a>

#### MetaLlamaChatGenerator.run\_async

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
