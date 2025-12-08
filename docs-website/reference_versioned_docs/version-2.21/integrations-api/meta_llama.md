---
title: "Meta Llama API"
id: integrations-meta-llama
description: "Meta Llama API integration for Haystack"
slug: "/integrations-meta-llama"
---

<a id="haystack_integrations.components.generators.meta_llama.chat.chat_generator"></a>

## Module haystack\_integrations.components.generators.meta\_llama.chat.chat\_generator

<a id="haystack_integrations.components.generators.meta_llama.chat.chat_generator.MetaLlamaChatGenerator"></a>

### MetaLlamaChatGenerator

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
             tools: Optional[ToolsType] = None)
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
- `response_format`: A JSON schema or a Pydantic model that enforces the structure of the model's response.
    If provided, the output will always be validated against this
    format (unless the model returns a tool call).
    For details, see the [OpenAI Structured Outputs documentation](https://platform.openai.com/docs/guides/structured-outputs).
    For structured outputs with streaming, the `response_format` must be a JSON
    schema and not a Pydantic model.
- `tools`: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
Each tool should have a unique name.

<a id="haystack_integrations.components.generators.meta_llama.chat.chat_generator.MetaLlamaChatGenerator.to_dict"></a>

#### MetaLlamaChatGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

