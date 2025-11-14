---
title: "Llama Stack"
id: integrations-llama-stack
description: "Llama Stack integration for Haystack"
slug: "/integrations-llama-stack"
---

<a id="haystack_integrations.components.generators.llama_stack.chat.chat_generator"></a>

## Module haystack\_integrations.components.generators.llama\_stack.chat.chat\_generator

<a id="haystack_integrations.components.generators.llama_stack.chat.chat_generator.LlamaStackChatGenerator"></a>

### LlamaStackChatGenerator

Enables text generation using Llama Stack framework.
Llama Stack Server supports multiple inference providers, including Ollama, Together,
and vLLM and other cloud providers.
For a complete list of inference providers, see [Llama Stack docs](https://llama-stack.readthedocs.io/en/latest/providers/inference/index.html).

Users can pass any text generation parameters valid for the OpenAI chat completion API
directly to this component using the `generation_kwargs`
parameter in `__init__` or the `generation_kwargs` parameter in `run` method.

This component uses the `ChatMessage` format for structuring both input and output,
ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
Details on the `ChatMessage` format can be found in the
[Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

Usage example:
You need to setup Llama Stack Server before running this example and have a model available. For a quick start on
how to setup server with Ollama, see [Llama Stack docs](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html).

```python
from haystack_integrations.components.generators.llama_stack import LlamaStackChatGenerator
from haystack.dataclasses import ChatMessage

messages = [ChatMessage.from_user("What's Natural Language Processing?")]

client = LlamaStackChatGenerator(model="ollama/llama3.2:3b")
response = client.run(messages)
print(response)

>>{'replies': [ChatMessage(_content=[TextContent(text='Natural Language Processing (NLP)
is a branch of artificial intelligence
>>that focuses on enabling computers to understand, interpret, and generate human language in a way that is
>>meaningful and useful.')], _role=<ChatRole.ASSISTANT: 'assistant'>, _name=None,
>>_meta={'model': 'ollama/llama3.2:3b', 'index': 0, 'finish_reason': 'stop',
>>'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}

<a id="haystack_integrations.components.generators.llama_stack.chat.chat_generator.LlamaStackChatGenerator.__init__"></a>

#### LlamaStackChatGenerator.\_\_init\_\_

```python
def __init__(*,
             model: str,
             api_base_url: str = "http://localhost:8321/v1/openai/v1",
             organization: Optional[str] = None,
             streaming_callback: Optional[StreamingCallbackT] = None,
             generation_kwargs: Optional[Dict[str, Any]] = None,
             timeout: Optional[int] = None,
             tools: Optional[ToolsType] = None,
             tools_strict: bool = False,
             max_retries: Optional[int] = None,
             http_client_kwargs: Optional[Dict[str, Any]] = None)
```

Creates an instance of LlamaStackChatGenerator. To use this chat generator,

you need to setup Llama Stack Server with an inference provider and have a model available.

**Arguments**:

- `model`: The name of the model to use for chat completion.
This depends on the inference provider used for the Llama Stack Server.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts StreamingChunk as an argument.
- `api_base_url`: The Llama Stack API base url. If not specified, the localhost is used with the default port 8321.
- `organization`: Your organization ID, defaults to `None`.
- `generation_kwargs`: Other parameters to use for the model. These parameters are all sent directly to
the Llama Stack endpoint. See [Llama Stack API docs](https://llama-stack.readthedocs.io/) for more details.
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
- `timeout`: Timeout for client calls using OpenAI API. If not set, it defaults to either the
`OPENAI_TIMEOUT` environment variable, or 30 seconds.
- `tools`: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
Each tool should have a unique name.
- `tools_strict`: Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
the schema provided in the `parameters` field of the tool definition, but this may increase latency.
- `max_retries`: Maximum number of retries to contact OpenAI after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="haystack_integrations.components.generators.llama_stack.chat.chat_generator.LlamaStackChatGenerator.to_dict"></a>

#### LlamaStackChatGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.generators.llama_stack.chat.chat_generator.LlamaStackChatGenerator.from_dict"></a>

#### LlamaStackChatGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "LlamaStackChatGenerator"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

