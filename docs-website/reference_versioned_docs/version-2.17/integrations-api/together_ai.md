---
title: "Together AI"
id: integrations-together-ai
description: "Together AI integration for Haystack"
slug: "/integrations-together-ai"
---

<a id="haystack_integrations.components.generators.together_ai.generator"></a>

## Module haystack\_integrations.components.generators.together\_ai.generator

<a id="haystack_integrations.components.generators.together_ai.generator.TogetherAIGenerator"></a>

### TogetherAIGenerator

Provides an interface to generate text using an LLM running on Together AI.

Usage example:
```python
from haystack_integrations.components.generators.together_ai import TogetherAIGenerator

generator = TogetherAIGenerator(model="deepseek-ai/DeepSeek-R1",
                            generation_kwargs={
                            "temperature": 0.9,
                            })

print(generator.run("Who is the best Italian actor?"))
```

<a id="haystack_integrations.components.generators.together_ai.generator.TogetherAIGenerator.__init__"></a>

#### TogetherAIGenerator.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("TOGETHER_API_KEY"),
             model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
             api_base_url: Optional[str] = "https://api.together.xyz/v1",
             streaming_callback: Optional[StreamingCallbackT] = None,
             system_prompt: Optional[str] = None,
             generation_kwargs: Optional[Dict[str, Any]] = None,
             timeout: Optional[float] = None,
             max_retries: Optional[int] = None)
```

Initialize the TogetherAIGenerator.

**Arguments**:

- `api_key`: The Together API key.
- `model`: The name of the model to use.
- `api_base_url`: The base URL of the Together AI API.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts StreamingChunk as an argument.
- `system_prompt`: The system prompt to use for text generation. If not provided, the system prompt is
omitted, and the default system prompt of the model is used.
- `generation_kwargs`: Other parameters to use for the model. These parameters are all sent directly to
the Together AI endpoint. See Together AI
[documentation](https://docs.together.ai/reference/chat-completions-1) for more details.
Some of the supported parameters:
- `max_tokens`: The maximum number of tokens the output text can have.
- `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
    Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
- `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
    considers the results of the tokens with top_p probability mass. So, 0.1 means only the tokens
    comprising the top 10% probability mass are considered.
- `n`: How many completions to generate for each prompt. For example, if the LLM gets 3 prompts and n is 2,
    it will generate two completions for each of the three prompts, ending up with 6 completions in total.
- `stop`: One or more sequences after which the LLM should stop generating tokens.
- `presence_penalty`: What penalty to apply if a token is already present at all. Bigger values mean
    the model will be less likely to repeat the same token in the text.
- `frequency_penalty`: What penalty to apply if a token has already been generated in the text.
    Bigger values mean the model will be less likely to repeat the same token in the text.
- `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens, and the
    values are the bias to add to that token.
- `timeout`: Timeout for together.ai Client calls, if not set it is inferred from the `OPENAI_TIMEOUT` environment
variable or set to 30.
- `max_retries`: Maximum retries to establish contact with Together AI if it returns an internal error, if not set it is
inferred from the `OPENAI_MAX_RETRIES` environment variable or set to 5.

<a id="haystack_integrations.components.generators.together_ai.generator.TogetherAIGenerator.to_dict"></a>

#### TogetherAIGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.generators.together_ai.generator.TogetherAIGenerator.from_dict"></a>

#### TogetherAIGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "TogetherAIGenerator"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="haystack_integrations.components.generators.together_ai.generator.TogetherAIGenerator.run"></a>

#### TogetherAIGenerator.run

```python
@component.output_types(replies=list[str], meta=list[dict[str, Any]])
def run(*,
        prompt: str,
        system_prompt: Optional[str] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[dict[str, Any]] = None) -> dict[str, Any]
```

Generate text completions synchronously.

**Arguments**:

- `prompt`: The input prompt string for text generation.
- `system_prompt`: An optional system prompt to provide context or instructions for the generation.
If not provided, the system prompt set in the `__init__` method will be used.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
If provided, this will override the `streaming_callback` set in the `__init__` method.
- `generation_kwargs`: Additional keyword arguments for text generation. These parameters will potentially override the parameters
passed in the `__init__` method. Supported parameters include temperature, max_new_tokens, top_p, etc.

**Returns**:

A dictionary with the following keys:
- `replies`: A list of generated text completions as strings.
- `meta`: A list of metadata dictionaries containing information about each generation,
including model name, finish reason, and token usage statistics.

<a id="haystack_integrations.components.generators.together_ai.generator.TogetherAIGenerator.run_async"></a>

#### TogetherAIGenerator.run\_async

```python
@component.output_types(replies=list[str], meta=list[dict[str, Any]])
async def run_async(
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[dict[str, Any]] = None) -> dict[str, Any]
```

Generate text completions asynchronously.

**Arguments**:

- `prompt`: The input prompt string for text generation.
- `system_prompt`: An optional system prompt to provide context or instructions for the generation.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
If provided, this will override the `streaming_callback` set in the `__init__` method.
- `generation_kwargs`: Additional keyword arguments for text generation. These parameters will potentially override the parameters
passed in the `__init__` method. Supported parameters include temperature, max_new_tokens, top_p, etc.

**Returns**:

A dictionary with the following keys:
- `replies`: A list of generated text completions as strings.
- `meta`: A list of metadata dictionaries containing information about each generation,
including model name, finish reason, and token usage statistics.

<a id="haystack_integrations.components.generators.together_ai.chat.chat_generator"></a>

## Module haystack\_integrations.components.generators.together\_ai.chat.chat\_generator

<a id="haystack_integrations.components.generators.together_ai.chat.chat_generator.TogetherAIChatGenerator"></a>

### TogetherAIChatGenerator

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
from haystack_integrations.components.generators.together_ai import TogetherAIChatGenerator
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

<a id="haystack_integrations.components.generators.together_ai.chat.chat_generator.TogetherAIChatGenerator.__init__"></a>

#### TogetherAIChatGenerator.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var("TOGETHER_API_KEY"),
             model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
             streaming_callback: Optional[StreamingCallbackT] = None,
             api_base_url: Optional[str] = "https://api.together.xyz/v1",
             generation_kwargs: Optional[Dict[str, Any]] = None,
             tools: Optional[Union[List[Tool], Toolset]] = None,
             timeout: Optional[float] = None,
             max_retries: Optional[int] = None,
             http_client_kwargs: Optional[Dict[str, Any]] = None)
```

Creates an instance of TogetherAIChatGenerator. Unless specified otherwise,

the default model is `meta-llama/Llama-3.3-70B-Instruct-Turbo`.

**Arguments**:

- `api_key`: The Together API key.
- `model`: The name of the Together AI chat completion model to use.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts StreamingChunk as an argument.
- `api_base_url`: The Together AI API Base url.
For more details, see Together AI [docs](https://docs.together.ai/docs/openai-api-compatibility).
- `generation_kwargs`: Other parameters to use for the model. These parameters are all sent directly to
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
- `tools`: A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
list of `Tool` objects or a `Toolset` instance.
- `timeout`: The timeout for the Together AI API call.
- `max_retries`: Maximum number of retries to contact Together AI after an internal error.
If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).

<a id="haystack_integrations.components.generators.together_ai.chat.chat_generator.TogetherAIChatGenerator.to_dict"></a>

#### TogetherAIChatGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.generators.together_ai.chat.chat_generator.TogetherAIChatGenerator.from_dict"></a>

#### TogetherAIChatGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OpenAIChatGenerator"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="haystack_integrations.components.generators.together_ai.chat.chat_generator.TogetherAIChatGenerator.run"></a>

#### TogetherAIChatGenerator.run

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

<a id="haystack_integrations.components.generators.together_ai.chat.chat_generator.TogetherAIChatGenerator.run_async"></a>

#### TogetherAIChatGenerator.run\_async

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
