---
title: "Google AI"
id: integrations-google-ai
description: "Google AI integration for Haystack"
slug: "/integrations-google-ai"
---

<a id="haystack_integrations.components.generators.google_ai.gemini"></a>

# Module haystack\_integrations.components.generators.google\_ai.gemini

<a id="haystack_integrations.components.generators.google_ai.gemini.GoogleAIGeminiGenerator"></a>

## GoogleAIGeminiGenerator

Generates text using multimodal Gemini models through Google AI Studio.

### Usage example

```python
from haystack.utils import Secret
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

gemini = GoogleAIGeminiGenerator(model="gemini-2.0-flash", api_key=Secret.from_token("<MY_API_KEY>"))
res = gemini.run(parts = ["What is the most interesting thing you know?"])
for answer in res["replies"]:
    print(answer)
```

#### Multimodal example

```python
import requests
from haystack.utils import Secret
from haystack.dataclasses.byte_stream import ByteStream
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

BASE_URL = (
    "https://raw.githubusercontent.com/deepset-ai/haystack-core-integrations"
    "/main/integrations/google_ai/example_assets"
)

URLS = [
    f"{BASE_URL}/robot1.jpg",
    f"{BASE_URL}/robot2.jpg",
    f"{BASE_URL}/robot3.jpg",
    f"{BASE_URL}/robot4.jpg"
]
images = [
    ByteStream(data=requests.get(url).content, mime_type="image/jpeg")
    for url in URLS
]

gemini = GoogleAIGeminiGenerator(model="gemini-2.0-flash", api_key=Secret.from_token("<MY_API_KEY>"))
result = gemini.run(parts = ["What can you tell me about this robots?", *images])
for answer in result["replies"]:
    print(answer)
```

<a id="haystack_integrations.components.generators.google_ai.gemini.GoogleAIGeminiGenerator.__init__"></a>

#### GoogleAIGeminiGenerator.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var("GOOGLE_API_KEY"),
             model: str = "gemini-2.0-flash",
             generation_config: Optional[Union[GenerationConfig,
                                               Dict[str, Any]]] = None,
             safety_settings: Optional[Dict[HarmCategory,
                                            HarmBlockThreshold]] = None,
             streaming_callback: Optional[Callable[[StreamingChunk],
                                                   None]] = None)
```

Initializes a `GoogleAIGeminiGenerator` instance.

To get an API key, visit: https://makersuite.google.com

**Arguments**:

- `api_key`: Google AI Studio API key.
- `model`: Name of the model to use. For available models, see https://ai.google.dev/gemini-api/docs/models/gemini
- `generation_config`: The generation configuration to use.
This can either be a `GenerationConfig` object or a dictionary of parameters.
For available parameters, see
[the `GenerationConfig` API reference](https://ai.google.dev/api/python/google/generativeai/GenerationConfig).
- `safety_settings`: The safety settings to use.
A dictionary with `HarmCategory` as keys and `HarmBlockThreshold` as values.
For more information, see [the API reference](https://ai.google.dev/api)
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts StreamingChunk as an argument.

<a id="haystack_integrations.components.generators.google_ai.gemini.GoogleAIGeminiGenerator.to_dict"></a>

#### GoogleAIGeminiGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.generators.google_ai.gemini.GoogleAIGeminiGenerator.from_dict"></a>

#### GoogleAIGeminiGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "GoogleAIGeminiGenerator"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.generators.google_ai.gemini.GoogleAIGeminiGenerator.run"></a>

#### GoogleAIGeminiGenerator.run

```python
@component.output_types(replies=List[str])
def run(parts: Variadic[Union[str, ByteStream, Part]],
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None)
```

Generates text based on the given input parts.

**Arguments**:

- `parts`: A heterogeneous list of strings, `ByteStream` or `Part` objects.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.

**Returns**:

A dictionary containing the following key:
- `replies`: A list of strings containing the generated responses.

<a id="haystack_integrations.components.generators.google_ai.chat.gemini"></a>

# Module haystack\_integrations.components.generators.google\_ai.chat.gemini

<a id="haystack_integrations.components.generators.google_ai.chat.gemini.GoogleAIGeminiChatGenerator"></a>

## GoogleAIGeminiChatGenerator

Completes chats using Gemini models through Google AI Studio.

It uses the [`ChatMessage`](https://docs.haystack.deepset.ai/docs/data-classes#chatmessage)
  dataclass to interact with the model.

### Usage example

```python
from haystack.utils import Secret
from haystack.dataclasses.chat_message import ChatMessage
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator


gemini_chat = GoogleAIGeminiChatGenerator(model="gemini-2.0-flash", api_key=Secret.from_token("<MY_API_KEY>"))

messages = [ChatMessage.from_user("What is the most interesting thing you know?")]
res = gemini_chat.run(messages=messages)
for reply in res["replies"]:
    print(reply.text)

messages += res["replies"] + [ChatMessage.from_user("Tell me more about it")]
res = gemini_chat.run(messages=messages)
for reply in res["replies"]:
    print(reply.text)
```


#### With function calling:

```python
from typing import Annotated
from haystack.utils import Secret
from haystack.dataclasses.chat_message import ChatMessage
from haystack.components.tools import ToolInvoker
from haystack.tools import create_tool_from_function

from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator

# example function to get the current weather
def get_current_weather(
    location: Annotated[str, "The city for which to get the weather, e.g. 'San Francisco'"] = "Munich",
    unit: Annotated[str, "The unit for the temperature, e.g. 'celsius'"] = "celsius",
) -> str:
    return f"The weather in {location} is sunny. The temperature is 20 {unit}."

tool = create_tool_from_function(get_current_weather)
tool_invoker = ToolInvoker(tools=[tool])

gemini_chat = GoogleAIGeminiChatGenerator(
    model="gemini-2.0-flash-exp",
    api_key=Secret.from_token("<MY_API_KEY>"),
    tools=[tool],
)
user_message = [ChatMessage.from_user("What is the temperature in celsius in Berlin?")]
replies = gemini_chat.run(messages=user_message)["replies"]
print(replies[0].tool_calls)

# actually invoke the tool
tool_messages = tool_invoker.run(messages=replies)["tool_messages"]
messages = user_message + replies + tool_messages

# transform the tool call result into a human readable message
final_replies = gemini_chat.run(messages=messages)["replies"]
print(final_replies[0].text)
```

<a id="haystack_integrations.components.generators.google_ai.chat.gemini.GoogleAIGeminiChatGenerator.__init__"></a>

#### GoogleAIGeminiChatGenerator.\_\_init\_\_

```python
def __init__(*,
             api_key: Secret = Secret.from_env_var("GOOGLE_API_KEY"),
             model: str = "gemini-2.0-flash",
             generation_config: Optional[Union[GenerationConfig,
                                               Dict[str, Any]]] = None,
             safety_settings: Optional[Dict[HarmCategory,
                                            HarmBlockThreshold]] = None,
             tools: Optional[List[Tool]] = None,
             tool_config: Optional[content_types.ToolConfigDict] = None,
             streaming_callback: Optional[StreamingCallbackT] = None)
```

Initializes a `GoogleAIGeminiChatGenerator` instance.

To get an API key, visit: https://aistudio.google.com/

**Arguments**:

- `api_key`: Google AI Studio API key. To get a key,
see [Google AI Studio](https://aistudio.google.com/).
- `model`: Name of the model to use. For available models, see https://ai.google.dev/gemini-api/docs/models/gemini.
- `generation_config`: The generation configuration to use.
This can either be a `GenerationConfig` object or a dictionary of parameters.
For available parameters, see
[the API reference](https://ai.google.dev/api/generate-content).
- `safety_settings`: The safety settings to use.
A dictionary with `HarmCategory` as keys and `HarmBlockThreshold` as values.
For more information, see [the API reference](https://ai.google.dev/api/generate-content)
- `tools`: A list of tools for which the model can prepare calls.
- `tool_config`: The tool config to use. See the documentation for
[ToolConfig](https://ai.google.dev/api/caching#ToolConfig).
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
The callback function accepts StreamingChunk as an argument.

<a id="haystack_integrations.components.generators.google_ai.chat.gemini.GoogleAIGeminiChatGenerator.to_dict"></a>

#### GoogleAIGeminiChatGenerator.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.generators.google_ai.chat.gemini.GoogleAIGeminiChatGenerator.from_dict"></a>

#### GoogleAIGeminiChatGenerator.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "GoogleAIGeminiChatGenerator"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.generators.google_ai.chat.gemini.GoogleAIGeminiChatGenerator.run"></a>

#### GoogleAIGeminiChatGenerator.run

```python
@component.output_types(replies=List[ChatMessage])
def run(messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        *,
        tools: Optional[List[Tool]] = None)
```

Generates text based on the provided messages.

**Arguments**:

- `messages`: A list of `ChatMessage` instances, representing the input messages.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
- `tools`: A list of tools for which the model can prepare calls. If set, it will override the `tools` parameter set
during component initialization.

**Returns**:

A dictionary containing the following key:
- `replies`:  A list containing the generated responses as `ChatMessage` instances.

<a id="haystack_integrations.components.generators.google_ai.chat.gemini.GoogleAIGeminiChatGenerator.run_async"></a>

#### GoogleAIGeminiChatGenerator.run\_async

```python
@component.output_types(replies=List[ChatMessage])
async def run_async(messages: List[ChatMessage],
                    streaming_callback: Optional[StreamingCallbackT] = None,
                    *,
                    tools: Optional[List[Tool]] = None)
```

Async version of the run method. Generates text based on the provided messages.

**Arguments**:

- `messages`: A list of `ChatMessage` instances, representing the input messages.
- `streaming_callback`: A callback function that is called when a new token is received from the stream.
- `tools`: A list of tools for which the model can prepare calls. If set, it will override the `tools` parameter set
during component initialization.

**Returns**:

A dictionary containing the following key:
- `replies`:  A list containing the generated responses as `ChatMessage` instances.
