---
title: "Audio"
id: audio-api
description: "Transcribes audio files."
slug: "/audio-api"
---


## whisper_local

### LocalWhisperTranscriber

Transcribes audio files using OpenAI's Whisper model on your local machine.

For the supported audio formats, languages, and other parameters, see the
[Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
[GitHub repository](https://github.com/openai/whisper).

### Usage example

```python
from haystack.components.audio import LocalWhisperTranscriber

whisper = LocalWhisperTranscriber(model="small")
transcription = whisper.run(sources=["test/test_files/audio/answer.wav"])
```

#### __init__

```python
__init__(
    model: WhisperLocalModel = "large",
    device: ComponentDevice | None = None,
    whisper_params: dict[str, Any] | None = None,
)
```

Creates an instance of the LocalWhisperTranscriber component.

**Parameters:**

- **model** (<code>WhisperLocalModel</code>) – The name of the model to use. Set to one of the following models:
  "tiny", "base", "small", "medium", "large" (default).
  For details on the models and their modifications, see the
  [Whisper documentation](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages).
- **device** (<code>ComponentDevice | None</code>) – The device for loading the model. If `None`, automatically selects the default device.

#### warm_up

```python
warm_up() -> None
```

Loads the model in memory.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> LocalWhisperTranscriber
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>LocalWhisperTranscriber</code> – The deserialized component.

#### run

```python
run(
    sources: list[str | Path | ByteStream],
    whisper_params: dict[str, Any] | None = None,
)
```

Transcribes a list of audio files into a list of documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – A list of paths or binary streams to transcribe.
- **whisper_params** (<code>dict\[str, Any\] | None</code>) – For the supported audio formats, languages, and other parameters, see the
  [Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
  [GitHup repo](https://github.com/openai/whisper).

**Returns:**

- – A dictionary with the following keys:
- `documents`: A list of documents where each document is a transcribed audio file. The content of
  the document is the transcription text, and the document's metadata contains the values returned by
  the Whisper model, such as the alignment data and the path to the audio file used
  for the transcription.

#### transcribe

```python
transcribe(
    sources: list[str | Path | ByteStream],
    **kwargs: list[str | Path | ByteStream]
) -> list[Document]
```

Transcribes the audio files into a list of Documents, one for each input file.

For the supported audio formats, languages, and other parameters, see the
[Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
[github repo](https://github.com/openai/whisper).

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – A list of paths or binary streams to transcribe.

**Returns:**

- <code>list\[Document\]</code> – A list of Documents, one for each file.

## whisper_remote

### RemoteWhisperTranscriber

Transcribes audio files using the OpenAI's Whisper API.

The component requires an OpenAI API key, see the
[OpenAI documentation](https://platform.openai.com/docs/api-reference/authentication) for more details.
For the supported audio formats, languages, and other parameters, see the
[Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text).

### Usage example

```python
from haystack.components.audio import RemoteWhisperTranscriber

whisper = RemoteWhisperTranscriber(model="whisper-1")
transcription = whisper.run(sources=["test/test_files/audio/answer.wav"])
```

#### __init__

```python
__init__(
    api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
    model: str = "whisper-1",
    api_base_url: str | None = None,
    organization: str | None = None,
    http_client_kwargs: dict[str, Any] | None = None,
    **kwargs: dict[str, Any] | None
)
```

Creates an instance of the RemoteWhisperTranscriber component.

**Parameters:**

- **api_key** (<code>Secret</code>) – OpenAI API key.
  You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
  during initialization.
- **model** (<code>str</code>) – Name of the model to use. Currently accepts only `whisper-1`.
- **organization** (<code>str | None</code>) – Your OpenAI organization ID. See OpenAI's documentation on
  [Setting Up Your Organization](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization).
- **api_base_url** (<code>str | None</code>) – An optional URL to use as the API base. For details, see the
  OpenAI [documentation](https://platform.openai.com/docs/api-reference/audio).
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
  For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
- **kwargs** – Other optional parameters for the model. These are sent directly to the OpenAI
  endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/audio) for more details.
  Some of the supported parameters are:
- `language`: The language of the input audio.
  Provide the input language in ISO-639-1 format
  to improve transcription accuracy and latency.
- `prompt`: An optional text to guide the model's
  style or continue a previous audio segment.
  The prompt should match the audio language.
- `response_format`: The format of the transcript
  output. This component only supports `json`.
- `temperature`: The sampling temperature, between 0
  and 1. Higher values like 0.8 make the output more
  random, while lower values like 0.2 make it more
  focused and deterministic. If set to 0, the model
  uses log probability to automatically increase the
  temperature until certain thresholds are hit.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> RemoteWhisperTranscriber
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>RemoteWhisperTranscriber</code> – The deserialized component.

#### run

```python
run(sources: list[str | Path | ByteStream])
```

Transcribes the list of audio files into a list of documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – A list of file paths or `ByteStream` objects containing the audio files to transcribe.

**Returns:**

- – A dictionary with the following keys:
- `documents`: A list of documents, one document for each file.
  The content of each document is the transcribed text.
