---
title: "Audio"
id: audio-api
description: "Transcribes audio files."
slug: "/audio-api"
---

<a id="whisper_local"></a>

# Module whisper\_local

<a id="whisper_local.LocalWhisperTranscriber"></a>

## LocalWhisperTranscriber

Transcribes audio files using OpenAI's Whisper model on your local machine.

For the supported audio formats, languages, and other parameters, see the
[Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
[GitHub repository](https://github.com/openai/whisper).

### Usage example

```python
from haystack.components.audio import LocalWhisperTranscriber

whisper = LocalWhisperTranscriber(model="small")
whisper.warm_up()
transcription = whisper.run(sources=["path/to/audio/file"])
```

<a id="whisper_local.LocalWhisperTranscriber.__init__"></a>

#### LocalWhisperTranscriber.\_\_init\_\_

```python
def __init__(model: WhisperLocalModel = "large",
             device: Optional[ComponentDevice] = None,
             whisper_params: Optional[dict[str, Any]] = None)
```

Creates an instance of the LocalWhisperTranscriber component.

**Arguments**:

- `model`: The name of the model to use. Set to one of the following models:
"tiny", "base", "small", "medium", "large" (default).
For details on the models and their modifications, see the
[Whisper documentation](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages).
- `device`: The device for loading the model. If `None`, automatically selects the default device.

<a id="whisper_local.LocalWhisperTranscriber.warm_up"></a>

#### LocalWhisperTranscriber.warm\_up

```python
def warm_up() -> None
```

Loads the model in memory.

<a id="whisper_local.LocalWhisperTranscriber.to_dict"></a>

#### LocalWhisperTranscriber.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="whisper_local.LocalWhisperTranscriber.from_dict"></a>

#### LocalWhisperTranscriber.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "LocalWhisperTranscriber"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="whisper_local.LocalWhisperTranscriber.run"></a>

#### LocalWhisperTranscriber.run

```python
@component.output_types(documents=list[Document])
def run(sources: list[Union[str, Path, ByteStream]],
        whisper_params: Optional[dict[str, Any]] = None)
```

Transcribes a list of audio files into a list of documents.

**Arguments**:

- `sources`: A list of paths or binary streams to transcribe.
- `whisper_params`: For the supported audio formats, languages, and other parameters, see the
[Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
[GitHup repo](https://github.com/openai/whisper).

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents where each document is a transcribed audio file. The content of
the document is the transcription text, and the document's metadata contains the values returned by
the Whisper model, such as the alignment data and the path to the audio file used
for the transcription.

<a id="whisper_local.LocalWhisperTranscriber.transcribe"></a>

#### LocalWhisperTranscriber.transcribe

```python
def transcribe(sources: list[Union[str, Path, ByteStream]],
               **kwargs) -> list[Document]
```

Transcribes the audio files into a list of Documents, one for each input file.

For the supported audio formats, languages, and other parameters, see the
[Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text) and the official Whisper
[github repo](https://github.com/openai/whisper).

**Arguments**:

- `sources`: A list of paths or binary streams to transcribe.

**Returns**:

A list of Documents, one for each file.

<a id="whisper_remote"></a>

# Module whisper\_remote

<a id="whisper_remote.RemoteWhisperTranscriber"></a>

## RemoteWhisperTranscriber

Transcribes audio files using the OpenAI's Whisper API.

The component requires an OpenAI API key, see the
[OpenAI documentation](https://platform.openai.com/docs/api-reference/authentication) for more details.
For the supported audio formats, languages, and other parameters, see the
[Whisper API documentation](https://platform.openai.com/docs/guides/speech-to-text).

### Usage example

```python
from haystack.components.audio import RemoteWhisperTranscriber

whisper = RemoteWhisperTranscriber(api_key=Secret.from_token("<your-api-key>"), model="tiny")
transcription = whisper.run(sources=["path/to/audio/file"])
```

<a id="whisper_remote.RemoteWhisperTranscriber.__init__"></a>

#### RemoteWhisperTranscriber.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
             model: str = "whisper-1",
             api_base_url: Optional[str] = None,
             organization: Optional[str] = None,
             http_client_kwargs: Optional[dict[str, Any]] = None,
             **kwargs)
```

Creates an instance of the RemoteWhisperTranscriber component.

**Arguments**:

- `api_key`: OpenAI API key.
You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
during initialization.
- `model`: Name of the model to use. Currently accepts only `whisper-1`.
- `organization`: Your OpenAI organization ID. See OpenAI's documentation on
[Setting Up Your Organization](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization).
- `api_base`: An optional URL to use as the API base. For details, see the
OpenAI [documentation](https://platform.openai.com/docs/api-reference/audio).
- `http_client_kwargs`: A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/`client`).
- `kwargs`: Other optional parameters for the model. These are sent directly to the OpenAI
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

<a id="whisper_remote.RemoteWhisperTranscriber.to_dict"></a>

#### RemoteWhisperTranscriber.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="whisper_remote.RemoteWhisperTranscriber.from_dict"></a>

#### RemoteWhisperTranscriber.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "RemoteWhisperTranscriber"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="whisper_remote.RemoteWhisperTranscriber.run"></a>

#### RemoteWhisperTranscriber.run

```python
@component.output_types(documents=list[Document])
def run(sources: list[Union[str, Path, ByteStream]])
```

Transcribes the list of audio files into a list of documents.

**Arguments**:

- `sources`: A list of file paths or `ByteStream` objects containing the audio files to transcribe.

**Returns**:

A dictionary with the following keys:
- `documents`: A list of documents, one document for each file.
The content of each document is the transcribed text.

