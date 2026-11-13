---
title: "FunASR"
id: integrations-funasr
description: "FunASR speech-to-text integration for Haystack"
slug: "/integrations-funasr"
---


## haystack_integrations.components.audio.funasr.transcriber

### FunASRTranscriber

Transcribes audio files to Documents using [FunASR](https://github.com/modelscope/FunASR).

FunASR is an open-source speech recognition toolkit from Alibaba DAMO Academy.
It supports 50+ languages, speaker diarization, and timestamp extraction, and runs
entirely locally — no API key required.

Models are downloaded from ModelScope on first use and cached in `~/.cache/modelscope`.

**Usage Example:**

```python
from haystack_integrations.components.audio.funasr import FunASRTranscriber

transcriber = FunASRTranscriber()
result = transcriber.run(sources=["speech.wav", "interview.mp3"])
documents = result["documents"]
```

**Speaker diarization and punctuation:**

```python
from haystack.utils import ComponentDevice

transcriber = FunASRTranscriber(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
    device=ComponentDevice.from_str("cuda"),
)
```

**SenseVoice with inverse text normalisation:**

```python
transcriber = FunASRTranscriber(
    model="iic/SenseVoiceSmall",
    generation_kwargs={"use_itn": True, "merge_vad": True, "language": "auto"},
)
```

#### __init__

```python
__init__(
    *,
    model: str = "iic/SenseVoiceSmall",
    vad_model: str | None = "fsmn-vad",
    punc_model: str | None = "ct-punc",
    spk_model: str | None = None,
    device: ComponentDevice | None = None,
    batch_size_s: int = 300,
    store_full_path: bool = False,
    generation_kwargs: dict[str, Any] | None = None
) -> None
```

Create a FunASRTranscriber component.

**Parameters:**

- **model** (<code>str</code>) – FunASR model name or local path. Defaults to `"iic/SenseVoiceSmall"`,
  a multilingual model supporting 50+ languages that is 5-10x faster than Whisper.
  Alternatives include `"paraformer-zh"` (Chinese) or `"paraformer-en"` (English).
  Browse available models at https://modelscope.github.io/FunASR/model-selection.html.
- **vad_model** (<code>str | None</code>) – Voice activity detection model used to split long audio into segments.
  Set to `None` to process the audio as a single stream.
  Browse available VAD models at https://www.modelscope.cn/models.
- **punc_model** (<code>str | None</code>) – Punctuation restoration model. Set to `None` to disable punctuation.
  Browse available punctuation models at https://www.modelscope.cn/models.
- **spk_model** (<code>str | None</code>) – Speaker diarization model (e.g. `"cam++"`). When set, a `"speakers"`
  key is included in the Document metadata. Defaults to `None` (diarization disabled).
  Browse available speaker diarization models at https://www.modelscope.cn/models.
- **device** (<code>ComponentDevice | None</code>) – The device to run inference on. If `None`, the default device is selected
  automatically. Use `ComponentDevice.from_str("cuda")` for GPU inference.
- **batch_size_s** (<code>int</code>) – Batch size in seconds for VAD-segmented audio. Larger values
  improve throughput at the cost of memory.
- **store_full_path** (<code>bool</code>) – If `True`, store the full audio file path in Document metadata.
  If `False` (default), store only the file name.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Extra keyword arguments forwarded to `AutoModel.generate()`.
  Use this for model-specific options such as `use_itn=True` or `merge_vad=True`
  for SenseVoice, or `hotword="..."` for contextual recognition.

#### warm_up

```python
warm_up() -> None
```

Load the FunASR model into memory.

Models are downloaded from ModelScope on first call and cached locally.
This method is idempotent — calling it multiple times is safe.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> FunASRTranscriber
```

Deserialize the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>FunASRTranscriber</code> – Deserialized component.

#### run

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document]]
```

Transcribe audio sources to Documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – Audio file paths (`str` or `Path`) or `ByteStream` objects.
  Supported formats: WAV, MP3, FLAC, OGG, M4A, AAC, and any format that
  FunASR's underlying audio backend (soundfile/ffmpeg) can decode.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Metadata to attach to the produced Documents. Pass a single dict
  to apply the same metadata to all Documents, or a list aligned with `sources`.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – Dictionary with key `"documents"` — one `Document` per source whose
  `content` holds the full transcript text.
