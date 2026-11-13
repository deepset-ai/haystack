---
title: "MiniCPM OCR"
id: integrations-minicpm-ocr
description: "MiniCPM-OCR integration for Haystack"
slug: "/integrations-minicpm-ocr"
---


## haystack_integrations.components.converters.minicpm_ocr.converter

### MiniCPMOCRDocumentConverter

Converts PDF and image files to Documents using self-hosted MiniCPM-OCR.

The component runs the official `minicpm-ocr` SDK locally: it detects the
layout of every page with PP-DocLayout, crops each block, sends it to a
MiniCPM-OCR inference server, and assembles the recognized blocks into
Markdown. The inference server can run on vLLM, SGLang, llama.cpp or
Ollama; only the layout model runs in this process.

Returns one Document per source, with pages separated by form feed characters (`\f`),
so `DocumentSplitter` can split the content by page.

### Usage example

```python
from haystack_integrations.components.converters.minicpm_ocr import MiniCPMOCRDocumentConverter

converter = MiniCPMOCRDocumentConverter(server_url="http://localhost:8000", model="minicpm-ocr")
converter.warm_up()
result = converter.run(sources=["sample.pdf"])
documents = result["documents"]
```

#### __init__

```python
__init__(
    *,
    server_url: str,
    model: str = "minicpm-ocr",
    backend: Backend = "vllm",
    api_key: Secret = Secret.from_env_var("MINICPM_OCR_API_KEY", strict=False),
    layout_model_dir: str | None = None,
    layout_device: str = "cpu",
    request_timeout: float = 300.0,
    page_workers: int = 1,
    max_workers: int = 1,
    store_full_path: bool = False
) -> None
```

Create a MiniCPM-OCR document converter.

**Parameters:**

- **server_url** (<code>str</code>) – Base URL of the MiniCPM-OCR inference server, for example
  `http://localhost:8000` for vLLM or `http://localhost:11434`
  for Ollama. A trailing `/v1` is accepted and removed, because the
  SDK appends `/v1/chat/completions` itself.
- **model** (<code>str</code>) – Model name served by the inference server.
- **backend** (<code>Backend</code>) – Server dialect: `vllm`, `llamacpp`, `ollama` or `openai`
  (SGLang and LMDeploy use `openai`).
- **api_key** (<code>Secret</code>) – Optional API key for the inference server.
- **layout_model_dir** (<code>str | None</code>) – Directory holding the PP-DocLayout model. When unset, the SDK uses
  `~/.cache/minicpm-ocr/layout`. The model is downloaded there in
  `warm_up()` if it is missing.
- **layout_device** (<code>str</code>) – Device for layout detection, for example `cpu` or `cuda:0`.
- **request_timeout** (<code>float</code>) – Timeout in seconds for one recognition request.
- **page_workers** (<code>int</code>) – Pages rendered and detected concurrently.
- **max_workers** (<code>int</code>) – Recognition requests in flight concurrently.
- **store_full_path** (<code>bool</code>) – If True, the full path of the source file is stored in the Document metadata.
  If False, only the file name is stored.

**Raises:**

- <code>ValueError</code> – If `server_url` is empty, `backend` is unknown, or a worker
  count is not positive.

#### warm_up

```python
warm_up() -> None
```

Make sure the PP-DocLayout layout model is available locally, downloading it on first use.

#### run

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document]]
```

Convert PDF or image files to Documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects. Supported formats are PDF,
  PNG, JPEG, BMP, TIFF and WebP.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents. A single dict is
  applied to all documents; a list must match the number of sources.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following key:
- `documents`: One Document per converted source. The content is Markdown with pages
  separated by `\f`, and the metadata includes `page_count` and `ocr_engine`.
