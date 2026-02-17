---
title: "PaddleOCR"
id: integrations-paddleocr
description: "PaddleOCR integration for Haystack"
slug: "/integrations-paddleocr"
---


## `haystack_integrations.components.converters.paddleocr.paddleocr_vl_document_converter`

### `PaddleOCRVLDocumentConverter`

This component extracts text from documents using PaddleOCR's large model
document parsing API.

PaddleOCR-VL is used behind the scenes. For more information, please
refer to:
https://www.paddleocr.ai/latest/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html

**Usage Example:**

```python
from haystack.utils import Secret
from haystack_integrations.components.converters.paddleocr import (
    PaddleOCRVLDocumentConverter,
)

converter = PaddleOCRVLDocumentConverter(
    api_url="http://xxxxx.aistudio-app.com/layout-parsing",
    access_token=Secret.from_env_var("AISTUDIO_ACCESS_TOKEN"),
)

result = converter.run(sources=["sample.pdf"])

documents = result["documents"]
raw_responses = result["raw_paddleocr_responses"]
```

#### `__init__`

```python
__init__(
    *,
    api_url: str,
    access_token: Secret = Secret.from_env_var("AISTUDIO_ACCESS_TOKEN"),
    file_type: FileTypeInput = None,
    use_doc_orientation_classify: bool | None = False,
    use_doc_unwarping: bool | None = False,
    use_layout_detection: bool | None = None,
    use_chart_recognition: bool | None = None,
    use_seal_recognition: bool | None = None,
    use_ocr_for_image_block: bool | None = None,
    layout_threshold: float | dict | None = None,
    layout_nms: bool | None = None,
    layout_unclip_ratio: float | tuple[float, float] | dict | None = None,
    layout_merge_bboxes_mode: str | dict | None = None,
    layout_shape_mode: str | None = None,
    prompt_label: str | None = None,
    format_block_content: bool | None = None,
    repetition_penalty: float | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    max_new_tokens: int | None = None,
    merge_layout_blocks: bool | None = None,
    markdown_ignore_labels: list[str] | None = None,
    vlm_extra_args: dict | None = None,
    prettify_markdown: bool | None = None,
    show_formula_number: bool | None = None,
    restructure_pages: bool | None = None,
    merge_tables: bool | None = None,
    relevel_titles: bool | None = None,
    visualize: bool | None = None,
    additional_params: dict[str, Any] | None = None
)
```

Create a `PaddleOCRVLDocumentConverter` component.

**Parameters:**

- **api_url** (<code>str</code>) – API URL. To obtain the API URL, visit the [PaddleOCR official
  website](https://aistudio.baidu.com/paddleocr), click the
  **API** button, choose the example code for PaddleOCR-VL, and copy
  the `API_URL`.
- **access_token** (<code>Secret</code>) – AI Studio access token. You can obtain it from [this
  page](https://aistudio.baidu.com/account/accessToken).
- **file_type** (<code>FileTypeInput</code>) – File type. Can be "pdf" for PDF files, "image" for
  image files, or `None` for auto-detection. If not specified, the
  file type will be inferred from the file extension.
- **use_doc_orientation_classify** (<code>bool | None</code>) – Whether to enable the document orientation classification
  function. Enabling this feature allows the input image to be
  automatically rotated to the correct orientation.
- **use_doc_unwarping** (<code>bool | None</code>) – Whether to enable the text image unwarping function. Enabling
  this feature allows automatic correction of distorted text images.
- **use_layout_detection** (<code>bool | None</code>) – Whether to enable the layout detection function.
- **use_chart_recognition** (<code>bool | None</code>) – Whether to enable the chart recognition function.
- **use_seal_recognition** (<code>bool | None</code>) – Whether to enable the seal recognition function.
- **use_ocr_for_image_block** (<code>bool | None</code>) – Whether to recognize text in image blocks.
- **layout_threshold** (<code>float | dict | None</code>) – Layout detection threshold. Can be a float or a dict with
  page-specific thresholds.
- **layout_nms** (<code>bool | None</code>) – Whether to perform NMS (Non-Maximum Suppression) on layout
  detection results.
- **layout_unclip_ratio** (<code>float | tuple\[float, float\] | dict | None</code>) – Layout unclip ratio. Can be a float, a tuple of (min, max), or a
  dict with page-specific values.
- **layout_merge_bboxes_mode** (<code>str | dict | None</code>) – Layout merge bounding boxes mode. Can be a string or a dict.
- **layout_shape_mode** (<code>str | None</code>) – Layout shape mode.
- **prompt_label** (<code>str | None</code>) – Prompt type for the VLM. Possible values are "ocr", "formula",
  "table", "chart", "seal", and "spotting".
- **format_block_content** (<code>bool | None</code>) – Whether to format block content.
- **repetition_penalty** (<code>float | None</code>) – Repetition penalty parameter used in VLM sampling.
- **temperature** (<code>float | None</code>) – Temperature parameter used in VLM sampling.
- **top_p** (<code>float | None</code>) – Top-p parameter used in VLM sampling.
- **min_pixels** (<code>int | None</code>) – Minimum number of pixels allowed during VLM preprocessing.
- **max_pixels** (<code>int | None</code>) – Maximum number of pixels allowed during VLM preprocessing.
- **max_new_tokens** (<code>int | None</code>) – Maximum number of tokens generated by the VLM.
- **merge_layout_blocks** (<code>bool | None</code>) – Whether to merge the layout detection boxes for cross-column or
  staggered top and bottom columns.
- **markdown_ignore_labels** (<code>list\[str\] | None</code>) – Layout labels that need to be ignored in Markdown.
- **vlm_extra_args** (<code>dict | None</code>) – Additional configuration parameters for the VLM.
- **prettify_markdown** (<code>bool | None</code>) – Whether to prettify the output Markdown text.
- **show_formula_number** (<code>bool | None</code>) – Whether to include formula numbers in the output markdown text.
- **restructure_pages** (<code>bool | None</code>) – Whether to restructure results across multiple pages.
- **merge_tables** (<code>bool | None</code>) – Whether to merge tables across pages.
- **relevel_titles** (<code>bool | None</code>) – Whether to relevel titles.
- **visualize** (<code>bool | None</code>) – Whether to return visualization results.
- **additional_params** (<code>dict\[str, Any\] | None</code>) – Additional parameters for calling the PaddleOCR API.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> PaddleOCRVLDocumentConverter
```

Deserialize the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>PaddleOCRVLDocumentConverter</code> – Deserialized component.

#### `run`

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]
```

Convert image or PDF files to Documents.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of image or PDF file paths or ByteStream objects.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single
  dictionary. If it's a single dictionary, its content is added to
  the metadata of all produced Documents. If it's a list, the length
  of the list must match the number of sources, because the two
  lists will be zipped. If `sources` contains ByteStream objects,
  their `meta` will be added to the output Documents.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: A list of created Documents.
- `raw_paddleocr_responses`: A list of raw PaddleOCR API responses.
