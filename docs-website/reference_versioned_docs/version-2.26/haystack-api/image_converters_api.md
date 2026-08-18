---
title: "Image Converters"
id: image-converters-api
description: "Various converters to transform image data from one format to another."
slug: "/image-converters-api"
---


## document_to_image

### DocumentToImageContent

Converts documents sourced from PDF and image files into ImageContents.

This component processes a list of documents and extracts visual content from supported file formats, converting
them into ImageContents that can be used for multimodal AI tasks. It handles both direct image files and PDF
documents by extracting specific pages as images.

Documents are expected to have metadata containing:

- The `file_path_meta_field` key with a valid file path that exists when combined with `root_path`
- A supported image format (MIME type must be one of the supported image types)
- For PDF files, a `page_number` key specifying which page to extract

### Usage example

````
```python
from haystack import Document
from haystack.components.converters.image.document_to_image import DocumentToImageContent

converter = DocumentToImageContent(
    file_path_meta_field="file_path",
    root_path="/data/files",
    detail="high",
    size=(800, 600)
)

documents = [
    Document(content="Optional description of image.jpg", meta={"file_path": "image.jpg"}),
    Document(content="Text content of page 1 of doc.pdf", meta={"file_path": "doc.pdf", "page_number": 1})
]

result = converter.run(documents)
image_contents = result["image_contents"]
# [ImageContent(
#    base64_image='/9j/4A...', mime_type='image/jpeg', detail='high', meta={'file_path': 'image.jpg'}
#  ),
#  ImageContent(
#    base64_image='/9j/4A...', mime_type='image/jpeg', detail='high',
#    meta={'page_number': 1, 'file_path': 'doc.pdf'}
#  )]
```
````

#### __init__

```python
__init__(
    *,
    file_path_meta_field: str = "file_path",
    root_path: str | None = None,
    detail: Literal["auto", "high", "low"] | None = None,
    size: tuple[int, int] | None = None
)
```

Initialize the DocumentToImageContent component.

**Parameters:**

- **file_path_meta_field** (<code>str</code>) – The metadata field in the Document that contains the file path to the image or PDF.
- **root_path** (<code>str | None</code>) – The root directory path where document files are located. If provided, file paths in
  document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
- **detail** (<code>Literal['auto', 'high', 'low'] | None</code>) – Optional detail level of the image (only supported by OpenAI). Can be "auto", "high", or "low".
  This will be passed to the created ImageContent objects.
- **size** (<code>tuple\[int, int\] | None</code>) – If provided, resizes the image to fit within the specified dimensions (width, height) while
  maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
  when working with models that have resolution constraints or when transmitting images to remote services.

#### run

```python
run(documents: list[Document]) -> dict[str, list[ImageContent | None]]
```

Convert documents with image or PDF sources into ImageContent objects.

This method processes the input documents, extracting images from supported file formats and converting them
into ImageContent objects.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – A list of documents to process. Each document should have metadata containing at minimum
  a 'file_path_meta_field' key. PDF documents additionally require a 'page_number' key to specify which
  page to convert.

**Returns:**

- <code>dict\[str, list\[ImageContent | None\]\]</code> – Dictionary containing one key:
- "image_contents": ImageContents created from the processed documents. These contain base64-encoded image
  data and metadata. The order corresponds to order of input documents.

**Raises:**

- <code>ValueError</code> – If any document is missing the required metadata keys, has an invalid file path, or has an unsupported
  MIME type. The error message will specify which document and what information is missing or incorrect.

## file_to_document

### ImageFileToDocument

Converts image file references into empty Document objects with associated metadata.

This component is useful in pipelines where image file paths need to be wrapped in `Document` objects to be
processed by downstream components such as the `SentenceTransformersImageDocumentEmbedder`.

It does **not** extract any content from the image files, instead it creates `Document` objects with `None` as
their content and attaches metadata such as file path and any user-provided values.

### Usage example

```python
from haystack.components.converters.image import ImageFileToDocument

converter = ImageFileToDocument()

sources = ["image.jpg", "another_image.png"]

result = converter.run(sources=sources)
documents = result["documents"]

print(documents)

# [Document(id=..., meta: {'file_path': 'image.jpg'}),
# Document(id=..., meta: {'file_path': 'another_image.png'})]
```

#### __init__

```python
__init__(*, store_full_path: bool = False)
```

Initialize the ImageFileToDocument component.

**Parameters:**

- **store_full_path** (<code>bool</code>) – If True, the full path of the file is stored in the metadata of the document.
  If False, only the file name is stored.

#### run

```python
run(
    *,
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None
) -> dict[str, list[Document]]
```

Convert image files into empty Document objects with metadata.

This method accepts image file references (as file paths or ByteStreams) and creates `Document` objects
without content. These documents are enriched with metadata derived from the input source and optional
user-provided metadata.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects to convert.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the documents.
  This value can be a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced documents.
  If it's a list, its length must match the number of sources, as they are zipped together.
  For ByteStream objects, their `meta` is added to the output documents.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary containing:
- `documents`: A list of `Document` objects with empty content and associated metadata.

## file_to_image

### ImageFileToImageContent

Converts image files to ImageContent objects.

### Usage example

```python
from haystack.components.converters.image import ImageFileToImageContent

converter = ImageFileToImageContent()

sources = ["image.jpg", "another_image.png"]

image_contents = converter.run(sources=sources)["image_contents"]
print(image_contents)

# [ImageContent(base64_image='...',
#               mime_type='image/jpeg',
#               detail=None,
#               meta={'file_path': 'image.jpg'}),
#  ...]
```

#### __init__

```python
__init__(
    *,
    detail: Literal["auto", "high", "low"] | None = None,
    size: tuple[int, int] | None = None
)
```

Create the ImageFileToImageContent component.

**Parameters:**

- **detail** (<code>Literal['auto', 'high', 'low'] | None</code>) – Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
  This will be passed to the created ImageContent objects.
- **size** (<code>tuple\[int, int\] | None</code>) – If provided, resizes the image to fit within the specified dimensions (width, height) while
  maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
  when working with models that have resolution constraints or when transmitting images to remote services.

#### run

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    *,
    detail: Literal["auto", "high", "low"] | None = None,
    size: tuple[int, int] | None = None
) -> dict[str, list[ImageContent]]
```

Converts files to ImageContent objects.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects to convert.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the ImageContent objects.
  This value can be a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced ImageContent objects.
  If it's a list, its length must match the number of sources as they're zipped together.
  For ByteStream objects, their `meta` is added to the output ImageContent objects.
- **detail** (<code>Literal['auto', 'high', 'low'] | None</code>) – Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
  This will be passed to the created ImageContent objects.
  If not provided, the detail level will be the one set in the constructor.
- **size** (<code>tuple\[int, int\] | None</code>) – If provided, resizes the image to fit within the specified dimensions (width, height) while
  maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
  when working with models that have resolution constraints or when transmitting images to remote services.
  If not provided, the size value will be the one set in the constructor.

**Returns:**

- <code>dict\[str, list\[ImageContent\]\]</code> – A dictionary with the following keys:
- `image_contents`: A list of ImageContent objects.

## pdf_to_image

### PDFToImageContent

Converts PDF files to ImageContent objects.

### Usage example

```python
from haystack.components.converters.image import PDFToImageContent

converter = PDFToImageContent()

sources = ["file.pdf", "another_file.pdf"]

image_contents = converter.run(sources=sources)["image_contents"]
print(image_contents)

# [ImageContent(base64_image='...',
#               mime_type='application/pdf',
#               detail=None,
#               meta={'file_path': 'file.pdf', 'page_number': 1}),
#  ...]
```

#### __init__

```python
__init__(
    *,
    detail: Literal["auto", "high", "low"] | None = None,
    size: tuple[int, int] | None = None,
    page_range: list[str | int] | None = None
)
```

Create the PDFToImageContent component.

**Parameters:**

- **detail** (<code>Literal['auto', 'high', 'low'] | None</code>) – Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
  This will be passed to the created ImageContent objects.
- **size** (<code>tuple\[int, int\] | None</code>) – If provided, resizes the image to fit within the specified dimensions (width, height) while
  maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
  when working with models that have resolution constraints or when transmitting images to remote services.
- **page_range** (<code>list\[str | int\] | None</code>) – List of page numbers and/or page ranges to convert to images. Page numbers start at 1.
  If None, all pages in the PDF will be converted. Pages outside the valid range (1 to number of pages)
  will be skipped with a warning. For example, page_range=[1, 3] will convert only the first and third
  pages of the document. It also accepts printable range strings, e.g.: ['1-3', '5', '8', '10-12']
  will convert pages 1, 2, 3, 5, 8, 10, 11, 12.

#### run

```python
run(
    sources: list[str | Path | ByteStream],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    *,
    detail: Literal["auto", "high", "low"] | None = None,
    size: tuple[int, int] | None = None,
    page_range: list[str | int] | None = None
) -> dict[str, list[ImageContent]]
```

Converts files to ImageContent objects.

**Parameters:**

- **sources** (<code>list\[str | Path | ByteStream\]</code>) – List of file paths or ByteStream objects to convert.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the ImageContent objects.
  This value can be a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced ImageContent objects.
  If it's a list, its length must match the number of sources as they're zipped together.
  For ByteStream objects, their `meta` is added to the output ImageContent objects.
- **detail** (<code>Literal['auto', 'high', 'low'] | None</code>) – Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
  This will be passed to the created ImageContent objects.
  If not provided, the detail level will be the one set in the constructor.
- **size** (<code>tuple\[int, int\] | None</code>) – If provided, resizes the image to fit within the specified dimensions (width, height) while
  maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
  when working with models that have resolution constraints or when transmitting images to remote services.
  If not provided, the size value will be the one set in the constructor.
- **page_range** (<code>list\[str | int\] | None</code>) – List of page numbers and/or page ranges to convert to images. Page numbers start at 1.
  If None, all pages in the PDF will be converted. Pages outside the valid range (1 to number of pages)
  will be skipped with a warning. For example, page_range=[1, 3] will convert only the first and third
  pages of the document. It also accepts printable range strings, e.g.: ['1-3', '5', '8', '10-12']
  will convert pages 1, 2, 3, 5, 8, 10, 11, 12.
  If not provided, the page_range value will be the one set in the constructor.

**Returns:**

- <code>dict\[str, list\[ImageContent\]\]</code> – A dictionary with the following keys:
- `image_contents`: A list of ImageContent objects.
