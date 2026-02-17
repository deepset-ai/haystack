---
title: "Unstructured"
id: integrations-unstructured
description: "Unstructured integration for Haystack"
slug: "/integrations-unstructured"
---


## `haystack_integrations.components.converters.unstructured.converter`

### `UnstructuredFileConverter`

A component for converting files to Haystack Documents using the Unstructured API (hosted or running locally).

For the supported file types and the specific API parameters, see
[Unstructured docs](https://docs.unstructured.io/api-reference/api-services/overview).

Usage example:

```python
from haystack_integrations.components.converters.unstructured import UnstructuredFileConverter

# make sure to either set the environment variable UNSTRUCTURED_API_KEY
# or run the Unstructured API locally:
# docker run -p 8000:8000 -d --rm --name unstructured-api quay.io/unstructured-io/unstructured-api:latest
# --port 8000 --host 0.0.0.0

converter = UnstructuredFileConverter(
    # api_url="http://localhost:8000/general/v0/general"  # <-- Uncomment this if running Unstructured locally
)
documents = converter.run(paths = ["a/file/path.pdf", "a/directory/path"])["documents"]
```

#### `__init__`

```python
__init__(
    api_url: str = UNSTRUCTURED_HOSTED_API_URL,
    api_key: Secret | None = Secret.from_env_var(
        "UNSTRUCTURED_API_KEY", strict=False
    ),
    document_creation_mode: Literal[
        "one-doc-per-file", "one-doc-per-page", "one-doc-per-element"
    ] = "one-doc-per-file",
    separator: str = "\n\n",
    unstructured_kwargs: dict[str, Any] | None = None,
    progress_bar: bool = True,
)
```

**Parameters:**

- **api_url** (<code>str</code>) – URL of the Unstructured API. Defaults to the URL of the hosted version.
  If you run the API locally, specify the URL of your local API (e.g. `"http://localhost:8000/general/v0/general"`).
- **api_key** (<code>Secret | None</code>) – API key for the Unstructured API.
  It can be explicitly passed or read the environment variable `UNSTRUCTURED_API_KEY` (recommended).
  If you run the API locally, it is not needed.
- **document_creation_mode** (<code>Literal['one-doc-per-file', 'one-doc-per-page', 'one-doc-per-element']</code>) – How to create Haystack Documents from the elements returned by Unstructured.
  `"one-doc-per-file"`: One Haystack Document per file. All elements are concatenated into one text field.
  `"one-doc-per-page"`: One Haystack Document per page.
  All elements on a page are concatenated into one text field.
  `"one-doc-per-element"`: One Haystack Document per element. Each element is converted to a Haystack Document.
- **separator** (<code>str</code>) – Separator between elements when concatenating them into one text field.
- **unstructured_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional parameters that are passed to the Unstructured API.
  For the available parameters, see
  [Unstructured API docs](https://docs.unstructured.io/api-reference/api-services/api-parameters).
- **progress_bar** (<code>bool</code>) – Whether to show a progress bar during the conversion.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> UnstructuredFileConverter
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>UnstructuredFileConverter</code> – Deserialized component.

#### `run`

```python
run(
    paths: list[str] | list[os.PathLike],
    meta: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, list[Document]]
```

Convert files to Haystack Documents using the Unstructured API.

**Parameters:**

- **paths** (<code>list\[str\] | list\[PathLike\]</code>) – List of paths to convert. Paths can be files or directories.
  If a path is a directory, all files in the directory are converted. Subdirectories are ignored.
- **meta** (<code>dict\[str, Any\] | list\[dict\[str, Any\]\] | None</code>) – Optional metadata to attach to the Documents.
  This value can be either a list of dictionaries or a single dictionary.
  If it's a single dictionary, its content is added to the metadata of all produced Documents.
  If it's a list, the length of the list must match the number of paths, because the two lists will be zipped.
  Please note that if the paths contain directories, `meta` can only be a single dictionary
  (same metadata for all files).

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with the following key:
- `documents`: List of Haystack Documents.

**Raises:**

- <code>ValueError</code> – If `meta` is a list and `paths` contains directories.
