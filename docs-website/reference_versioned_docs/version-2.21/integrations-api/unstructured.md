---
title: "Unstructured"
id: integrations-unstructured
description: "Unstructured integration for Haystack"
slug: "/integrations-unstructured"
---

<a id="haystack_integrations.components.converters.unstructured.converter"></a>

## Module haystack\_integrations.components.converters.unstructured.converter

<a id="haystack_integrations.components.converters.unstructured.converter.UnstructuredFileConverter"></a>

### UnstructuredFileConverter

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

<a id="haystack_integrations.components.converters.unstructured.converter.UnstructuredFileConverter.__init__"></a>

#### UnstructuredFileConverter.\_\_init\_\_

```python
def __init__(api_url: str = UNSTRUCTURED_HOSTED_API_URL,
             api_key: Optional[Secret] = Secret.from_env_var(
                 "UNSTRUCTURED_API_KEY", strict=False),
             document_creation_mode: Literal[
                 "one-doc-per-file", "one-doc-per-page",
                 "one-doc-per-element"] = "one-doc-per-file",
             separator: str = "\n\n",
             unstructured_kwargs: Optional[Dict[str, Any]] = None,
             progress_bar: bool = True)
```

**Arguments**:

- `api_url`: URL of the Unstructured API. Defaults to the URL of the hosted version.
If you run the API locally, specify the URL of your local API (e.g. `"http://localhost:8000/general/v0/general"`).
- `api_key`: API key for the Unstructured API.
It can be explicitly passed or read the environment variable `UNSTRUCTURED_API_KEY` (recommended).
If you run the API locally, it is not needed.
- `document_creation_mode`: How to create Haystack Documents from the elements returned by Unstructured.
`"one-doc-per-file"`: One Haystack Document per file. All elements are concatenated into one text field.
`"one-doc-per-page"`: One Haystack Document per page.
All elements on a page are concatenated into one text field.
`"one-doc-per-element"`: One Haystack Document per element. Each element is converted to a Haystack Document.
- `separator`: Separator between elements when concatenating them into one text field.
- `unstructured_kwargs`: Additional parameters that are passed to the Unstructured API.
For the available parameters, see
[Unstructured API docs](https://docs.unstructured.io/api-reference/api-services/api-parameters).
- `progress_bar`: Whether to show a progress bar during the conversion.

<a id="haystack_integrations.components.converters.unstructured.converter.UnstructuredFileConverter.to_dict"></a>

#### UnstructuredFileConverter.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_integrations.components.converters.unstructured.converter.UnstructuredFileConverter.from_dict"></a>

#### UnstructuredFileConverter.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "UnstructuredFileConverter"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.components.converters.unstructured.converter.UnstructuredFileConverter.run"></a>

#### UnstructuredFileConverter.run

```python
@component.output_types(documents=List[Document])
def run(
    paths: Union[List[str], List[os.PathLike]],
    meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
) -> Dict[str, List[Document]]
```

Convert files to Haystack Documents using the Unstructured API.

**Arguments**:

- `paths`: List of paths to convert. Paths can be files or directories.
If a path is a directory, all files in the directory are converted. Subdirectories are ignored.
- `meta`: Optional metadata to attach to the Documents.
This value can be either a list of dictionaries or a single dictionary.
If it's a single dictionary, its content is added to the metadata of all produced Documents.
If it's a list, the length of the list must match the number of paths, because the two lists will be zipped.
Please note that if the paths contain directories, `meta` can only be a single dictionary
(same metadata for all files).

**Raises**:

- `ValueError`: If `meta` is a list and `paths` contains directories.

**Returns**:

A dictionary with the following key:
- `documents`: List of Haystack Documents.
