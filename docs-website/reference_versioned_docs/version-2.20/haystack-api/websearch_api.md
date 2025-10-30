---
title: "Websearch"
id: websearch-api
description: "Web search engine for Haystack."
slug: "/websearch-api"
---

<a id="serper_dev"></a>

## Module serper\_dev

<a id="serper_dev.SerperDevWebSearch"></a>

### SerperDevWebSearch

Uses [Serper](https://serper.dev/) to search the web for relevant documents.

See the [Serper Dev website](https://serper.dev/) for more details.

Usage example:
```python
from haystack.components.websearch import SerperDevWebSearch
from haystack.utils import Secret

websearch = SerperDevWebSearch(top_k=10, api_key=Secret.from_token("test-api-key"))
results = websearch.run(query="Who is the boyfriend of Olivia Wilde?")

assert results["documents"]
assert results["links"]

# Example with domain filtering - exclude subdomains
websearch_filtered = SerperDevWebSearch(
    top_k=10,
    allowed_domains=["example.com"],
    exclude_subdomains=True,  # Only results from example.com, not blog.example.com
    api_key=Secret.from_token("test-api-key")
)
results_filtered = websearch_filtered.run(query="search query")
```

<a id="serper_dev.SerperDevWebSearch.__init__"></a>

#### SerperDevWebSearch.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("SERPERDEV_API_KEY"),
             top_k: Optional[int] = 10,
             allowed_domains: Optional[list[str]] = None,
             search_params: Optional[dict[str, Any]] = None,
             *,
             exclude_subdomains: bool = False)
```

Initialize the SerperDevWebSearch component.

**Arguments**:

- `api_key`: API key for the Serper API.
- `top_k`: Number of documents to return.
- `allowed_domains`: List of domains to limit the search to.
- `exclude_subdomains`: Whether to exclude subdomains when filtering by allowed_domains.
If True, only results from the exact domains in allowed_domains will be returned.
If False, results from subdomains will also be included. Defaults to False.
- `search_params`: Additional parameters passed to the Serper API.
For example, you can set 'num' to 20 to increase the number of search results.
See the [Serper website](https://serper.dev/) for more details.

<a id="serper_dev.SerperDevWebSearch.to_dict"></a>

#### SerperDevWebSearch.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="serper_dev.SerperDevWebSearch.from_dict"></a>

#### SerperDevWebSearch.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "SerperDevWebSearch"
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="serper_dev.SerperDevWebSearch.run"></a>

#### SerperDevWebSearch.run

```python
@component.output_types(documents=list[Document], links=list[str])
def run(query: str) -> dict[str, Union[list[Document], list[str]]]
```

Use [Serper](https://serper.dev/) to search the web.

**Arguments**:

- `query`: Search query.

**Raises**:

- `SerperDevError`: If an error occurs while querying the SerperDev API.
- `TimeoutError`: If the request to the SerperDev API times out.

**Returns**:

A dictionary with the following keys:
- "documents": List of documents returned by the search engine.
- "links": List of links returned by the search engine.

<a id="searchapi"></a>

## Module searchapi

<a id="searchapi.SearchApiWebSearch"></a>

### SearchApiWebSearch

Uses [SearchApi](https://www.searchapi.io/) to search the web for relevant documents.

Usage example:
```python
from haystack.components.websearch import SearchApiWebSearch
from haystack.utils import Secret

websearch = SearchApiWebSearch(top_k=10, api_key=Secret.from_token("test-api-key"))
results = websearch.run(query="Who is the boyfriend of Olivia Wilde?")

assert results["documents"]
assert results["links"]
```

<a id="searchapi.SearchApiWebSearch.__init__"></a>

#### SearchApiWebSearch.\_\_init\_\_

```python
def __init__(api_key: Secret = Secret.from_env_var("SEARCHAPI_API_KEY"),
             top_k: Optional[int] = 10,
             allowed_domains: Optional[list[str]] = None,
             search_params: Optional[dict[str, Any]] = None)
```

Initialize the SearchApiWebSearch component.

**Arguments**:

- `api_key`: API key for the SearchApi API
- `top_k`: Number of documents to return.
- `allowed_domains`: List of domains to limit the search to.
- `search_params`: Additional parameters passed to the SearchApi API.
For example, you can set 'num' to 100 to increase the number of search results.
See the [SearchApi website](https://www.searchapi.io/) for more details.

The default search engine is Google, however, users can change it by setting the `engine`
parameter in the `search_params`.

<a id="searchapi.SearchApiWebSearch.to_dict"></a>

#### SearchApiWebSearch.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="searchapi.SearchApiWebSearch.from_dict"></a>

#### SearchApiWebSearch.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "SearchApiWebSearch"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: The dictionary to deserialize from.

**Returns**:

The deserialized component.

<a id="searchapi.SearchApiWebSearch.run"></a>

#### SearchApiWebSearch.run

```python
@component.output_types(documents=list[Document], links=list[str])
def run(query: str) -> dict[str, Union[list[Document], list[str]]]
```

Uses [SearchApi](https://www.searchapi.io/) to search the web.

**Arguments**:

- `query`: Search query.

**Raises**:

- `TimeoutError`: If the request to the SearchApi API times out.
- `SearchApiError`: If an error occurs while querying the SearchApi API.

**Returns**:

A dictionary with the following keys:
- "documents": List of documents returned by the search engine.
- "links": List of links returned by the search engine.
