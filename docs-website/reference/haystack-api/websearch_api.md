---
title: "Websearch"
id: websearch-api
description: "Web search engine for Haystack."
slug: "/websearch-api"
---


## `haystack.components.websearch.searchapi`

### `haystack.components.websearch.searchapi.SearchApiWebSearch`

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

#### `__init__`

```python
__init__(
    api_key: Secret = Secret.from_env_var("SEARCHAPI_API_KEY"),
    top_k: int | None = 10,
    allowed_domains: list[str] | None = None,
    search_params: dict[str, Any] | None = None,
) -> None
```

Initialize the SearchApiWebSearch component.

**Parameters:**

- **api_key** (<code>Secret</code>) – API key for the SearchApi API
- **top_k** (<code>int | None</code>) – Number of documents to return.
- **allowed_domains** (<code>list\[str\] | None</code>) – List of domains to limit the search to.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Additional parameters passed to the SearchApi API.
  For example, you can set 'num' to 100 to increase the number of search results.
  See the [SearchApi website](https://www.searchapi.io/) for more details.

The default search engine is Google, however, users can change it by setting the `engine`
parameter in the `search_params`.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> SearchApiWebSearch
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>SearchApiWebSearch</code> – The deserialized component.

#### `run`

```python
run(query: str) -> dict[str, list[Document] | list[str]]
```

Uses [SearchApi](https://www.searchapi.io/) to search the web.

**Parameters:**

- **query** (<code>str</code>) – Search query.

**Returns:**

- <code>dict\[str, list\[Document\] | list\[str\]\]</code> – A dictionary with the following keys:
- "documents": List of documents returned by the search engine.
- "links": List of links returned by the search engine.

**Raises:**

- <code>TimeoutError</code> – If the request to the SearchApi API times out.
- <code>SearchApiError</code> – If an error occurs while querying the SearchApi API.

#### `run_async`

```python
run_async(query: str) -> dict[str, list[Document] | list[str]]
```

Asynchronously uses [SearchApi](https://www.searchapi.io/) to search the web.

This is the asynchronous version of the `run` method with the same parameters and return values.

**Parameters:**

- **query** (<code>str</code>) – Search query.

**Returns:**

- <code>dict\[str, list\[Document\] | list\[str\]\]</code> – A dictionary with the following keys:
- "documents": List of documents returned by the search engine.
- "links": List of links returned by the search engine.

**Raises:**

- <code>TimeoutError</code> – If the request to the SearchApi API times out.
- <code>SearchApiError</code> – If an error occurs while querying the SearchApi API.

## `haystack.components.websearch.serper_dev`

### `haystack.components.websearch.serper_dev.SerperDevWebSearch`

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

#### `__init__`

```python
__init__(
    api_key: Secret = Secret.from_env_var("SERPERDEV_API_KEY"),
    top_k: int | None = 10,
    allowed_domains: list[str] | None = None,
    search_params: dict[str, Any] | None = None,
    *,
    exclude_subdomains: bool = False
) -> None
```

Initialize the SerperDevWebSearch component.

**Parameters:**

- **api_key** (<code>Secret</code>) – API key for the Serper API.
- **top_k** (<code>int | None</code>) – Number of documents to return.
- **allowed_domains** (<code>list\[str\] | None</code>) – List of domains to limit the search to.
- **exclude_subdomains** (<code>bool</code>) – Whether to exclude subdomains when filtering by allowed_domains.
  If True, only results from the exact domains in allowed_domains will be returned.
  If False, results from subdomains will also be included. Defaults to False.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Additional parameters passed to the Serper API.
  For example, you can set 'num' to 20 to increase the number of search results.
  See the [Serper website](https://serper.dev/) for more details.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> SerperDevWebSearch
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary to deserialize from.

**Returns:**

- <code>SerperDevWebSearch</code> – The deserialized component.

#### `run`

```python
run(query: str) -> dict[str, list[Document] | list[str]]
```

Use [Serper](https://serper.dev/) to search the web.

**Parameters:**

- **query** (<code>str</code>) – Search query.

**Returns:**

- <code>dict\[str, list\[Document\] | list\[str\]\]</code> – A dictionary with the following keys:
- "documents": List of documents returned by the search engine.
- "links": List of links returned by the search engine.

**Raises:**

- <code>SerperDevError</code> – If an error occurs while querying the SerperDev API.
- <code>TimeoutError</code> – If the request to the SerperDev API times out.

#### `run_async`

```python
run_async(query: str) -> dict[str, list[Document] | list[str]]
```

Asynchronously uses [Serper](https://serper.dev/) to search the web.

This is the asynchronous version of the `run` method with the same parameters and return values.

**Parameters:**

- **query** (<code>str</code>) – Search query.

**Returns:**

- <code>dict\[str, list\[Document\] | list\[str\]\]</code> – A dictionary with the following keys:
- "documents": List of documents returned by the search engine.
- "links": List of links returned by the search engine.

**Raises:**

- <code>SerperDevError</code> – If an error occurs while querying the SerperDev API.
- <code>TimeoutError</code> – If the request to the SerperDev API times out.
