---
title: "Brave Search"
id: integrations-brave
description: "Brave Search integration for Haystack"
slug: "/integrations-brave"
---


## haystack_integrations.components.websearch.brave.brave_websearch

### BraveWebSearch

A component that uses the Brave Search API to search the web and return results as Haystack Documents.

You need a Brave Search API key from [brave.com/search/api](https://brave.com/search/api/).

### Usage example

```python
from haystack_integrations.components.websearch.brave import BraveWebSearch
from haystack.utils import Secret

websearch = BraveWebSearch(
    api_key=Secret.from_env_var("BRAVE_API_KEY"),
    top_k=5,
)
result = websearch.run(query="What is Haystack by deepset?")
documents = result["documents"]
links = result["links"]
```

#### __init__

```python
__init__(
    api_key: Secret = Secret.from_env_var("BRAVE_API_KEY"),
    top_k: int | None = 10,
    country: str | None = None,
    search_lang: str | None = None,
    extra_params: dict[str, Any] | None = None,
    timeout: int = 10,
    max_retries: int = 3,
) -> None
```

Initialize the BraveWebSearch component.

**Parameters:**

- **api_key** (<code>Secret</code>) – Brave Search API key. Defaults to the `BRAVE_API_KEY` environment variable.
- **top_k** (<code>int | None</code>) – Maximum number of results to return. Maps to the `count` parameter in the Brave API.
- **country** (<code>str | None</code>) – 2-letter country code to bias search results (e.g. `"US"`, `"DE"`).
- **search_lang** (<code>str | None</code>) – Language code for search results (e.g. `"en"`, `"de"`).
- **extra_params** (<code>dict\[str, Any\] | None</code>) – Additional query parameters passed directly to the Brave Search API.
- **timeout** (<code>int</code>) – Timeout in seconds for the HTTP request. Defaults to 10.
- **max_retries** (<code>int</code>) – Maximum number of retry attempts on transient failures. Defaults to 3.

#### run

```python
run(query: str, top_k: int | None = None) -> dict[str, Any]
```

Search the web using Brave Search and return results as Documents.

**Parameters:**

- **query** (<code>str</code>) – Search query string.
- **top_k** (<code>int | None</code>) – Optional per-run override of the maximum number of results.
  If not provided, the init-time `top_k` is used.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with:
- `documents`: List of Documents containing search result content.
- `links`: List of URLs from the search results.

#### run_async

```python
run_async(query: str, top_k: int | None = None) -> dict[str, Any]
```

Asynchronously search the web using Brave Search and return results as Documents.

**Parameters:**

- **query** (<code>str</code>) – Search query string.
- **top_k** (<code>int | None</code>) – Optional per-run override of the maximum number of results.
  If not provided, the init-time `top_k` is used.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with:
- `documents`: List of Documents containing search result content.
- `links`: List of URLs from the search results.
