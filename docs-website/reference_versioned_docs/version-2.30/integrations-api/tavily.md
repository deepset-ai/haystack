---
title: "Tavily"
id: integrations-tavily
description: "Tavily integration for Haystack"
slug: "/integrations-tavily"
---


## haystack_integrations.components.websearch.tavily.tavily_websearch

### TavilyWebSearch

A component that uses Tavily to search the web and return results as Haystack Documents.

This component wraps the Tavily Search API, enabling web search queries that return
structured documents with content and links.

Tavily is an AI-powered search API optimized for LLM applications. You need a Tavily
API key from [tavily.com](https://tavily.com).

### Usage example

```python
from haystack_integrations.components.websearch.tavily import TavilyWebSearch
from haystack.utils import Secret

websearch = TavilyWebSearch(
    api_key=Secret.from_env_var("TAVILY_API_KEY"),
    top_k=5,
)
result = websearch.run(query="What is Haystack by deepset?")
documents = result["documents"]
links = result["links"]
```

#### __init__

```python
__init__(
    api_key: Secret = Secret.from_env_var("TAVILY_API_KEY"),
    top_k: int | None = 10,
    search_params: dict[str, Any] | None = None,
) -> None
```

Initialize the TavilyWebSearch component.

**Parameters:**

- **api_key** (<code>Secret</code>) – API key for Tavily. Defaults to the `TAVILY_API_KEY` environment variable.
- **top_k** (<code>int | None</code>) – Maximum number of results to return.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Additional parameters passed to the Tavily search API.
  See the [Tavily API reference](https://docs.tavily.com/docs/tavily-api/rest_api)
  for available options. Supported keys include: `search_depth`, `include_answer`,
  `include_raw_content`, `include_domains`, `exclude_domains`.

#### warm_up

```python
warm_up() -> None
```

Initialize the Tavily sync and async clients.

Called automatically on first use. Can be called explicitly to avoid cold-start latency.

#### run

```python
run(query: str, search_params: dict[str, Any] | None = None) -> dict[str, Any]
```

Search the web using Tavily and return results as Documents.

**Parameters:**

- **query** (<code>str</code>) – Search query string.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Optional per-run override of search parameters.
  If provided, fully replaces the init-time `search_params`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with:
- `documents`: List of Documents containing search result content.
- `links`: List of URLs from the search results.

#### run_async

```python
run_async(
    query: str, search_params: dict[str, Any] | None = None
) -> dict[str, Any]
```

Asynchronously search the web using Tavily and return results as Documents.

**Parameters:**

- **query** (<code>str</code>) – Search query string.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Optional per-run override of search parameters.
  If provided, fully replaces the init-time `search_params`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with:
- `documents`: List of Documents containing search result content.
- `links`: List of URLs from the search results.
