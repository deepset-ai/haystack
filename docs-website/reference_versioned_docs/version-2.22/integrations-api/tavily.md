---
title: "Tavily"
id: integrations-tavily
description: "Tavily integration for Haystack"
slug: "/integrations-tavily"
---


## haystack_integrations.components.fetchers.tavily.tavily_fetcher

### TavilyFetcher

A component that uses the Tavily Extract API to fetch and extract content from URLs as Haystack Documents.

This component wraps the Tavily Extract API, which retrieves and parses web page content from
one or more specified URLs. Unlike web search, it fetches content directly from the given URLs
rather than discovering them via a query. PDF URLs are also supported for extraction.

Tavily is an AI-powered search and extraction API optimized for LLM applications. You need a Tavily
API key from [tavily.com](https://tavily.com).

### Usage example

```python
from haystack_integrations.components.fetchers.tavily import TavilyFetcher
from haystack.utils import Secret

fetcher = TavilyFetcher(
    api_key=Secret.from_env_var("TAVILY_API_KEY"),
    extract_depth="basic",
)
result = fetcher.run(urls=["https://haystack.deepset.ai"])
documents = result["documents"]
meta = result["meta"]
```

#### __init__

```python
__init__(
    api_key: Secret = Secret.from_env_var("TAVILY_API_KEY"),
    *,
    extract_depth: Literal["basic", "advanced"] = "basic",
    include_images: bool = False,
    extract_params: dict[str, Any] | None = None
) -> None
```

Initialize the TavilyFetcher component.

**Parameters:**

- **api_key** (<code>Secret</code>) – API key for Tavily. Defaults to the `TAVILY_API_KEY` environment variable.
- **extract_depth** (<code>Literal['basic', 'advanced']</code>) – Extraction depth: `"basic"` (fast, lower cost) or `"advanced"` (more data including
  tables, higher latency and cost). Defaults to `"basic"`.
- **include_images** (<code>bool</code>) – If `True`, extracted image URLs are included in each Document's metadata under
  the `"images"` key. Defaults to `False`.
- **extract_params** (<code>dict\[str, Any\] | None</code>) – Additional parameters passed to the Tavily Extract API, such as `format`,
  `include_favicon`, `query`, or `chunks_per_source`.
  See the [Tavily Extract API reference](https://docs.tavily.com/documentation/api-reference/endpoint/extract)
  for available options.

#### warm_up

```python
warm_up() -> None
```

Initialize the Tavily sync and async clients.

Called automatically on first use. Can be called explicitly to avoid cold-start latency.

#### run

```python
run(
    urls: list[str], extract_params: dict[str, Any] | None = None
) -> dict[str, Any]
```

Fetch and extract content from the given URLs using the Tavily Extract API.

**Parameters:**

- **urls** (<code>list\[str\]</code>) – List of URLs to extract content from. Maximum 20 URLs per request.
- **extract_params** (<code>dict\[str, Any\] | None</code>) – Optional per-run override of extract parameters.
  If provided, fully replaces the init-time `extract_params`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with:
- `documents`: List of Documents containing extracted page content.
  Each Document's `meta` includes `"url"` and, if `include_images` is True, `"images"`.
- `meta`: Request-level metadata containing `"response_time"`, `"usage"`,
  `"request_id"`, and `"failed_results"` for URLs that could not be processed.

#### run_async

```python
run_async(
    urls: list[str], extract_params: dict[str, Any] | None = None
) -> dict[str, Any]
```

Asynchronously fetch and extract content from the given URLs using the Tavily Extract API.

**Parameters:**

- **urls** (<code>list\[str\]</code>) – List of URLs to extract content from. Maximum 20 URLs per request.
- **extract_params** (<code>dict\[str, Any\] | None</code>) – Optional per-run override of extract parameters.
  If provided, fully replaces the init-time `extract_params`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with:
- `documents`: List of Documents containing extracted page content.
  Each Document's `meta` includes `"url"` and, if `include_images` is True, `"images"`.
- `meta`: Request-level metadata containing `"response_time"`, `"usage"`,
  `"request_id"`, and `"failed_results"` for URLs that could not be processed.

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
