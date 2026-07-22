---
title: "Linkup"
id: integrations-linkup
description: "Linkup integration for Haystack"
slug: "/integrations-linkup"
---


## haystack_integrations.components.websearch.linkup.linkup_websearch

### LinkupWebSearch

A component that uses Linkup to search the web and return results as Haystack Documents.

This component wraps the Linkup Search API, enabling web search queries that return
structured documents with content and links.

Linkup is a web search API optimized for LLM applications. You need a Linkup API key
from [linkup.so](https://www.linkup.so).

### Usage example

```python
from haystack_integrations.components.websearch.linkup import LinkupWebSearch
from haystack.utils import Secret

websearch = LinkupWebSearch(
    api_key=Secret.from_env_var("LINKUP_API_KEY"),
    top_k=5,
)
result = websearch.run(query="What is Haystack by deepset?")
documents = result["documents"]
links = result["links"]
```

#### __init__

```python
__init__(
    api_key: Secret = Secret.from_env_var("LINKUP_API_KEY"),
    top_k: int | None = 10,
    depth: Literal["fast", "standard", "deep"] = "standard",
    search_params: dict[str, Any] | None = None,
) -> None
```

Initialize the LinkupWebSearch component.

**Parameters:**

- **api_key** (<code>Secret</code>) – API key for Linkup. Defaults to the `LINKUP_API_KEY` environment variable.
- **top_k** (<code>int | None</code>) – Maximum number of results to return. Maps to the `max_results` parameter of the Linkup API.
- **depth** (<code>Literal['fast', 'standard', 'deep']</code>) – The depth of the search. Can be `"fast"` (beta, sub-second, keyword-based queries only),
  `"standard"` for a simple search, or `"deep"` for a more powerful agentic workflow.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Additional parameters passed to the Linkup search API.
  See the [Linkup API reference](https://docs.linkup.so/pages/documentation/api-reference/endpoint/post-search)
  for available options. Supported keys include: `include_images`, `from_date`, `to_date`,
  `include_domains`, `exclude_domains`.

#### warm_up

```python
warm_up() -> None
```

Initialize the Linkup client.

Called automatically on first use. Can be called explicitly to avoid cold-start latency.

#### run

```python
run(
    query: str,
    top_k: int | None = None,
    depth: Literal["fast", "standard", "deep"] | None = None,
    search_params: dict[str, Any] | None = None,
) -> dict[str, Any]
```

Search the web using Linkup and return results as Documents.

**Parameters:**

- **query** (<code>str</code>) – Search query string.
- **top_k** (<code>int | None</code>) – Optional per-run override of the maximum number of results.
  If not provided, the init-time `top_k` is used.
- **depth** (<code>Literal['fast', 'standard', 'deep'] | None</code>) – Optional per-run override of the search depth.
  If not provided, the init-time `depth` is used.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Optional per-run override of search parameters.
  If provided, fully replaces the init-time `search_params`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with:
- `documents`: List of Documents containing search result content.
- `links`: List of URLs from the search results.

#### run_async

```python
run_async(
    query: str,
    top_k: int | None = None,
    depth: Literal["fast", "standard", "deep"] | None = None,
    search_params: dict[str, Any] | None = None,
) -> dict[str, Any]
```

Asynchronously search the web using Linkup and return results as Documents.

**Parameters:**

- **query** (<code>str</code>) – Search query string.
- **top_k** (<code>int | None</code>) – Optional per-run override of the maximum number of results.
  If not provided, the init-time `top_k` is used.
- **depth** (<code>Literal['fast', 'standard', 'deep'] | None</code>) – Optional per-run override of the search depth.
  If not provided, the init-time `depth` is used.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Optional per-run override of search parameters.
  If provided, fully replaces the init-time `search_params`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with:
- `documents`: List of Documents containing search result content.
- `links`: List of URLs from the search results.
