---
title: "ddgs"
id: integrations-ddgs
description: "ddgs (Dux Distributed Global Search) integration for Haystack"
slug: "/integrations-ddgs"
---


## haystack_integrations.components.websearch.ddgs.ddgs_websearch

### DDGSWebSearch

Searches the web with ddgs (Dux Distributed Global Search) and returns results as Haystack Documents.

[ddgs](https://github.com/deedy5/ddgs) is a free, **keyless** metasearch library that aggregates
results from multiple backends (DuckDuckGo, Google, Bing, Brave, Yahoo, Yandex, Mullvad, and more),
so no API key is required.

### Usage example

```python
from haystack_integrations.components.websearch.ddgs import DDGSWebSearch

websearch = DDGSWebSearch(top_k=5)
result = websearch.run(query="What is Haystack by deepset?")

documents = result["documents"]
links = result["links"]
```

#### __init__

```python
__init__(
    top_k: int = 10,
    backend: str = "auto",
    region: str = "us-en",
    safesearch: str = "moderate",
    search_params: dict[str, Any] | None = None,
) -> None
```

Initialize the DDGSWebSearch component.

**Parameters:**

- **top_k** (<code>int</code>) – Maximum number of results to return.
- **backend** (<code>str</code>) – Comma-separated ddgs backends to query, or `"auto"` to let ddgs choose
  (for example `"duckduckgo, google, brave"`). See the ddgs docs for the full list.
- **region** (<code>str</code>) – Region/locale for the search, for example `"us-en"`, `"de-de"`, or `"wt-wt"` (no region).
- **safesearch** (<code>str</code>) – Safe-search level: `"on"`, `"moderate"`, or `"off"`.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments forwarded to `DDGS().text()` (for example `page` or
  `timelimit`). Values here override `backend`, `region`, `safesearch`, and `top_k`
  on conflict.

#### warm_up

```python
warm_up() -> None
```

Initialize the ddgs client.

Called automatically on first use. Can be called explicitly to avoid cold-start latency.

#### run

```python
run(query: str) -> dict[str, list[Document] | list[str]]
```

Use ddgs to search the web.

**Parameters:**

- **query** (<code>str</code>) – Search query.

**Returns:**

- <code>dict\[str, list\[Document\] | list\[str\]\]</code> – A dictionary with the following keys:
- `documents`: List of documents returned by the search backends.
- `links`: List of links returned by the search backends.

#### run_async

```python
run_async(query: str) -> dict[str, list[Document] | list[str]]
```

Asynchronously use ddgs to search the web.

ddgs has no native async API, so the blocking search runs in a worker thread. Same parameters
and return values as :meth:`run`.

**Parameters:**

- **query** (<code>str</code>) – Search query.

**Returns:**

- <code>dict\[str, list\[Document\] | list\[str\]\]</code> – A dictionary with `documents` and `links` keys.
