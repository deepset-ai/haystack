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
run(
    query: str,
    top_k: int | None = None,
    *,
    backend: str | None = None,
    region: str | None = None,
    safesearch: str | None = None,
    search_params: dict[str, Any] | None = None
) -> dict[str, list[Document] | list[str]]
```

Use ddgs to search the web.

**Parameters:**

- **query** (<code>str</code>) – Search query.
- **top_k** (<code>int | None</code>) – Optional per-run override of the maximum number of results. If not provided, the
  init-time `top_k` is used.
- **backend** (<code>str | None</code>) – Optional per-run override of the ddgs backends. If not provided, the init-time
  `backend` is used.
- **region** (<code>str | None</code>) – Optional per-run override of the region/locale. If not provided, the init-time
  `region` is used.
- **safesearch** (<code>str | None</code>) – Optional per-run override of the safe-search level. If not provided, the init-time
  `safesearch` is used.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Optional per-run override of the extra `DDGS().text()` arguments. If provided, fully
  replaces the init-time `search_params`.

**Returns:**

- <code>dict\[str, list\[Document\] | list\[str\]\]</code> – A dictionary with the following keys:
- `documents`: List of documents returned by the search backends.
- `links`: List of links returned by the search backends.

#### run_async

```python
run_async(
    query: str,
    top_k: int | None = None,
    *,
    backend: str | None = None,
    region: str | None = None,
    safesearch: str | None = None,
    search_params: dict[str, Any] | None = None
) -> dict[str, list[Document] | list[str]]
```

Asynchronously use ddgs to search the web.

ddgs has no native async API, so the blocking search runs in a worker thread. Same parameters
and return values as :meth:`run`.

**Parameters:**

- **query** (<code>str</code>) – Search query.
- **top_k** (<code>int | None</code>) – Optional per-run override of the maximum number of results. If not provided, the
  init-time `top_k` is used.
- **backend** (<code>str | None</code>) – Optional per-run override of the ddgs backends. If not provided, the init-time
  `backend` is used.
- **region** (<code>str | None</code>) – Optional per-run override of the region/locale. If not provided, the init-time
  `region` is used.
- **safesearch** (<code>str | None</code>) – Optional per-run override of the safe-search level. If not provided, the init-time
  `safesearch` is used.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Optional per-run override of the extra `DDGS().text()` arguments. If provided, fully
  replaces the init-time `search_params`.

**Returns:**

- <code>dict\[str, list\[Document\] | list\[str\]\]</code> – A dictionary with `documents` and `links` keys.
