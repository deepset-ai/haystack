---
title: "You.com Search"
id: integrations-youcom
description: "You.com Search integration for Haystack"
slug: "/integrations-youcom"
---


## haystack_integrations.components.websearch.youcom.youcom_websearch

### YouComError

Bases: <code>ComponentError</code>

An error occurred while querying the You.com Search API.

### YouComWebSearch

A component that uses the You.com Search API to search the web and return results as Haystack Documents.

Works with zero configuration: when no API key is available, searches use You.com's
[keyless free tier](https://you.com/docs/api-reference/search/v1-agents-search) (rate limited
per IP), so getting-started pipelines run without any setup. Set the `YOUDOTCOM_API_KEY`
environment variable (or pass `api_key`) to use the keyed
[You.com Search API](https://you.com/docs/api-reference/search/v1-search) with higher limits.

Pass `keyless_fallback=False` to require a key and fail fast instead of degrading to the
keyless tier — useful in production pipelines where a missing key should surface as an error.

### Usage example

```python
from haystack_integrations.components.websearch.youcom import YouComWebSearch

websearch = YouComWebSearch(top_k=5)  # no API key needed to get started
result = websearch.run(query="What is Haystack by deepset?")
documents = result["documents"]
links = result["links"]
```

#### __init__

```python
__init__(
    api_key: Secret = Secret.from_env_var(API_KEY_ENV_VAR, strict=False),
    keyless_fallback: bool = True,
    top_k: int | None = 10,
    freshness: str | None = None,
    country: str | None = None,
    search_lang: str | None = None,
    safesearch: str | None = None,
    extra_params: dict[str, Any] | None = None,
    timeout: int = 10,
    max_retries: int = 3,
) -> None
```

Initialize the YouComWebSearch component.

**Parameters:**

- **api_key** (<code>Secret</code>) – You.com API key. Defaults to the `YOUDOTCOM_API_KEY` environment variable. Resolved
  leniently, so an unset key is not an error — see `keyless_fallback` for what happens then.
- **keyless_fallback** (<code>bool</code>) – What to do when no API key resolves. When `True` (the default), search the
  [keyless free tier](https://you.com/docs/api-reference/search/v1-agents-search),
  which needs no credentials but is rate limited per IP; the component logs which
  endpoint it selected. When `False`, raise `YouComError` instead, so a missing key
  fails fast rather than silently degrading.
- **top_k** (<code>int | None</code>) – Maximum number of results to return per section (web, news). Maps to the
  `count` parameter in the You.com API (1-100).
- **freshness** (<code>str | None</code>) – Only return results from within the given window: `"day"`, `"week"`, `"month"`,
  `"year"`, or a date range in the format `"YYYY-MM-DDtoYYYY-MM-DD"`.
- **country** (<code>str | None</code>) – 2-letter country code determining the geographical focus of web results (e.g. `"US"`, `"DE"`).
- **search_lang** (<code>str | None</code>) – Language of the returned web results in BCP 47 format (e.g. `"EN"`, `"PT-BR"`).
  Maps to the `language` parameter in the You.com API.
- **safesearch** (<code>str | None</code>) – Content moderation level: `"off"`, `"moderate"`, or `"strict"`.
- **extra_params** (<code>dict\[str, Any\] | None</code>) – Additional query parameters passed directly to the You.com Search API
  (e.g. `{"include_domains": "nytimes.com,bbc.com"}`).
- **timeout** (<code>int</code>) – Timeout in seconds for the HTTP request. Defaults to 10.
- **max_retries** (<code>int</code>) – Maximum number of retry attempts on transient failures. Defaults to 3.

#### run

```python
run(query: str, top_k: int | None = None) -> dict[str, Any]
```

Search the web using the You.com Search API and return results as Documents.

**Parameters:**

- **query** (<code>str</code>) – Search query string.
- **top_k** (<code>int | None</code>) – Optional per-run override of the maximum number of results.
  If not provided, the init-time `top_k` is used.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with:
- `documents`: List of Documents containing search result content.
- `links`: List of URLs from the search results.

**Raises:**

- <code>YouComError</code> – If the You.com Search API request fails.

#### run_async

```python
run_async(query: str, top_k: int | None = None) -> dict[str, Any]
```

Asynchronously search the web using the You.com Search API and return results as Documents.

**Parameters:**

- **query** (<code>str</code>) – Search query string.
- **top_k** (<code>int | None</code>) – Optional per-run override of the maximum number of results.
  If not provided, the init-time `top_k` is used.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with:
- `documents`: List of Documents containing search result content.
- `links`: List of URLs from the search results.

**Raises:**

- <code>YouComError</code> – If the You.com Search API request fails.
