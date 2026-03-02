---
title: "Firecrawl"
id: integrations-firecrawl
description: "Firecrawl integration for Haystack"
slug: "/integrations-firecrawl"
---


## haystack_integrations.components.fetchers.firecrawl.firecrawl_crawler

### FirecrawlCrawler

A component that uses Firecrawl to crawl one or more URLs and return the content as Haystack Documents.

Crawling starts from each given URL and follows links to discover subpages, up to a configurable limit.
This is useful for ingesting entire websites or documentation sites, not just single pages.

Firecrawl is a service that crawls websites and returns content in a structured format (e.g. Markdown)
suitable for LLMs. You need a Firecrawl API key from [firecrawl.dev](https://firecrawl.dev).

### Usage example

```python
from haystack_integrations.components.fetchers.firecrawl import FirecrawlFetcher

fetcher = FirecrawlFetcher(
    api_key=Secret.from_env_var("FIRECRAWL_API_KEY"),
    params={"limit": 5},
)
fetcher.warm_up()

result = fetcher.run(urls=["https://docs.haystack.deepset.ai/docs/intro"])
documents = result["documents"]
```

#### __init__

```python
__init__(
    api_key: Secret = Secret.from_env_var("FIRECRAWL_API_KEY"),
    params: dict[str, Any] | None = None,
) -> None
```

Initialize the FirecrawlFetcher.

**Parameters:**

- **api_key** (<code>Secret</code>) – API key for Firecrawl.
  Defaults to the `FIRECRAWL_API_KEY` environment variable.
- **params** (<code>dict\[str, Any\] | None</code>) – Parameters for the crawl request. See the
  [Firecrawl API reference](https://docs.firecrawl.dev/api-reference/endpoint/crawl-post)
  for available parameters.
  Defaults to `{"limit": 1, "scrape_options": {"formats": ["markdown"]}}`.
  Without a limit, Firecrawl may crawl all subpages and consume credits quickly.

#### run

```python
run(urls: list[str], params: dict[str, Any] | None = None) -> dict[str, Any]
```

Crawls the given URLs and returns the extracted content as Documents.

**Parameters:**

- **urls** (<code>list\[str\]</code>) – List of URLs to crawl.
- **params** (<code>dict\[str, Any\] | None</code>) – Optional override of crawl parameters for this run.
  If provided, fully replaces the init-time params.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: List of documents, one for each URL crawled.

#### run_async

```python
run_async(
    urls: list[str], params: dict[str, Any] | None = None
) -> dict[str, Any]
```

Asynchronously crawls the given URLs and returns the extracted content as Documents.

**Parameters:**

- **urls** (<code>list\[str\]</code>) – List of URLs to crawl.
- **params** (<code>dict\[str, Any\] | None</code>) – Optional override of crawl parameters for this run.
  If provided, fully replaces the init-time params.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- `documents`: List of documents, one for each URL crawled.

#### warm_up

```python
warm_up() -> None
```

Warm up the Firecrawl client by initializing the clients.
This is useful to avoid cold start delays when crawling many URLs.
