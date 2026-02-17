---
title: "Fetchers"
id: fetchers-api
description: "Fetches content from a list of URLs and returns a list of extracted content streams."
slug: "/fetchers-api"
---


## `LinkContentFetcher`

Fetches and extracts content from URLs.

It supports various content types, retries on failures, and automatic user-agent rotation for failed web
requests. Use it as the data-fetching step in your pipelines.

You may need to convert LinkContentFetcher's output into a list of documents. Use HTMLToDocument
converter to do this.

### Usage example

```python
from haystack.components.fetchers.link_content import LinkContentFetcher

fetcher = LinkContentFetcher()
streams = fetcher.run(urls=["https://www.google.com"])["streams"]

assert len(streams) == 1
assert streams[0].meta == {'content_type': 'text/html', 'url': 'https://www.google.com'}
assert streams[0].data
```

For async usage:

```python
import asyncio
from haystack.components.fetchers import LinkContentFetcher

async def fetch_async():
    fetcher = LinkContentFetcher()
    result = await fetcher.run_async(urls=["https://www.google.com"])
    return result["streams"]

streams = asyncio.run(fetch_async())
```

### `__init__`

```python
__init__(raise_on_failure: bool = True, user_agents: list[str] | None = None, retry_attempts: int = 2, timeout: int = 3, http2: bool = False, client_kwargs: dict | None = None, request_headers: dict[str, str] | None = None)
```

Initializes the component.

**Parameters:**

- **raise_on_failure** (<code>bool</code>) – If `True`, raises an exception if it fails to fetch a single URL.
  For multiple URLs, it logs errors and returns the content it successfully fetched.
- **user_agents** (<code>list\[str\] | None</code>) – [User agents](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent)
  for fetching content. If `None`, a default user agent is used.
- **retry_attempts** (<code>int</code>) – The number of times to retry to fetch the URL's content.
- **timeout** (<code>int</code>) – Timeout in seconds for the request.
- **http2** (<code>bool</code>) – Whether to enable HTTP/2 support for requests. Defaults to False.
  Requires the 'h2' package to be installed (via `pip install httpx[http2]`).
- **client_kwargs** (<code>dict | None</code>) – Additional keyword arguments to pass to the httpx client.
  If `None`, default values are used.

### `run`

```python
run(urls: list[str])
```

Fetches content from a list of URLs and returns a list of extracted content streams.

Each content stream is a `ByteStream` object containing the extracted content as binary data.
Each ByteStream object in the returned list corresponds to the contents of a single URL.
The content type of each stream is stored in the metadata of the ByteStream object under
the key "content_type". The URL of the fetched content is stored under the key "url".

**Parameters:**

- **urls** (<code>list\[str\]</code>) – A list of URLs to fetch content from.

**Returns:**

- – `ByteStream` objects representing the extracted content.

**Raises:**

- <code>Exception</code> – If the provided list of URLs contains only a single URL, and `raise_on_failure` is set to
  `True`, an exception will be raised in case of an error during content retrieval.
  In all other scenarios, any retrieval errors are logged, and a list of successfully retrieved `ByteStream`
  objects is returned.

### `run_async`

```python
run_async(urls: list[str])
```

Asynchronously fetches content from a list of URLs and returns a list of extracted content streams.

This is the asynchronous version of the `run` method with the same parameters and return values.

**Parameters:**

- **urls** (<code>list\[str\]</code>) – A list of URLs to fetch content from.

**Returns:**

- – `ByteStream` objects representing the extracted content.
