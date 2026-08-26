---
title: "Microsoft SharePoint"
id: integrations-microsoft-sharepoint
description: "Microsoft SharePoint integration for Haystack"
slug: "/integrations-microsoft-sharepoint"
---


## haystack_integrations.components.fetchers.microsoft_sharepoint.fetcher

### MSSharePointFetcher

Fetches the full content of Microsoft SharePoint and OneDrive items via the Microsoft Graph API.

The fetcher complements `MSSharePointRetriever`, which only returns Search snippets and metadata. Wire the
retriever's `documents` (or a list of `web_url`s) into this fetcher to download the full content. It
dispatches on the entity type of each hit and always returns `ByteStream`s, ready for a downstream converter
(for example a `FileTypeRouter` in front of `PyPDFToDocument`, `DOCXToDocument`, `HTMLToDocument`, or a JSON
converter):

- **Files** (`driveItem`) are downloaded as their raw bytes (PDF, DOCX, ...).
- **List items** (`listItem`) are returned as a JSON `ByteStream` of the item's column values (`fields`).
- **SharePoint pages** (`sitePage`) are returned as an HTML `ByteStream` built from the page's web parts.

Each `ByteStream`'s `meta` carries `url`, `file_name`, `content_type`, and a normalized `entity_type`
(`driveItem`, `listItem`, or `sitePage`).

Everything is resolved through the Microsoft Graph `shares` endpoint (plus the Pages API for pages), so only
the `web_url` already exposed by the retriever is needed. The fetcher takes a per-user `access_token` as a run
input, typically wired from an upstream `OAuthTokenResolver`. The token must carry delegated Microsoft Graph
permissions (for example `Files.Read.All` for files and `Sites.Read.All` for list items and pages).

### Usage example

```python
from haystack_integrations.components.fetchers.microsoft_sharepoint import MSSharePointFetcher

fetcher = MSSharePointFetcher()

# `access_token` is a per-user delegated Microsoft Graph bearer token.
result = fetcher.run(
    access_token="my-delegated-graph-token",
    targets=["https://contoso.sharepoint.com/sites/contoso-team/contoso-designs.docx"],
)
streams = result["streams"]
```

In a pipeline, connect `MSSharePointRetriever.documents` to the fetcher's `targets` input and an upstream
component that emits a per-user `access_token` to the fetcher's `access_token` input.

#### __init__

```python
__init__(
    *,
    graph_url: str = DEFAULT_GRAPH_URL,
    timeout: float = 30.0,
    max_retries: int = 3,
    max_concurrent_requests: int = 5,
    raise_on_failure: bool = True
) -> None
```

Initialize the fetcher.

**Parameters:**

- **graph_url** (<code>str</code>) – The Microsoft Graph base URL. Defaults to `https://graph.microsoft.com/v1.0`.
  Override for sovereign clouds.
- **timeout** (<code>float</code>) – The HTTP timeout in seconds for each request to Microsoft Graph.
- **max_retries** (<code>int</code>) – The maximum number of retries for throttled (HTTP 429) or transient server errors.
- **max_concurrent_requests** (<code>int</code>) – The maximum number of items fetched concurrently by `run_async`. Bounds
  the in-flight requests to Microsoft Graph to avoid tripping its rate limits. Has no effect on the
  synchronous `run`, which fetches items one at a time.
- **raise_on_failure** (<code>bool</code>) – If `True`, a fetch failure raises an exception. If `False`, the failure is
  logged and the item is skipped, so the other items are still returned.

**Raises:**

- <code>SharePointConfigError</code> – If `max_retries` is negative or `max_concurrent_requests` is not positive.

#### run

```python
run(
    access_token: str | Secret, targets: list[Document | str]
) -> dict[str, list[ByteStream]]
```

Fetch the content of SharePoint and OneDrive items and return them as `ByteStream`s.

**Parameters:**

- **access_token** (<code>str | Secret</code>) – A delegated Microsoft Graph bearer token for the user whose content is fetched,
  typically wired from an upstream `OAuthTokenResolver` (which emits a plain `str`). A `Secret` is also
  accepted and resolved internally.
- **targets** (<code>list\[Document | str\]</code>) – The items to fetch, as either `Document`s emitted by `MSSharePointRetriever` or raw
  SharePoint/OneDrive `web_url` strings (the two may also be mixed in one list). For a `Document`, the
  `web_url` in its meta is fetched and `file_name`, `mime_type`, `entity_type`, and the SharePoint IDs
  are reused when present; container hits with no extractable content (for example `site` or `list`) are
  skipped. For a raw URL, the item is probed as a file and falls back to a list item.

**Returns:**

- <code>dict\[str, list\[ByteStream\]\]</code> – A dictionary with a `streams` key holding the fetched content as `ByteStream` objects. Each
  stream's `meta` carries `url`, `file_name`, `content_type`, and `entity_type`.

**Raises:**

- <code>SharePointConfigError</code> – If an item is neither a `Document` nor a `str`, or if `access_token` is a
  `Secret` that does not resolve to a string.
- <code>SharePointRequestError</code> – If a fetch fails and `raise_on_failure` is `True`.

#### run_async

```python
run_async(
    access_token: str | Secret, targets: list[Document | str]
) -> dict[str, list[ByteStream]]
```

Asynchronously fetch the content of SharePoint and OneDrive items and return them as `ByteStream`s.

**Parameters:**

- **access_token** (<code>str | Secret</code>) – A delegated Microsoft Graph bearer token for the user whose content is fetched,
  typically wired from an upstream `OAuthTokenResolver` (which emits a plain `str`). A `Secret` is also
  accepted and resolved internally.
- **targets** (<code>list\[Document | str\]</code>) – The items to fetch, as either `Document`s emitted by `MSSharePointRetriever` or raw
  SharePoint/OneDrive `web_url` strings (the two may also be mixed in one list). For a `Document`, the
  `web_url` in its meta is fetched and `file_name`, `mime_type`, `entity_type`, and the SharePoint IDs
  are reused when present; container hits with no extractable content (for example `site` or `list`) are
  skipped. For a raw URL, the item is probed as a file and falls back to a list item.

**Returns:**

- <code>dict\[str, list\[ByteStream\]\]</code> – A dictionary with a `streams` key holding the fetched content as `ByteStream` objects. Each
  stream's `meta` carries `url`, `file_name`, `content_type`, and `entity_type`.

**Raises:**

- <code>SharePointConfigError</code> – If an item is neither a `Document` nor a `str`, or if `access_token` is a
  `Secret` that does not resolve to a string.
- <code>SharePointRequestError</code> – If a fetch fails and `raise_on_failure` is `True`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> MSSharePointFetcher
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary representation of this component.

**Returns:**

- <code>MSSharePointFetcher</code> – The deserialized component instance.

## haystack_integrations.components.retrievers.microsoft_sharepoint.retriever

### MSSharePointRetriever

Retrieves content from Microsoft SharePoint and OneDrive via the Microsoft Search (Graph) API.

Given a query, the retriever calls `POST /search/query` and maps each hit to a Haystack `Document`
whose `content` is the search snippet and whose `meta` carries the resource metadata (`file_name`,
`web_url`, `entity_type`, `created_date_time`, `last_modified_date_time`, `created_by`, `last_modified_by`,
`mime_type`, and `file_extension`), plus the SharePoint identifiers a downstream fetcher needs to read
list items and pages by ID (`site_id`, `list_id`, `list_item_id`, `list_item_unique_id`). It does not
download or convert the underlying files. Compose a downstream fetcher/converter (such as
`MSSharePointFetcher`) when full content is needed.

The retriever takes a per-user `access_token` as a run input, typically wired
from an upstream `OAuthResolver`. The token must carry delegated Microsoft Graph permissions
(for example `Files.Read.All` and, for site/list scoping, `Sites.Read.All`). The Search API supports
delegated permissions only.

### Usage example

```python
from haystack_integrations.components.retrievers.microsoft_sharepoint import (
    MSSharePointRetriever,
)

retriever = MSSharePointRetriever(top_k=5)

# `access_token` is a per-user delegated Microsoft Graph bearer token.
result = retriever.run(
    query="quarterly roadmap", access_token="my-delegated-graph-token"
)
documents = result["documents"]
```

In a pipeline, connect an upstream component that emits a per-user `access_token` to the retriever's
`access_token` input. See the integration documentation for a full example that obtains the token from
an OAuth provider.

#### __init__

```python
__init__(
    *,
    entity_types: list[str] | None = None,
    top_k: int = 10,
    fields: list[str] | None = None,
    query_template: str | None = None,
    graph_url: str = DEFAULT_GRAPH_URL,
    timeout: float = 30.0,
    max_retries: int = 3
) -> None
```

Initialize the retriever.

**Parameters:**

- **entity_types** (<code>list\[str\] | None</code>) – The Microsoft Search entity types to query. Defaults to `["driveItem", "listItem"]`,
  which covers files, folders, SharePoint pages and news, and list items. Other valid values are
  `"list"` and `"site"`. See the supported values and combinations in the
  [Microsoft docs](https://learn.microsoft.com/en-us/graph/api/resources/searchrequest).
- **top_k** (<code>int</code>) – The maximum number of documents to return. Maps to the Search API `size` and is paginated
  when it exceeds a single page.
- **fields** (<code>list\[str\] | None</code>) – Optional list of resource properties to request via the Search API `fields` selection
  (only honored for `listItem` and `driveItem` entity types). See
  [Get selected properties](https://learn.microsoft.com/en-us/graph/api/resources/search-api-overview#get-selected-properties).
- **query_template** (<code>str | None</code>) – Optional query template used to scope the search, for example
  `'{searchTerms} path:"https://contoso.sharepoint.com/sites/Team"'`. The literal `{searchTerms}`
  placeholder is replaced by the run-time query. The template uses
  [Keyword Query Language (KQL)](https://learn.microsoft.com/en-us/sharepoint/dev/general-development/keyword-query-language-kql-syntax-reference).
- **graph_url** (<code>str</code>) – The Microsoft Graph base URL. Defaults to `https://graph.microsoft.com/v1.0`.
  Override for sovereign clouds.
- **timeout** (<code>float</code>) – The HTTP timeout in seconds for each request to Microsoft Graph.
- **max_retries** (<code>int</code>) – The maximum number of retries for throttled (HTTP 429) or transient server errors.

**Raises:**

- <code>SharePointConfigError</code> – If `entity_types` is empty, `top_k` is not positive, or `max_retries` is
  negative.

#### run

```python
run(
    query: str, access_token: str | Secret, top_k: int | None = None
) -> dict[str, list[Document]]
```

Search SharePoint and OneDrive and return the matching documents.

**Parameters:**

- **query** (<code>str</code>) – The search query string. Filter results by embedding Keyword Query Language (KQL)
  operators directly in the query, for example `filetype:docx`, `author:"Jane Doe"`, or
  `path:"https://contoso.sharepoint.com/sites/Team"`. See the
  [KQL syntax reference](https://learn.microsoft.com/en-us/sharepoint/dev/general-development/keyword-query-language-kql-syntax-reference).
- **access_token** (<code>str | Secret</code>) – A delegated Microsoft Graph bearer token for the user whose content is searched,
  typically wired from an upstream `OAuthResolver` (which emits a plain `str`). A `Secret` is also
  accepted and resolved internally.
- **top_k** (<code>int | None</code>) – Overrides the `top_k` configured at initialization for this run.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with a `documents` key holding the list of retrieved `Document` objects.

**Raises:**

- <code>SharePointConfigError</code> – If `access_token` is a `Secret` that does not resolve to a string.
- <code>SharePointRequestError</code> – If Microsoft Graph returns an error response.

#### run_async

```python
run_async(
    query: str, access_token: str | Secret, top_k: int | None = None
) -> dict[str, list[Document]]
```

Asynchronously search SharePoint and OneDrive and return the matching documents.

**Parameters:**

- **query** (<code>str</code>) – The search query string. Filter results by embedding Keyword Query Language (KQL)
  operators directly in the query, for example `filetype:docx`, `author:"Jane Doe"`, or
  `path:"https://contoso.sharepoint.com/sites/Team"`. See the
  [KQL syntax reference](https://learn.microsoft.com/en-us/sharepoint/dev/general-development/keyword-query-language-kql-syntax-reference).
- **access_token** (<code>str | Secret</code>) – A delegated Microsoft Graph bearer token for the user whose content is searched,
  typically wired from an upstream `OAuthResolver` (which emits a plain `str`). A `Secret` is also
  accepted and resolved internally.
- **top_k** (<code>int | None</code>) – Overrides the `top_k` configured at initialization for this run.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with a `documents` key holding the list of retrieved `Document` objects.

**Raises:**

- <code>SharePointConfigError</code> – If `access_token` is a `Secret` that does not resolve to a string.
- <code>SharePointRequestError</code> – If Microsoft Graph returns an error response.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> MSSharePointRetriever
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary representation of this component.

**Returns:**

- <code>MSSharePointRetriever</code> – The deserialized component instance.
