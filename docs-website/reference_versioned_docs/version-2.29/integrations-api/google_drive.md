---
title: "Google Drive"
id: integrations-google-drive
description: "Google Drive integration for Haystack"
slug: "/integrations-google-drive"
---


## haystack_integrations.components.retrievers.google_drive.retriever

### GoogleDriveRetriever

Retrieves files from Google Drive via the Drive API v3 search (`files.list`) endpoint.

Given a query, the retriever runs a full-text search over the user's Drive (and optionally shared
drives) and maps each matching file to a Haystack `Document`. By default, each `Document` carries
resource metadata (`file_name`, `file_id`, `web_url`, `mime_type`, `file_extension`, author, and
timestamps) and uses the file `description` or `name` as `content`, because the Drive search API does
not return a text snippet. Set `include_content=True` to additionally export native Google
Docs/Sheets/Slides to text and use that as the `Document` content. Binary files (PDF, DOCX, ...) are
never downloaded. Compose a downstream fetcher/converter on the returned `web_url`/`file_id` when full
file content is needed.

The retriever takes a per-user `access_token` as a run input, typically wired from an upstream
`OAuthResolver`. The token must carry a delegated Google OAuth scope that allows search
(for example `https://www.googleapis.com/auth/drive.readonly`). The metadata-only
`drive.metadata.readonly` scope cannot search file content or export documents.

### Usage example

```python
from haystack import Pipeline
from haystack.utils import Secret
from haystack_integrations.components.connectors.oauth import OAuthResolver
from haystack_integrations.utils.oauth import RefreshTokenSource
from haystack_integrations.components.retrievers.google_drive import GoogleDriveRetriever

pipeline = Pipeline()
pipeline.add_component(
    "resolver",
    OAuthResolver(
        token_source=RefreshTokenSource(
            token_url="https://oauth2.googleapis.com/token",
            client_id="aaa-bbb-ccc",
            refresh_token=Secret.from_env_var("GOOGLE_REFRESH_TOKEN"),
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        ),
    ),
)
pipeline.add_component("retriever", GoogleDriveRetriever(top_k=5))
pipeline.connect("resolver.access_token", "retriever.access_token")

result = pipeline.run({"retriever": {"query": "quarterly roadmap"}})
documents = result["retriever"]["documents"]
```

#### __init__

```python
__init__(
    *,
    include_content: bool = False,
    top_k: int = 10,
    query_filter: str | None = None,
    include_shared_drives: bool = False,
    order_by: str | None = None,
    fields: list[str] | None = None,
    api_base_url: str = _DEFAULT_API_BASE_URL,
    timeout: float = 30.0,
    max_retries: int = 3
) -> None
```

Initialize the retriever.

**Parameters:**

- **include_content** (<code>bool</code>) – When `True`, native Google Docs/Sheets/Slides are exported to text and the
  result becomes the `Document` content. Binary files are never downloaded. When `False` (the
  default), `content` is the file `description` or `name` and no export request is made.
- **top_k** (<code>int</code>) – The maximum number of documents to return. Maps to the Drive `pageSize` and is
  paginated when it exceeds a single page.
- **query_filter** (<code>str | None</code>) – Optional Drive query clause AND-ed with the full-text search term, for example
  `"mimeType != 'application/vnd.google-apps.folder'"` or `"'<folderId>' in parents"`.
- **include_shared_drives** (<code>bool</code>) – When `True`, the search spans shared drives as well as the user's My
  Drive (sets `includeItemsFromAllDrives`, `supportsAllDrives`, and `corpora=allDrives`).
- **order_by** (<code>str | None</code>) – Optional Drive `orderBy` expression, for example `"modifiedTime desc"`.
- **fields** (<code>list\[str\] | None</code>) – Optional list of file properties to request via the Drive `fields` selection.
  Defaults to a standard set covering the returned metadata.
- **api_base_url** (<code>str</code>) – The Drive API base URL. Defaults to `https://www.googleapis.com/drive/v3`.
- **timeout** (<code>float</code>) – The HTTP timeout in seconds for each request to the Drive API.
- **max_retries** (<code>int</code>) – The maximum number of retries on HTTP 429 (rate limit), 500, 502, 503,
  or 504 responses. Set to 0 to disable retries.

**Raises:**

- <code>GoogleDriveConfigError</code> – If `top_k` is not positive or `max_retries` is negative.

#### run

```python
run(
    query: str, access_token: str | Secret, top_k: int | None = None
) -> dict[str, list[Document]]
```

Search Google Drive and return the matching documents.

**Parameters:**

- **query** (<code>str</code>) – The search query string, matched against the full text of files via
  `fullText contains`.
- **access_token** (<code>str | Secret</code>) – A delegated Google OAuth bearer token for the user whose Drive is searched,
  typically wired from an upstream `OAuthResolver` (which emits a plain `str`). A `Secret` is also
  accepted and resolved internally.
- **top_k** (<code>int | None</code>) – Overrides the `top_k` configured at initialization for this run.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with a `documents` key holding the list of retrieved `Document` objects.

**Raises:**

- <code>GoogleDriveConfigError</code> – If `access_token` is a `Secret` that does not resolve to a string.
- <code>GoogleDriveRequestError</code> – If the Drive API returns an error response.
- <code>httpx.HTTPError</code> – If a network-level error occurs (for example a timeout or connection failure).

#### run_async

```python
run_async(
    query: str, access_token: str | Secret, top_k: int | None = None
) -> dict[str, list[Document]]
```

Asynchronously search Google Drive and return the matching documents.

**Parameters:**

- **query** (<code>str</code>) – The search query string, matched against the full text of files via
  `fullText contains`.
- **access_token** (<code>str | Secret</code>) – A delegated Google OAuth bearer token for the user whose Drive is searched,
  typically wired from an upstream `OAuthResolver` (which emits a plain `str`). A `Secret` is also
  accepted and resolved internally.
- **top_k** (<code>int | None</code>) – Overrides the `top_k` configured at initialization for this run.

**Returns:**

- <code>dict\[str, list\[Document\]\]</code> – A dictionary with a `documents` key holding the list of retrieved `Document` objects.

**Raises:**

- <code>GoogleDriveConfigError</code> – If `access_token` is a `Secret` that does not resolve to a string.
- <code>GoogleDriveRequestError</code> – If the Drive API returns an error response.
- <code>httpx.HTTPError</code> – If a network-level error occurs (for example a timeout or connection failure).

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> GoogleDriveRetriever
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary representation of this component.

**Returns:**

- <code>GoogleDriveRetriever</code> – The deserialized component instance.
