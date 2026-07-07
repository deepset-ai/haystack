---
title: "Google Drive"
id: integrations-google-drive
description: "Google Drive integration for Haystack"
slug: "/integrations-google-drive"
---


## haystack_integrations.components.fetchers.google_drive.fetcher

### GoogleDriveFetcher

Fetches the full content of Google Drive files via the Drive API v3.

The fetcher complements `GoogleDriveRetriever`, which returns only metadata (and optionally exported text).
Wire the retriever's `documents` (or a list of file ids / Drive URLs) into this fetcher to download the full
content. It dispatches on each file's mime type and always returns `ByteStream`s, ready for a downstream
converter (for example a `FileTypeRouter` in front of `PyPDFToDocument`, `DOCXToDocument`, `XLSXToDocument`,
or `PPTXToDocument`):

- **Binary files** (PDF, DOCX, images, ...) are downloaded as-is via `files.get?alt=media`.
- **Native Google Docs/Sheets/Slides** are exported with `files.export`, by default to the Office formats
  (DOCX/XLSX/PPTX), configurable via `export_mime_types`.
- **Folders** and other non-downloadable Google types (Forms, Sites, ...) are skipped.

Each `ByteStream`'s `meta` carries `file_id`, `web_url`, `file_name`, and `content_type`.

The fetcher takes a per-user `access_token` as a run input, typically wired from an upstream `OAuthTokenResolver`.
The token must carry a delegated Google OAuth scope that allows reading file content (for example
`https://www.googleapis.com/auth/drive.readonly`).

### Usage example

```python
from haystack_integrations.components.fetchers.google_drive import GoogleDriveFetcher

fetcher = GoogleDriveFetcher()

# `access_token` is a per-user delegated Google OAuth bearer token.
result = fetcher.run(
    access_token="my-delegated-google-token",
    targets=["https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view"],
)
streams = result["streams"]
```

In a pipeline, connect `GoogleDriveRetriever.documents` to the fetcher's `targets` input and an upstream
component that emits a per-user `access_token` to the fetcher's `access_token` input.

#### __init__

```python
__init__(
    *,
    api_base_url: str = DEFAULT_API_BASE_URL,
    timeout: float = 30.0,
    max_retries: int = 3,
    max_concurrent_requests: int = 5,
    raise_on_failure: bool = True,
    export_mime_types: dict[str, str] | None = None
) -> None
```

Initialize the fetcher.

**Parameters:**

- **api_base_url** (<code>str</code>) – The Drive API base URL. Defaults to `https://www.googleapis.com/drive/v3`.
- **timeout** (<code>float</code>) – The HTTP timeout in seconds for each request to the Drive API.
- **max_retries** (<code>int</code>) – The maximum number of retries for throttled (HTTP 429) or transient server errors.
- **max_concurrent_requests** (<code>int</code>) – The maximum number of files fetched concurrently by `run_async`. Bounds
  the in-flight requests to Drive to avoid tripping its rate limits. Has no effect on the synchronous
  `run`, which fetches files one at a time.
- **raise_on_failure** (<code>bool</code>) – If `True`, a fetch failure raises an exception. If `False`, the failure is
  logged and the file is skipped, so the other files are still returned.
- **export_mime_types** (<code>dict\[str, str\] | None</code>) – Optional mapping of native Google mime type (for example
  `application/vnd.google-apps.document`) to the mime type to export it as. Replaces the default mapping
  (Docs/Sheets/Slides to DOCX/XLSX/PPTX). Drive caps a single export at 10 MB.

**Raises:**

- <code>GoogleDriveConfigError</code> – If `max_retries` is negative or `max_concurrent_requests` is not positive.

#### run

```python
run(
    access_token: str | Secret, targets: list[Document | str]
) -> dict[str, list[ByteStream]]
```

Fetch the content of Google Drive files and return them as `ByteStream`s.

**Parameters:**

- **access_token** (<code>str | Secret</code>) – A delegated Google OAuth bearer token for the user whose files are fetched, typically
  wired from an upstream `OAuthTokenResolver` (which emits a plain `str`). A `Secret` is also accepted and
  resolved internally.
- **targets** (<code>list\[Document | str\]</code>) – The files to fetch, as either `Document`s emitted by `GoogleDriveRetriever` or raw Google
  Drive file ids / URLs (the two may also be mixed in one list). For a `Document`, the `file_id` in its
  meta is fetched and `mime_type`, `file_name`, and `web_url` are reused when present. For a raw string,
  the file id is parsed from a Drive URL (or used as-is) and the file's mime type is looked up. Folders
  and non-downloadable Google types are skipped.

**Returns:**

- <code>dict\[str, list\[ByteStream\]\]</code> – A dictionary with a `streams` key holding the fetched content as `ByteStream` objects. Each
  stream's `meta` carries `file_id`, `web_url`, `file_name`, and `content_type`.

**Raises:**

- <code>GoogleDriveConfigError</code> – If an item is neither a `Document` nor a `str`, or if `access_token` is a
  `Secret` that does not resolve to a string.
- <code>GoogleDriveRequestError</code> – If a fetch fails and `raise_on_failure` is `True`.

#### run_async

```python
run_async(
    access_token: str | Secret, targets: list[Document | str]
) -> dict[str, list[ByteStream]]
```

Asynchronously fetch the content of Google Drive files and return them as `ByteStream`s.

**Parameters:**

- **access_token** (<code>str | Secret</code>) – A delegated Google OAuth bearer token for the user whose files are fetched, typically
  wired from an upstream `OAuthTokenResolver` (which emits a plain `str`). A `Secret` is also accepted and
  resolved internally.
- **targets** (<code>list\[Document | str\]</code>) – The files to fetch, as either `Document`s emitted by `GoogleDriveRetriever` or raw Google
  Drive file ids / URLs (the two may also be mixed in one list). For a `Document`, the `file_id` in its
  meta is fetched and `mime_type`, `file_name`, and `web_url` are reused when present. For a raw string,
  the file id is parsed from a Drive URL (or used as-is) and the file's mime type is looked up. Folders
  and non-downloadable Google types are skipped.

**Returns:**

- <code>dict\[str, list\[ByteStream\]\]</code> – A dictionary with a `streams` key holding the fetched content as `ByteStream` objects. Each
  stream's `meta` carries `file_id`, `web_url`, `file_name`, and `content_type`.

**Raises:**

- <code>GoogleDriveConfigError</code> – If an item is neither a `Document` nor a `str`, or if `access_token` is a
  `Secret` that does not resolve to a string.
- <code>GoogleDriveRequestError</code> – If a fetch fails and `raise_on_failure` is `True`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> GoogleDriveFetcher
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary representation of this component.

**Returns:**

- <code>GoogleDriveFetcher</code> – The deserialized component instance.

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
    api_base_url: str = DEFAULT_API_BASE_URL,
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
