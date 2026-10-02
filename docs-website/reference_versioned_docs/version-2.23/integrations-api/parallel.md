---
title: "Parallel"
id: integrations-parallel
description: "Parallel integration for Haystack"
slug: "/integrations-parallel"
---


## haystack_integrations.components.generators.parallel.chat.chat_generator

### ParallelChatGenerator

Bases: <code>OpenAIResponsesChatGenerator</code>

Completes chats using Parallel's web-research model.

Powered by the Parallel Responses API (`POST /v1/responses`, OpenAI Responses-compatible).
Every answer is grounded in live web research with citations; the `reasoning.effort`
parameter selects the research tier: `low` (~5-10s), `medium` (~15-20s, default), or
`high` (~30-60s).
See the [Parallel Responses API quickstart](https://docs.parallel.ai/responses-api/responses-quickstart)
for details.

It uses the [ChatMessage](https://docs.haystack.deepset.ai/docs/chatmessage) format in input and output.
Web grounding is built in, so tool calling and sampling parameters (`tools`, `temperature`,
`top_p`, ...) are accepted for SDK compatibility but silently ignored by the API; this component
warns when it sees them.

Because a single call runs live research, `timeout` defaults to 120 seconds rather than the
30 seconds inherited from the OpenAI client, so that the `high` tier fits comfortably.

### Usage example

```python
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.parallel import ParallelChatGenerator

messages = [ChatMessage.from_user("What did Parallel Web Systems announce this year?")]

client = ParallelChatGenerator(generation_kwargs={"reasoning": {"effort": "low"}})
response = client.run(messages)
print(response)
```

#### SUPPORTED_MODELS

```python
SUPPORTED_MODELS: list[str] = ['parallel']
```

The Parallel Responses API models supported by this component.
See https://docs.parallel.ai/responses-api/responses-quickstart for details.

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("PARALLEL_API_KEY"),
    model: str = "parallel",
    api_base_url: str | None = "https://api.parallel.ai/v1",
    streaming_callback: StreamingCallbackT | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    timeout: float | None = 120.0,
    extra_headers: dict[str, Any] | None = None,
    max_retries: int | None = 3,
    http_client_kwargs: dict[str, Any] | None = None
) -> None
```

Initialize the ParallelChatGenerator component.

**Parameters:**

- **api_key** (<code>Secret</code>) – The Parallel API key.
- **model** (<code>str</code>) – The Parallel Responses API model to use.
- **api_base_url** (<code>str | None</code>) – The Parallel API base URL.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback function called when a new token is received from the stream.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional parameters sent directly to the Parallel Responses API, such as
  `reasoning` (e.g. `{"effort": "low"}`) to select the research tier or
  `text` for structured output.
- **timeout** (<code>float | None</code>) – Timeout in seconds for Parallel API calls. Defaults to 120 seconds, which leaves room for
  the `high` research tier (~30-60s). Pass `None` to fall back to the OpenAI client default
  (the `OPENAI_TIMEOUT` environment variable, or 30 seconds), which is too short for most
  research calls.
- **extra_headers** (<code>dict\[str, Any\] | None</code>) – Additional HTTP headers to include in requests to the Parallel API.
- **max_retries** (<code>int | None</code>) – Maximum number of retries to contact Parallel after an internal error. Kept low because
  every retry runs a full research call. Pass `None` to fall back to the OpenAI client
  default (the `OPENAI_MAX_RETRIES` environment variable, or 5).
- **http_client_kwargs** (<code>dict\[str, Any\] | None</code>) – A dictionary of keyword arguments to configure a custom `httpx.Client` or `httpx.AsyncClient`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

## haystack_integrations.components.websearch.parallel.parallel_websearch

### ParallelWebSearch

A component that uses Parallel to search the web and return results as Haystack Documents.

This component wraps the Parallel Search API, enabling web search queries that return
LLM-optimized excerpts as structured documents with content and links, plus the
session identifier that ties related searches together.

You need a Parallel API key from [parallel.ai](https://parallel.ai).

### Usage example

```python
from haystack_integrations.components.websearch.parallel import ParallelWebSearch
from haystack.utils import Secret

websearch = ParallelWebSearch(
    api_key=Secret.from_env_var("PARALLEL_API_KEY"),
    top_k=5,
)
result = websearch.run(query="What is Haystack by deepset?")
documents = result["documents"]
links = result["links"]

# Pass the session back on follow-up searches that are part of the same task
# to get better contextual results.
follow_up = websearch.run(
    query="Who maintains Haystack?",
    search_params={"session_id": result["session_id"]},
)
```

#### __init__

```python
__init__(
    *,
    api_key: Secret = Secret.from_env_var("PARALLEL_API_KEY"),
    top_k: int | None = 10,
    search_params: dict[str, Any] | None = None,
    timeout: float = 30.0
) -> None
```

Initialize the ParallelWebSearch component.

**Parameters:**

- **api_key** (<code>Secret</code>) – API key for Parallel. Defaults to the `PARALLEL_API_KEY` environment variable.
- **top_k** (<code>int | None</code>) – Maximum number of results to return. Maps to the `advanced_settings.max_results` API parameter.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Additional parameters passed to the Parallel Search API.
  See the [Parallel Search API reference](https://docs.parallel.ai/api-reference/search/search)
  for available options. Supported keys include: `objective` (natural-language search goal,
  defaults to the query), `mode` (`turbo`, `fast`, `basic`, or `advanced`, in increasing
  order of latency and quality; the API defaults to `advanced`), `max_chars_total`,
  `session_id`, `client_model`, and `advanced_settings` (nested `source_policy` domain and
  date filters, `fetch_policy`, `excerpt_settings`, `location`, `max_results`).
  Pass `session_id` to link several searches into one task; the identifier the API used
  is always returned in the `session_id` output, whether it was sent or server-generated.
- **timeout** (<code>float</code>) – Request timeout in seconds.

#### warm_up

```python
warm_up() -> None
```

Initialize the sync HTTP client.

Called automatically on first use. Can be called explicitly to avoid cold-start latency.

#### warm_up_async

```python
warm_up_async() -> None
```

Initialize the async HTTP client on the serving event loop.

Called automatically on first use. Can be called explicitly to avoid cold-start latency.

#### close

```python
close() -> None
```

Release the sync HTTP client.

#### close_async

```python
close_async() -> None
```

Release the async HTTP client.

#### run

```python
run(query: str, search_params: dict[str, Any] | None = None) -> dict[str, Any]
```

Search the web using Parallel and return results as Documents.

**Parameters:**

- **query** (<code>str</code>) – Search query string.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Optional per-run override of search parameters.
  If provided, fully replaces the init-time `search_params`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with:
- `documents`: List of Documents containing search result excerpts.
- `links`: List of URLs from the search results.
- `session_id`: Session identifier for this search, echoed back from
  `search_params["session_id"]` if it was provided and generated by the API otherwise.
  Pass it to subsequent searches that belong to the same task.

#### run_async

```python
run_async(
    query: str, search_params: dict[str, Any] | None = None
) -> dict[str, Any]
```

Asynchronously search the web using Parallel and return results as Documents.

**Parameters:**

- **query** (<code>str</code>) – Search query string.
- **search_params** (<code>dict\[str, Any\] | None</code>) – Optional per-run override of search parameters.
  If provided, fully replaces the init-time `search_params`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with:
- `documents`: List of Documents containing search result excerpts.
- `links`: List of URLs from the search results.
- `session_id`: Session identifier for this search, echoed back from
  `search_params["session_id"]` if it was provided and generated by the API otherwise.
  Pass it to subsequent searches that belong to the same task.
