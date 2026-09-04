# Issue 草稿：chroma-core/chroma
> 状态：✅ 已发出 → https://github.com/chroma-core/chroma/issues/7671 （2026-09-02）
> 查重：2026-09-02 无任何 httpx2 相关 issue/PR（GitHub 搜索 `httpx2` → 0 结果；search API `total_count=0`），亦无 httpx→1.0 迁移 issue → 零竞争 ✅

---

**Title:** [HTTPXodus] Consider migrating from `httpx` to `httpx2` (the actively maintained fork)

**Body:**

![HTTPXodus — The Great HTTP Ecosystem Migration](https://raw.githubusercontent.com/ProgrammerPlus1998/httpxodus/main/assets/banner.png)

> 🏷️ *This issue is part of **HTTPXodus**, a community effort to help major projects plan their path off the stalled `httpx` stable line onto [`httpx2`](https://github.com/pydantic/httpx2), the actively maintained fork by Pydantic Services (with original httpx author Tom Christie involved). One coordinated issue per project — no spam, no drive-by PRs.*

## Background — what happened to httpx

`httpx` is one of the most important HTTP clients in the Python ecosystem (370k+ dependent repositories). However:

- The last **stable** release is **0.28.1 (December 2024)** — over 20 months ago.
- For most of 2025 the repository was dormant; maintenance trackers (e.g. Snyk) now flag it as **inactive**.
- Encouragingly, the project has recently resumed activity: `1.0.dev4–dev6` shipped between **2026-08-19 and 2026-08-31**. A 1.0 is clearly in the works, but there is **no committed date for a stable 1.0**, and the dev line carries breaking changes.

Meanwhile, the ecosystem has already begun moving. Projects that have adopted `httpx2`:

| Project | Stars | Status |
|---------|-------|--------|
| [Starlette](https://github.com/encode/starlette) | 12.5k+ | ✅ Migrated |
| [FastAPI](https://github.com/fastapi/fastapi) | 102k+ | ✅ Dual support (httpx2 test dependency) |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | 31.5k+ | ✅ Supports httpx2 as an optional extra |
| [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) | 3.9k+ | ✅ Migrated in v1.0 |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | 24k+ | ✅ Migrated in v2.0 |

## What is httpx2

[`httpx2`](https://github.com/pydantic/httpx2) is a fork of httpx 0.28.1 maintained by **Pydantic Services Inc.**, with Tom Christie involved. It is actively released (v2.0.0 → v2.12.0 since May 2026) and keeps a **compatible public API**:

```python
import httpx2
r = httpx2.get("https://example.org")
```

## Why migrate — the benefits

1. **Active maintenance** — regular releases, reviewed PRs, and a funded maintainer team, versus an uncertain stable-release timeline.
2. **Modern TLS by default** — certificates are verified against the **OS trust store** instead of the bundled `certifi`, removing a common source of stale-CA issues.
3. **New capabilities** — built-in Server-Sent Events (`client.sse()`) and WebSocket support (`httpx2[ws]`), without extra dependencies.
4. **Ecosystem alignment** — as frameworks and SDKs migrate, staying on httpx increasingly means duplicate HTTP stacks in one dependency tree.

## Risks of staying on httpx

- **Security exposure**: no stable-line releases means no stable-line security fixes. If a CVE lands in 0.28.x today, there is no maintained branch to patch it.
- **Dependency conflicts**: packages that pin `httpx<1.0` already conflict with migrated peers (see e.g. the langfuse discussion); the longer the wait, the worse the resolver pain for downstream users.
- **Compounding migration cost**: the gap between 0.28.x and whatever 1.0 becomes keeps growing; migrating to httpx2 now is a small, well-documented step (official migration guide: https://pydantic.dev/docs/httpx2/get-started/migration/).

## What migration could look like here

In Chroma, `httpx` is the transport that connects the **Python client to a Chroma server** (and to Chroma Cloud). It sits behind the public constructors `chromadb.HttpClient()`, `chromadb.AsyncHttpClient()`, and `chromadb.CloudClient()`, which build on three files in `chromadb/api/`:

- **`chromadb/api/fastapi.py`** — `class FastAPI(BaseHTTPClient, ServerAPI)`: the sync client. Constructs an `httpx.Client` and stores it as the private `self._session`.
- **`chromadb/api/async_fastapi.py`** — `class AsyncFastAPI(BaseHTTPClient, AsyncServerAPI)`: the async client. Constructs an `httpx.AsyncClient` per event loop, stored in `_clients: Dict[int, httpx.AsyncClient]`.
- **`chromadb/api/base_http_client.py`** — `class BaseHTTPClient`: shared base that builds the connection `httpx.Limits` and maps error responses.

Two properties of this design make a migration here comparatively low-risk:

- **httpx is essentially an implementation detail.** Callers cannot inject their own httpx client, and every response is parsed to JSON (`orjson.loads(response.text)`) before being returned — so `httpx.Response` never leaks into the results users receive. The only httpx type on the importable surface is `httpx.Limits`, exposed via the public `BaseHTTPClient.http_limits` property.
- **Usage is concentrated.** Beyond the client layer, only a couple of embedding functions touch httpx directly — `chromadb/utils/embedding_functions/jina_embedding_function.py` (`self._session = httpx.Client()`) and `roboflow_embedding_function.py` (a lazy `importlib.import_module("httpx")`). The other remote embedding providers (OpenAI, Mistral, Ollama, Baseten, …) go through their vendor SDKs, so their HTTP layer is those SDKs' concern, not Chroma's.

Two viable paths, depending on your compatibility goals:

- **Option A — dual support (recommended for libraries)**: prefer httpx2 when available, fall back to httpx:

  ```python
  try:
      import httpx2 as httpx
  except ModuleNotFoundError:
      import httpx
  ```

  with `httpx2; python_version >= "3.10"` added as an extra/optional dependency. This is the natural fit here for two reasons: Chroma is a published library (`pip install chromadb`), and Chroma currently supports **Python ≥ 3.9** while httpx2 requires **Python ≥ 3.10** — so the marker lets 3.10+ users get httpx2 while 3.9 keeps working on httpx. Because the public surface barely exposes httpx types, the `import httpx2 as httpx` alias keeps `http_limits` (→ `httpx.Limits`) and the rest of the client working unchanged.

- **Option B — hard switch (fine for applications / next major)**: replace `httpx>=0.27.0` with `httpx2>=2.12` outright and switch imports to `import httpx2 as httpx`. Requires bumping `requires-python` to `>=3.10`, i.e. dropping Python 3.9 support — a bigger commitment, so probably one for a future major release.

⚠️ **One caveat worth calling out**: httpx2's move to the OS trust store can change TLS behavior in containers and corporate-proxy environments that relied on certifi's bundle. Worth a note in your changelog either way.

## Notes

- **No duplication**: as of 2026-09-02 there is no existing httpx2-related issue or PR in this repo (GitHub search `httpx2` → 0 results; search API `total_count = 0`), and no issue proposing a move to httpx 1.0. This issue would not duplicate anything.
- **Dependency**: `httpx>=0.27.0` is a **core runtime dependency** — declared in `pyproject.toml` `[project].dependencies` and in `requirements.txt` — not optional and not dev-only. It backs both the self-hosted-server client path and the `CloudClient` (Chroma Cloud) path.
- **Python floor**: `requires-python = ">=3.9"` in `pyproject.toml`. This is the main reason Option A (dual import with a `python_version >= "3.10"` marker) is the lower-friction path; Option B implies dropping 3.9.
- **Active, related work in-repo**: there's a live cluster of issues/PRs touching this exact transport layer — TLS verification defaults in the async client (#7511, #7517, #7549, #7578), client resource cleanup (#7533), and per-event-loop connection-pool scoping (#6882, #7303). None of them propose replacing httpx, but all of them are edits to the same few files a migration would touch, so it may be a convenient moment to consider the question. (Separately, issue #7097 proposes adding *more* httpx usage via a native `HttpEmbeddingFunction` — worth being aware of in either direction.)

Happy to open a PR for either option if the maintainers are interested — and equally happy to close this if you'd rather wait for httpx 1.0 stable. No pressure; the goal is just to make sure the decision is an informed one. 🙏
