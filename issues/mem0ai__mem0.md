# Issue 草稿：mem0ai/mem0
> 状态：✅ 已发出 → https://github.com/mem0ai/mem0/issues/7207 （2026-09-02）
> 查重：2026-09-02 GitHub 站内搜索 `repo:mem0ai/mem0 httpx2` = 0 条（主控用 gh 复核）✅

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

Meanwhile, the ecosystem has already begun moving: **Starlette**, the **Anthropic Python SDK (v1.0)**, and the **MCP Python SDK (2.0)** have migrated to `httpx2`; the **OpenAI Python SDK** supports it as an optional extra.

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
- **Dependency conflicts**: packages that pin `httpx<1.0` already conflict with migrated peers; the longer the wait, the worse the resolver pain for downstream users.
- **Compounding migration cost**: the gap between 0.28.x and whatever 1.0 becomes keeps growing; migrating to httpx2 now is a small, well-documented step (official migration guide: https://pydantic.dev/docs/httpx2/get-started/migration/).

## What migration could look like here

`httpx` is a direct, declared runtime dependency of the published `mem0ai` package — `"httpx>=0.28.0"` in the root `pyproject.toml` (currently v2.0.19), notably **with no upper bound**. Direct usage in the library is small and well-contained — two files:

- **`mem0/client/main.py`** — the hosted-platform API client (`from mem0 import MemoryClient`). `MemoryClient` builds an `httpx.Client(...)` and `AsyncMemoryClient` builds an `httpx.AsyncClient(...)`, and `httpx.HTTPStatusError` is caught during API-key validation. Importantly, **httpx types are part of mem0's public API surface**: both constructors accept user-supplied clients — `client: Optional[httpx.Client]` and `client: Optional[httpx.AsyncClient]` — so downstream users can inject their own configured httpx client.
- **`mem0/proxy/main.py`** — the LiteLLM-facing completion wrapper exposes `timeout: Optional[Union[float, str, httpx.Timeout]]` in its public signature.

The rest of the package talks to provider SDKs rather than httpx directly (checked: `mem0/memory/main.py`, `mem0/memory/telemetry.py`, `mem0/llms/openai.py`, `mem0/llms/azure_openai.py`, `mem0/embeddings/openai.py`, `mem0/embeddings/azure_openai.py`, `mem0/vector_stores/azure_ai_search.py`).

Since `mem0ai` is a **published library**, we'd suggest Option A:

- **Option A — dual support (recommended for libraries)**: prefer httpx2 when available, fall back to httpx:

  ```python
  try:
      import httpx2 as httpx
  except ModuleNotFoundError:
      import httpx
  ```

  with `httpx2; python_version >= "3.10"` added as an extra/optional dependency. The API surface you use (`Client`, `AsyncClient`, `HTTPStatusError`, `Timeout`) is compatible.

- **Option B — hard switch**: replace the dependency outright. mem0 already requires **Python >=3.10,<4.0**, so httpx2's Python ≥ 3.10 floor is a non-issue — but as a library, breaking users who inject their own `httpx.Client` into `MemoryClient`/`AsyncMemoryClient` would argue for Option A or a major-version bump.

⚠️ **One caveat worth calling out**: httpx2 verifies TLS against the **OS trust store** instead of the bundled `certifi`. mem0 is frequently self-hosted in containers (your own `server/` Dockerfile flow) and behind corporate proxies — environments that sometimes rely on certifi's bundle or `SSL_CERT_FILE`. Worth a line in the changelog/docs either way.

## Notes

- No existing httpx2-related issue or PR in this repo as of 2026-09-02 (checked via web search; a quick maintainer-side GitHub search to confirm is welcome).
- **The loose `httpx>=0.28.0` spec cuts both ways**: with no upper bound, pip will happily resolve a future breaking httpx 1.0 stable into mem0 installs the day it ships — capping the range or migrating both address that.
- **One subtlety with Option A here**: because `httpx.Client` / `httpx.AsyncClient` / `httpx.Timeout` appear in *public type annotations*, the dual import changes what those annotations resolve to when httpx2 is installed. Runtime duck-typing means a user-passed `httpx.Client` still works fine, but type-checking downstream users may see an annotation mismatch — worth a sentence in the migration notes.
- Scope: this covers the published `mem0ai` package (root `pyproject.toml`). The in-repo `server/` FastAPI app has no direct httpx dependency (`server/requirements.txt` — it inherits httpx via `mem0ai`); the `mem0-ts/` tree is TypeScript and out of scope.

Happy to open a PR for either option if the maintainers are interested — and equally happy to close this if you'd rather wait for httpx 1.0 stable. No pressure; the goal is just to make sure the decision is an informed one. 🙏
