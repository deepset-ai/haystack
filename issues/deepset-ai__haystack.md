---
repo: deepset-ai/haystack
star_count: 26400
recon_date: 2026-09-04
status: draft-ready
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
| [OpenAI Python SDK](https://github.com/openai/openai-python) | 31.5k+ | ✅ Migrated |
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
- **Dependency conflicts**: packages that pin `httpx<1.0` already conflict with migrated peers; the longer the wait, the worse the resolver pain for downstream users.
- **Compounding migration cost**: the gap between 0.28.x and whatever 1.0 becomes keeps growing; migrating to httpx2 now is a small, well-documented step (official migration guide: https://pydantic.dev/docs/httpx2/get-started/migration/).

## What migration could look like here

**Usage** (from `recon.py` / source inspection):

- `haystack/utils/http_client.py`: `import httpx`; defines `init_http_client()` returning `httpx.Client | httpx.AsyncClient | None`; also references `httpx.Limits`; exposes `Client` / `AsyncClient` types in overload signatures.
- `pyproject.toml`: `"httpx"` in core dependencies; `"httpx[http2]"` in test env (needed by link-content-fetcher tests).
- Only one source file directly imports `httpx`; downstream consumers mostly receive clients via `init_http_client`.

Two viable paths, depending on your compatibility goals:

- **Option A — dual support (recommended for libraries)**: prefer httpx2 when available, fall back to httpx:

  ```python
  try:
      import httpx2 as httpx
  except ModuleNotFoundError:
      import httpx
  ```

  with `httpx2; python_version >= "3.10"` added as an optional/extra dependency (or alongside `httpx` in core deps for dual-run).

- **Option B — hard switch (fine for applications / next major)**: replace the dependency outright. Requires Python ≥ 3.10 (already `requires-python >= 3.10`).

⚠️ **One caveat worth calling out**: httpx2's move to the OS trust store can change behavior in containers and corporate-proxy environments that relied on `certifi`. Worth a note in your changelog either way.

**Notes:**
- Previous related PRs `#12336`, `#12348`, `#12332` (closed) deal with `openai` / `numpy` pinning — not an `httpx` migration; no duplicate `httpx2` migration issue exists for this repo.
- `pyproject.toml` pins no upper bound on `httpx`; with `httpx` 1.0 dev active, tightening or dual-importing now avoids future resolver conflicts.
- Migration should update `pyproject.toml` (core `httpx2` optional + keep `httpx`) and `haystack/utils/http_client.py` (dual import), and mirror any test refs that assert on `httpx.Client` instances.
- No PR will be opened by this campaign until maintainers confirm interest; happy to close if you'd rather wait for `httpx` 1.0 stable.

Happy to open a PR for either option if the maintainers are interested — and equally happy to close this if you'd rather wait for httpx 1.0 stable. No pressure; the goal is just to make sure the decision is an informed one. 🙏

---
*Draft prepared by HTTPXodus campaign (author: xic; no AI co-signature). Issue not yet created on GitHub — awaiting human review per HTTPXodus charter (no drive-by PRs, issue first).*