# Issue 草稿：lm-sys/FastChat
> 状态：✅ 已发出 → https://github.com/lm-sys/FastChat/issues/3929 （2026-09-02）
> 查重：2026-09-02 无任何 httpx2 相关 issue/PR ✅（GitHub issues 搜索 `httpx2` 返回 "No results"）

---

**Title:** [HTTPXodus] Consider migrating from `httpx` to `httpx2` (the actively maintained fork)

**Body:**

> 🏷️ *This issue is part of **HTTPXodus**, a community effort to help major projects plan their path off the stalled `httpx` stable line onto [`httpx2`](https://github.com/pydantic/httpx2), the actively maintained fork by Pydantic Services (with original httpx author Tom Christie involved). One coordinated issue per project — no spam, no drive-by PRs.*

## Background — what happened to httpx

`httpx` is one of the most important HTTP clients in the Python ecosystem (370k+ dependent repositories). However:

- The last **stable** release is **0.28.1 (December 2024)** — over 20 months ago.
- For most of 2025 the repository was dormant; maintenance trackers (e.g. Snyk) now flag it as **inactive**.
- Encouragingly, the project has recently resumed activity: `1.0.dev4–dev6` shipped between **2026-08-19 and 2026-08-31**. A 1.0 is clearly in the works, but there is **no committed date for a stable 1.0**, and the dev line carries breaking changes.

Meanwhile, the ecosystem has already begun moving: **Starlette**, the **Anthropic Python SDK (v1.0)**, and the **MCP Python SDK (2.0)** have migrated to `httpx2`; **FastAPI** supports it as a test dependency; the **OpenAI Python SDK** supports it as an optional extra.

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

FastChat's `httpx` usage is unusually narrow — this is a genuinely small migration:

- **A single call site.** Reviewing the serving stack, `httpx` is imported in exactly **one** place: `fastchat/serve/openai_api_server.py`. Everywhere else the project makes HTTP calls (the controller, the non-streaming completion path, the Gradio servers, `api_provider.py`, the monitor/judge scripts) it uses `aiohttp` or `requests` instead.
- **It sits on the core streaming path.** In `generate_completion_stream(payload, worker_addr)` the OpenAI-compatible API server opens an `httpx.AsyncClient`, then `client.stream("POST", worker_addr + "/worker_generate_stream", ...)` and iterates the body with `async for raw_chunk in response.aiter_raw():`, splitting on a `b"\0"` delimiter. So this is the request path that streams completions from a model worker back to the API client — not an edge tool.
- **Async only, internal only.** No `httpx` types leak into `fschat`'s public Python API; it's purely internal server→worker communication. (Note the non-streaming sibling path in the same file uses `fetch_remote(...)` over `aiohttp`, so the codebase already mixes the two clients.)
- **Unpinned dependency.** `pyproject.toml` declares plain `"httpx"` (no version bound) and `requires-python = ">=3.8"`. The lack of an `<1.0` pin means no resolver conflict today, but it also means FastChat will silently pick up httpx **1.0's breaking changes** the day it ships — an argument for choosing the HTTP layer deliberately rather than by accident.

Because `httpx` is confined to one internal function, either option is cheap; the deciding factor is your Python floor:

- **Option A — dual support (recommended)**: `fschat` is a published library and currently supports Python **≥3.8**, while httpx2 requires **≥3.10**. To avoid dropping 3.8/3.9 users, prefer httpx2 when present and fall back to httpx:

  ```python
  try:
      import httpx2 as httpx
  except ModuleNotFoundError:
      import httpx
  ```

  with `httpx2; python_version >= "3.10"` added as an optional/extra dependency. The public API used here (`AsyncClient`, `client.stream`, `aiter_raw`, `timeout`) is unchanged in httpx2, so the streaming path works identically under either import.

- **Option B — hard switch**: if you're willing to bump `requires-python` to `>=3.10` (reasonable in 2026 — 3.8 and 3.9 are both EOL), the switch is trivial: replace `"httpx"` with `"httpx2"` in `pyproject.toml` and change the single import in `openai_api_server.py`. Given there's only one call site, this is close to a one-line change.

⚠️ **One caveat worth calling out**: httpx2 verifies TLS against the **OS trust store** instead of the bundled `certifi`. FastChat deployments that front worker endpoints with TLS behind corporate proxies, or run in minimal containers relying on certifi's CA bundle, may need `SSL_CERT_FILE` or system CA configuration. Worth a line in the deployment docs either way.

## Notes

- **No existing httpx2 issue/PR** in this repo as of 2026-09-02 (checked via GitHub issue search — zero competition).
- **Version constraint**: `httpx` is declared **unpinned**; `requires-python = ">=3.8"`. Any migration must either keep 3.8/3.9 working (Option A) or explicitly drop them (Option B).
- **Project activity — honest assessment.** FastChat's own development has slowed considerably: the latest release is **v0.2.36 (2024-02-11)**, the most recent commit on `main` is a minor `constants.py` update dated **2025-06-02**, and the repo was last pushed **2026-05-01** (a constants.py bump, #3733). The hosted Chatbot Arena also moved to a separate site (LMArena) in September 2024, leaving this repo as primarily the legacy open-source serving/eval stack. So while the `httpx` change itself is tiny and safe, the realistic value of migrating is modest and depends on whether the maintainers intend to keep the serving stack current. We're flagging it for completeness and because the unpinned `httpx` leaves the project exposed to httpx 1.0's breaking changes — but we fully understand if the answer is "we'll wait for httpx 1.0 stable."

Happy to open a PR for either option if the maintainers are interested — and equally happy to close this if you'd rather wait for httpx 1.0 stable. No pressure; the goal is just to make sure the decision is an informed one. 🙏
