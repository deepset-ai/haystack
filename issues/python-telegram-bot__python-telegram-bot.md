# Issue 草稿：python-telegram-bot/python-telegram-bot
> 状态：✅ 接管 issue #5258 完成（comment 5523003311，2026-09-03）
> 查重：2026-09-03 GitHub 搜索 `repo:python-telegram-bot/python-telegram-bot httpx2` = 1 result (issue #5258 itself), `total_count=1`，无 PR
> 现有 issue #5258 由维护者 BjoernPetersen 自己开启（2026-06-05），标签 `🛠 breaking / 🛠 refactor`；harshil21 在 #5258 评论提及"正在考虑不同的网络后端"；Kludex 提供了 https://httpx2.pydantic.dev/migration/ 链接

---

**Title:** [HTTPXodus] Consider migrating from `httpx` to `httpx2` (the actively maintained fork)

**Body:**

![HTTPXodus — The Great HTTP Ecosystem Migration](https://raw.githubusercontent.com/ProgrammerPlus1998/httpxodus/main/assets/banner.png)

> 🏷️ *This issue is part of **HTTPXodus**, a community effort to help major projects plan their path off the stalled `httpx` stable line onto [`httpx2`](https://github.com/pydantic/httpx2), the actively maintained fork by Pydantic Services (with original httpx author Tom Christie involved). One coordinated issue per project — no spam, no drive-by PRs.*

> 🔁 *This HTTPXodus thread is intentionally being posted on the existing [issue #5258](https://github.com/python-telegram-bot/python-telegram-bot/issues/5258) opened by @BjoernPetersen, rather than as a new issue — per the campaign's "no duplicates" rule. The original issue body and Kludex's migration-guide link are the right starting point; the data below is meant to accelerate the next step.*

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
- **Dependency conflicts**: packages that pin `httpx<1.0` already conflict with migrated peers; the longer the wait, the worse the resolver pain for downstream users.
- **Compounding migration cost**: the gap between 0.28.x and whatever 1.0 becomes keeps growing; migrating to httpx2 now is a small, well-documented step (official migration guide: https://httpx2.pydantic.dev/migration/, also linked by @Kludex in this thread).

## What migration could look like here

`httpx` is a **direct runtime dependency** of `python-telegram-bot` — `"httpx >=0.27,<0.29"` in `pyproject.toml`. The usage is small and well-contained: with the recent `src/` refactor, only **two** production files import `httpx`:

- **`src/telegram/request/_httpxrequest.py`** — the entire networking layer (`class HTTPXRequest`). Builds `httpx.AsyncClient` / `httpx.Timeout` / `httpx.Limits` / `httpx.AsyncHTTPTransport` and catches `httpx.TimeoutException` (with an `isinstance(err, httpx.PoolTimeout)` narrowing) plus `httpx.HTTPError` as the base class.
- **`src/telegram/ext/_applicationbuilder.py`** — only `httpx.Proxy` / `httpx.URL` as **public type annotations** on `Builder.proxy()` and `Builder.get_updates_proxy()`.

That said, the public API does expose some httpx types:

- `HTTPXRequest`'s constructor signature uses `proxy: str | httpx.Proxy | httpx.URL | None` and `httpx_kwargs: dict[str, Any] | None` (which downstream users typically pass `httpx.Timeout` / `httpx.Limits` through).
- The `Builder.proxy()` and `Builder.get_updates_proxy()` setters are typed as `str | httpx.Proxy | httpx.URL`.
- `get_updates_client` (returned by `ApplicationBuilder.get_updates_client()`) is documented as returning an `httpx.AsyncClient` — so `httpx.AsyncClient` is a documented return type for downstream code.

Since `python-telegram-bot` is a **published library** with a very large user surface (PTB is one of the most-installed Telegram bot frameworks), the recommended shape is Option A:

- **Option A — dual support (recommended for libraries)**: prefer httpx2 when available, fall back to httpx:

  ```python
  try:
      import httpx2 as httpx
  except ModuleNotFoundError:
      import httpx
  ```

  with `httpx2>=2.12.0; python_version >= "3.10"` added alongside the existing `httpx >=0.27,<0.29`. The marker is a no-op under your current `requires-python = ">=3.10"` but documents the floor and keeps pip's resolver clean.

- **Option B — hard switch (fine for the next major)**: replace the dependency outright, change the `Builder.proxy()` annotation to `httpx2.Proxy | httpx2.URL`, etc. Less surface-area leakage, but a clearly breaking change for users who pass `httpx.Proxy`/`httpx.URL` into `Builder.proxy()`.

⚠️ **One caveat worth calling out**: httpx2's move to the OS trust store can change TLS behavior in containers and corporate-proxy environments that relied on certifi. Worth a line in the changelog either way.

## Notes

- **The right place for this conversation is the existing #5258** (this draft is local-only — not being posted as a new issue). The two existing comments from @harshil21 ("we are considering different networking backends") and @Kludex (migration guide link) are the right context to build on.
- **No existing PR** as of 2026-09-03. Branch and PR are easy to spin up the moment maintainers signal which option they'd prefer.
- **`httpx2` already maintains the exception hierarchy** that `_httpxrequest.py` relies on: `httpx2.TimeoutException`, `httpx2.PoolTimeout`, `httpx2.HTTPError`, `httpx2.Proxy`, `httpx2.URL`, `httpx2.Timeout`, `httpx2.Limits`, `httpx2.AsyncClient`, `httpx2.AsyncHTTPTransport` are all present in the 0.28.1 lineage. No `httpx.HTTPStatusError` is caught in PTB's networking layer, so the dual-import class-identity concerns that bit AutoGPT (their `service.py` needed `_real_httpx` to catch starlette's real httpx exceptions) don't apply here.
- **`python-telegram-bot[http2]` and `[socks]` extras** stay valid with Option A — `httpx2` ships the same `httpx2[http2]` / `httpx2[socks]` extras under the same module path semantics. The extras block in `pyproject.toml` would just need a parallel entry.
- **Test surface**: 6 test files reference `httpx` (conftest, test_bot, test_official/scraper, auxil/networking, ext/test_applicationbuilder, request/test_request, ext/test_updater). With Option A these would each need a paired `import httpx2 as httpx` so the assertion / monkeypatch targets match the module the SUT now imports.

Happy to open a PR for Option A (the recommended path) once maintainers signal which direction the "different networking backends" consideration is going — and equally happy to leave this dormant if the maintainer team prefers to wait for httpx 1.0 stable or pursue a non-httpx backend. No pressure; the goal is just to make sure the option is visible alongside the others. 🙏

---

## HTTPXodus metadata
- This draft was **not** posted as a new issue. The HTTPXodus takeover comment is on #5258 at https://github.com/python-telegram-bot/python-telegram-bot/issues/5258#issuecomment-5523003311
- Posted by: `ProgrammerPlus1998` (HTTPXodus coordinator)
- Date: 2026-09-03
- Reason for takeover mode: maintainer-owned feature issue with zero replies from maintainers beyond "we're considering different backends" — adding data, not duplicating the conversation.
