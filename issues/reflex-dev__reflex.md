# Issue 草稿：reflex-dev/reflex
> 状态：✅ 已发出 → https://github.com/reflex-dev/reflex/issues/7034 （2026-09-02）
> 查重：2026-09-02 未发现 httpx2 相关 issue/PR ✅（方法见下方"查重说明"）

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

The good news first: in Reflex, **httpx is used only by internal framework / CLI tooling — never by end-user app code** (user apps are served over ASGI via Starlette/Granian; httpx never enters the request path). All call sites are **synchronous** — there is no `AsyncClient` anywhere in the package — and most are lazily imported inside functions. Verified against the 0.9.10 sdist:

- **`reflex/utils/net.py`** — the central helper. `_httpx_client()` builds one shared `httpx.Client` with a custom `httpx.HTTPTransport(local_address=..., verify=...)` and per-scheme `mounts` that set `httpx.HTTPTransport(proxy=httpx.Proxy(url=url), verify=...)`. It also powers plain `httpx.head()` connectivity probes (e.g. `http://1.1.1.1`, IPv6) and wraps `get`/`post`/`head` for the rest of the CLI.
- **`reflex/utils/telemetry.py`** — `_send_event()` fires anonymous usage telemetry to PostHog via `httpx.post(POSTHOG_API_URL, ...)` (all exceptions swallowed).
- **`reflex/utils/js_runtimes.py`**, **`reflex/utils/templates.py`**, **`reflex/utils/frontend_skeleton.py`** — download Node/Bun runtimes, app templates, and the frontend skeleton; each catches `httpx.HTTPError`.
- **`reflex/utils/registry.py`** and **`reflex/custom_components/custom_components.py`** — query / publish to the custom-component registry via `httpx.post`, checking `response.status_code == httpx.codes.FORBIDDEN`.

Because the blast radius is a handful of internal, sync-only call sites, this is one of the lower-risk migrations in the ecosystem. One concrete thing to verify up front (it matters for **either** option below):

> ⚠️ **`net.py` reaches into a private module**: `from httpx._utils import get_environment_proxies`. That underscore module is not part of httpx's public API, so before switching you'll want to confirm `httpx2` still ships an equivalent helper (and that the `mounts=` / `httpx.Proxy(...)` construction behaves the same). This is the single non-mechanical change in an otherwise drop-in migration.

Two viable paths, depending on your compatibility goals:

- **Option A — dual support (recommended for libraries)**: prefer httpx2 when available, fall back to httpx:

  ```python
  try:
      import httpx2 as httpx
  except ModuleNotFoundError:
      import httpx
  ```

  with `httpx2; python_version >= "3.10"` added as an extra/optional dependency. This keeps Reflex installable alongside apps that still pin `httpx`.

- **Option B — hard switch (fine here too)**: replace `httpx >=0.26,<1.0` with `httpx2` and switch imports to `import httpx2 as httpx`. Because every call site is internal and synchronous, this is genuinely low-risk for Reflex — arguably cleaner than dual support, since nothing in the public API re-exports httpx types. Requires Python ≥ 3.10, which Reflex already mandates (`>=3.10,<4.0`).

⚠️ **One caveat worth calling out**: httpx2 verifies TLS against the **OS trust store** instead of the bundled `certifi`. This is especially relevant to Reflex because `net.py` already has first-class proxy / custom-`verify` handling for exactly the corporate-proxy and locked-down-container users this change affects — those environments may need `SSL_CERT_FILE` or system CA configuration after the switch. Worth a line in the changelog either way.

## Notes

- **Current constraint**: `httpx >=0.26,<1.0` — a **runtime** dependency in `[project.dependencies]` (`pyproject.toml`, line 25), not a dev/optional extra. **Python requirement**: `>=3.10,<4.0`, so httpx2's Python ≥ 3.10 floor is already satisfied. (Verified against the published 0.9.10.post1 sdist.)
- **Ecosystem alignment, close to home**: Reflex already depends on `starlette >=1.3.1`, and Starlette has adopted httpx2 (in its TestClient). Migrating keeps Reflex on the same HTTP line as one of its own core dependencies.
- **Sibling distribution**: `reflex-hosting-cli` (a separate package Reflex depends on) independently pins `httpx >=0.25.1,<1.0`. A complete migration story would want to cover it too; `reflex-base` has **no** httpx dependency.
- **No duplication**: as of 2026-09-02 there is no existing httpx2-related issue or PR in this repository that we could find.

Happy to open a PR for either option if the maintainers are interested — and equally happy to close this if you'd rather wait for httpx 1.0 stable. No pressure; the goal is just to make sure the decision is an informed one. 🙏
