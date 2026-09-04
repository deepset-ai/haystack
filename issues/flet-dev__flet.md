# Issue 草稿：flet-dev/flet
> 状态：✅ 已发出 → https://github.com/flet-dev/flet/issues/6809 （2026-09-02）
> 查重：2026-09-02 无任何 httpx2 相关 issue/PR ✅
> 侦察方式备注：本次侦察期间本机 WebFetch 与 api.github.com 均不可达，星数采用战役简报口径（~16.6k）未经 API 复核；pyproject/METADATA 事实来自 PyPI 镜像（Gemfury）与官方 changelog，httpx 运行时用法经 issue #6258 + 包内源码双重确认。建议发出前用 gh 复核星数与文件树。

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

Flet is published to PyPI (`pip install flet`), so it is a **library** — whichever HTTP client it depends on is inherited by every downstream app. That makes the choice worth making deliberately.

httpx is a genuine **runtime** dependency of the core `flet` package, and it is already unpinned at the top:

```
Requires-Dist: httpx>=0.28.1; platform_system != "Emscripten"
```

an open floor (`>=0.28.1`, **no `<1.0` upper bound**), excluded only on the web target (`platform_system != "Emscripten"`, i.e. Pyodide/WebAssembly) where there is no socket support anyway.

Where it is actually used — the **OAuth / authentication** feature:

- `sdk/python/packages/flet/src/flet/auth/authorization_service.py` uses `httpx` (and `oauthlib`) inside its OAuth 2.0 flow — exchanging the authorization code for tokens and fetching the user profile / groups over **async** HTTP (`httpx.AsyncClient`). It backs the `AuthorizationService` / `page.login()` API, alongside `flet/auth/oauth_provider.py`, `oauth_token.py`, and the built-in GitHub / Google / Azure / Auth0 providers. The `httpx` import itself is lazy (inside the methods, not a top-level line), which is why the web build could drop it — and that is exactly what happened: once httpx was excluded on Pyodide, `flet build web` crashed (#6258, now closed). That incident is the concrete proof that httpx is a real runtime dependency, not a dev/optional extra.
- The desktop/mobile client↔server protocol itself runs over WebSockets (via `flet-web` / FastAPI), not httpx — httpx's role in the SDK is specifically these OAuth HTTP calls.

Two viable paths, depending on your compatibility goals:

- **Option A — dual support (recommended for libraries)**: prefer httpx2 when available, fall back to httpx:

  ```python
  try:
      import httpx2 as httpx
  except ModuleNotFoundError:
      import httpx
  ```

  with `httpx2; python_version >= "3.10"` added as an extra/optional dependency. Either way, **keep the `platform_system != "Emscripten"` marker** on whichever package is declared — httpx2, like httpx, cannot run under Pyodide/wasm, so the web exclusion must carry over.

- **Option B — hard switch**: replace the dependency outright and switch the import to `import httpx2 as httpx`. This is unusually clean here: Flet already declares `requires-python >=3.10`, which satisfies httpx2's Python ≥ 3.10 floor, so no supported interpreter is dropped — and with a 1.0 on the way, a small dependency swap fits naturally into the release notes.

⚠️ **One caveat worth calling out**: httpx2's move to the OS trust store can change behavior in containers and corporate-proxy environments that relied on certifi. For Flet this mostly affects self-hosted desktop/server deployments; worth a note in your changelog either way.

## Notes

- No existing httpx2-related issue or PR in this repo as of 2026-09-02 (checked via GitHub search).
- **Dependency constraint**: `httpx>=0.28.1` — an open floor with no `<1.0` pin, so there is no resolver conflict to untangle. The only special handling is the `platform_system != "Emscripten"` marker (web/Pyodide exclusion), which a migration must replicate for httpx2.
- **Python floor**: `requires-python >=3.10` already satisfies httpx2's ≥ 3.10 requirement, so even a hard switch drops no currently-supported Python.
- **Timing**: Flet is on the road to 1.0 (1.0 Beta shipped 2025-12; recent releases ~0.8x), an ideal window if you would rather fold this into a major than carry it as a compat shim.
- The public API surface relevant here (`AsyncClient`, `Response.raise_for_status()`, `HTTPStatusError`) behaves the same in httpx2, so the OAuth code path should need no logic changes beyond the import.

Happy to open a PR for either option if the maintainers are interested — and equally happy to close this if you'd rather wait for httpx 1.0 stable. No pressure; the goal is just to make sure the decision is an informed one. 🙏
