# Issue 草稿：instructor-ai/instructor (实际仓库 567-labs/instructor)
> 状态：🔜 待发出
> 查重：2026-09-03 在 567-labs/instructor 发现
> - issue #2553 [OPEN] "Adding support to openai v3.x.x"（跟进 OpenAI 3 升级）
> - PR #2583 [OPEN] "fix(openai): support SDK 3 HTTPX2 clients"（已通过 OpenAI 3 升级接入 httpx2 传输）
> - 0 个标题含 "httpx2" 的 issue/PR
> 评估：**非完整 duplicate，但目标已部分达成**。HTTPXodus issue 提的"测试代码也走 httpx2/httpx 双导入兼容"是 PR #2583 没碰的剩余工作。

---

**Title:** [HTTPXodus] Support `httpx2` in the OpenAI 3 path (complement to #2583)

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

## Why this matters for instructor specifically

First, full disclosure: **instructor is already on a good path here**. Issue #2553 ("Adding support to openai v3.x.x") and PR #2583 ("fix(openai): support SDK 3 HTTPX2 clients") are landing OpenAI SDK 3 support, which transitively pulls in the httpx2 transport. The legacy `import httpx` and the `httpx.Client` / `httpx.AsyncClient` type annotations in the `openai/` builder are being removed in that PR. This is great — it's how the OpenAI-3 path already gives you httpx2 for free.

So this issue is **not asking you to redo that work**. The point of this HTTPXodus ticket is to flag the **residual places in the codebase that still import or annotate `httpx` directly**, which PR #2583 doesn't cover, and to suggest how to keep them in step.

## Where instructor still references `httpx` directly today

`grep -rn 'import httpx' instructor/ tests/` (against `567-labs/instructor` @ `main`, 2026-09-03):

| File | Lines | Usage |
|------|-------|-------|
| `instructor/v2/auto_client.py` | 186, 192, 234, 247 | Lazy import in `_build_openai`; `cast(Optional[httpx.AsyncClient], ...)` and `cast(Optional[httpx.Client], ...)` for the public `http_client` kwarg. The `httpx` symbol in the `ImportError` branch's `{missing_root not in {"openai", "httpx"}}` set is also httpx-specific. |
| `tests/coverage/test_openai_support_coverage.py` | 9, 54, 58, 60, 88, 92, 94-95, 124, 148, 153, 178, 183 | Uses `httpx.MockTransport` + `httpx.Client(trust_env=False)` / `AsyncClient(trust_env=False)` to mock OpenAI responses. |
| `tests/coverage/test_anthropic_support_coverage.py` | 14, 78, 109, 114 | Same pattern, Anthropic variant. |
| `tests/coverage/test_core_client_coverage.py` | 7, 448-449, 454, 470-471, 476 | `httpx.MockTransport` for retry behaviour. |
| `tests/coverage/test_cohere_coverage.py` | 15, 48-55, 102, 107, 162, 167, 225, 230, 284, 289, 307, 312 | `httpx.Client(trust_env=False)` + `AsyncClient` fixtures. |
| `tests/coverage/test_provider_clients_coverage.py` | 12, 167, 202, 254, 267, 269, 297, 324, 329, 331-332, 368 | Same — multi-provider coverage suite. |
| `tests/v2/test_auto_client_deterministic.py` | 87, 99, 100, 103 | Builds a synthetic `httpx` module via `ModuleType` for monkey-patching — only file the PR #2583 *does* touch. |
| `tests/providers/test_auto_client.py` | 559, 573 | Docstring + assertion text only (`"Make sure to install httpx using \`pip install httpx[socks]\`."`) — no functional use. |

What this means in practice:

- After PR #2583 lands, the **production** path (`instructor/v2/auto_client.py`) drops `httpx` entirely. Good.
- The **6 coverage test files** will still depend on `import httpx` for their `MockTransport` shim. Those tests will run on whichever httpx-class library is installed, but the user-facing *type surface* of the test fixtures, and the public `from_provider(..., http_client=...)` kwarg's type hint in v2, will be inconsistent — `httpx.Client` in tests, "no httpx type" in the public signature.
- The test in `test_auto_client.py` at line 573 still tells users to `pip install httpx[socks]` — once the production path no longer imports httpx, that guidance stops being correct for users who only have httpx2 transitively.

## What a small follow-up could look like

Two low-touch options, depending on how strictly you want to keep the public API decoupled from a specific HTTP library:

- **Option A — dual import, library-style (recommended here)**: in the few places that still mention `httpx` directly, prefer httpx2 when present and fall back to httpx, mirroring the [OpenAI Python SDK's](https://github.com/openai/openai-python) approach:

  ```python
  try:
      import httpx2 as httpx
  except ModuleNotFoundError:
      import httpx
  ```

  with `httpx2>=2.12.0; python_version >= "3.10"` added as a `dev` optional dep (the `tests/coverage/*` suites already require `httpx` to be installed in dev environments; an `httpx2; python_version >= "3.10"` line alongside would let modern dev envs install httpx2 only).

- **Option B — keep tests pinning `httpx<0.29`, don't change production**: this is essentially "let the OpenAI 3 transport be httpx2, and let the test mocks use real httpx 0.28.x." Easiest to land, but means dev/CI environments will keep two HTTP stacks (httpx2 for the SDK, httpx for the mocks).

The migration guide is short — `httpx2` keeps the same `Client` / `AsyncClient` / `MockTransport` / `Request` / `Response` / `Timeout` public surface, so test fixtures don't need any rewrites, just the import line (or nothing, if you go with Option B).

⚠️ **One caveat worth calling out**: httpx2 verifies TLS against the **OS trust store** instead of the bundled `certifi`. instructor is run against hosted LLMs in many containerised setups and behind corporate proxies; if any of the test fixtures rely on `trust_env=False` to bypass a proxy (which several of them do, per the grep above), that already isolates them from the TLS change, so the practical impact is low. Worth a line in the changelog either way.

## Notes

- The work tracked under issue **#2553** and PR **#2583** is the main thing here — this HTTPXodus issue is essentially a follow-up "and don't forget the test side / public type hint in v2."
- instructor `requires-python = "<4.0,>=3.9"`, so any dependency on `httpx2` must be gated with `python_version >= "3.10"` (httpx2's own floor is Python 3.10) — this is what keeps the 3.9 install path working.
- Per the project's `AGENT.md` / `CLAUDE.md`, tests use real API calls by default; the `tests/coverage/*` files using `httpx.MockTransport` are an exception worth handling deliberately, not silently.
- **Repo URL note**: this is filed against `instructor-ai/instructor`, but that org redirects 301 → `https://github.com/567-labs/instructor` (the project's actual home, 13.8k★). Opening the PR against `567-labs/instructor` is the right move.

Happy to open a PR for Option A on top of #2583 if the maintainers are interested — and equally happy to close this if the test-side httpx2 support is already on the roadmap. No pressure; the goal is just to make sure the decision is an informed one. 🙏
