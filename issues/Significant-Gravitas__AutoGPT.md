# Issue 草稿：Significant-Gravitas/AutoGPT
> 状态：✅ 已发出 → https://github.com/Significant-Gravitas/AutoGPT/issues/14268 （2026-09-02）
> 查重：2026-09-02 无任何 httpx2 相关 issue/PR ✅

---

**Title:** [HTTPXodus] Consider migrating from `httpx` to `httpx2` (the actively maintained fork)

**Body:**

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

httpx shows up in AutoGPT in a somewhat unusual way, which makes this worth a look regardless of the migration question:

- **22 Python files import `httpx` directly**, including production code — most notably the inter-service RPC layer in `autogpt_platform/backend/backend/util/service.py`, which builds `httpx.Client` / `httpx.AsyncClient` with custom connection `Limits` (max 500 connections, keep-alive tuning), plus blocks (`stripe_link`, `exa/websets`) and copilot adapters (Slack / Teams / Telegram).
- Yet `httpx = "^0.28.1"` is declared **only under `[tool.poetry.group.dev.dependencies]`** in both `autogpt_platform/backend/pyproject.toml` and `autogpt_platform/autogpt_libs/pyproject.toml` — production code currently relies on httpx arriving **transitively** (via openai, supabase, langsmith, …). If any of those upstreams drop or swap their HTTP layer, AutoGPT breaks without anyone touching a line of AutoGPT code.

Since AutoGPT is an application (not a published library) and already requires **Python >=3.10,<3.14**, a clean migration is feasible:

- **Step 1 (independent of migration)**: promote `httpx` to a real runtime dependency in `backend/pyproject.toml` — it is imported by production code.
- **Step 2 — the switch**: replace `httpx = "^0.28.1"` with `httpx2 = "^2.12"` and change imports to `import httpx2 as httpx` (or the dual-import fallback if you'd rather keep both working during a transition window). The public API is compatible — `Client`, `AsyncClient`, `Limits`, `HTTPStatusError`, `Response.raise_for_status()` all behave the same.

⚠️ **One caveat worth calling out**: httpx2 verifies TLS against the **OS trust store** instead of the bundled `certifi`. Self-hosted AutoGPT deployments behind corporate proxies or in minimal containers that relied on certifi's CA bundle may need `SSL_CERT_FILE` or system CA configuration. Worth a line in the deployment docs.

## Notes

- No existing httpx2-related issue or PR in this repo as of 2026-09-02 (checked via GitHub search).
- The `classic/` tree also uses httpx (`classic/forge/forge/components/web/web_fetch.py`), but it appears to be legacy; the migration scope above covers `autogpt_platform/`.

Happy to open a PR for either step if the maintainers are interested — and equally happy to close this if you'd rather wait for httpx 1.0 stable. No pressure; the goal is just to make sure the decision is an informed one. 🙏
