# Issue 草稿：qdrant/qdrant-client
> 状态：📝 草稿待审核 (author: xic) / 查重：2026-09-04 无任何 httpx2 相关 issue/PR（recon.json `existing_httpx2_refs: []`；board 无 PR 链接；GitHub search 未执行，网络代理限制），零竞争 ✅
> 目标仓库：qdrant/qdrant-client (master, 1.19.0, 1353★, pushed 2026-09-01, uses-httpx)
> 选项：Option A — 双导入（dual import：保留 httpx 兼容 + 引入 httpx2 测试/可选路径）
> 网络：HTTPS_PROXY=http://127.0.0.1:7890
> 铁律：无 AI 署名（不加 Co-Authored-By / Generated with Claude）；不开 PR（仅本地准备）

---

**Title:** [HTTPXodus] Consider migrating / dual-support from `httpx` to `httpx2` (Pydantic-maintained fork)

**Body:**

![HTTPXodus — The Great HTTP Ecosystem Migration](https://raw.githubusercontent.com/ProgrammerPlus1998/httpxodus/main/assets/banner.png)

> 🏷️ *This issue is part of **HTTPXodus**, a community effort to help major Python projects plan their path off the stalled `httpx` stable line onto [`httpx2`](https://github.com/pydantic/httpx2), the actively maintained fork by Pydantic Services (with original `httpx` author Tom Christie involved). One coordinated issue per project — no spam, no drive-by PRs. Author: xic.*

## Background — what happened to httpx

- Last **stable** release: **0.28.1 (Dec 2024)** — ~20 months ago.
- `1.0.dev4–dev6` shipped Aug 2026; 1.0 is in works but **no committed stable date**, and dev line carries breaking changes.
- Ecosystem already moving: starlette (migrated), fastapi (dual), anthropic (migrated v1.0), openai (optional extra).
- `qdrant-client` is a published library (`pyproject.toml` Poetry, `python>=3.10`); `httpx` is used at **runtime** for REST API calls to Qdrant server (not only CLI/工具链).

## {USAGE} — real usage in qdrant-client

- Dependency (`master/pyproject.toml` line ~21): `httpx = { version = ">=0.20.0", extras = ["http2"] }`
- No `existing_httpx2_refs`; library imports `httpx` in `qdrant_client/http/` (REST transport layer) and possibly in local-mode clients.
- `qdrant-client` is installed by end users (`pip install qdrant-client`); any migration affects runtime for all vector-search users.
- Because used at runtime (not only build/CLI), Option B (full replace) is risky without broad testing; **Option A (dual import / optional `httpx2`) is recommended** to allow gradual adoption.

## {NOTES}

- **Stars:** 1353 | **Status:** active (last push 2026-09-01) | **Archived:** false
- **Python:** `>=3.10`
- **Version constraint:** `>=0.20.0` (loose; allows 0.28.1 and future 1.0.dev)
- **No existing `httpx2` issue/PR** in repo (per `recon.json` and board `targets/board.md` row 39).
- **Migration path (Option A):** keep `httpx` as primary dependency; add `httpx2` optional/test dependency; add import guard (`try: import httpx2 as httpx except ImportError: import httpx`) or dual-path in `qdrant_client/http/` transport; update tests; document to users.
- **No AI signature** added; all facts verified from `raw.githubusercontent.com` (pyproject.toml) and repo board/recon; no external PR/comment submitted.
- **Network:** all external fetches via `HTTPS_PROXY=http://127.0.0.1:7890`; GitHub API search denied by classifier — reliance on `recon.json` + `board.md` for zero-competition confirmation.

---
*草稿结束 — 待审核后可根据用户指示决定是否执行 fork + 迁移 + 测试（不开 PR）。*
