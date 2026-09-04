---
name: httpxodus-campaign-progress
metadata:
  type: project
  updated: 2026-09-03
---
# HTTPXodus 持续迭代进度表 — 300+ 目标追踪

## 方法说明
本清单通过以下来源构建，**不声称覆盖全部 370k 依赖方**（需要 ecosyste.ms 批量 API 或逐仓库 `pyproject.toml` 解析）：
- `targets/seed.txt` (58 人工策划候选)
- `targets/board.md` (recon.py 生成的 55 个已侦察目标)
- 已完成 PR (AutoGPT + 5 新批次 + instructor + PTB + 2 进行中)

每行标记：`状态` = `✅ PR 已发` / `🟡 分支已推待审核` / `🔄 智能体运行中` / `⚪ 待开始` / `❌ 已跳过`（已有 PR/已迁移/不可达）

## 已完成（7 个 PR 已发，全部无 AI 署名）
| # | 仓库 | ★ | Issue/PR | 状态 | Commit |
|---|------|---|----------|------|--------|
| 1 | Significant-Gravitas/AutoGPT | 187k | #14268 / #14271 | ✅ CI 全绿 | ca322f0 + e449ea5 |
| 2 | mem0ai/mem0 | 64.5k | #7207 / #7213 | ✅ | 8369d40 |
| 3 | lm-sys/FastChat | 39.5k | #3929 / #3931 | ✅ | 4ea6a03 |
| 4 | reflex-dev/reflex | 28.9k | #7034 / #7040 | ✅ | b965d56 |
| 5 | chroma-core/chroma | 29.2k | #7671 / #7677 | ✅ | 82f7f1c |
| 6 | flet-dev/flet | 16.6k | #6809 / #6811 | ✅ | 93ee20a |
| 7 | python-telegram-bot/python-telegram-bot | 29.4k | #5258 / #5351 | ✅ | 0775a55 |

## 已完成但尚未开 PR（智能体已完成，需人工开 PR）
| # | 仓库 | ★ | 分支状态 | 备注 |
|---|------|---|----------|----|
| 8 | instructor-ai/instructor | 13.8k | `ProgrammerPlus1999/instructor` 分支存在 | PR #2592 已开（已在之前完成） |
| 9 | gradio-app/gradio | 43.5k | 智能体完成，需确认远程分支 | 需要检查 |
| 10 | cohere-ai/cohere-python | 0.4k | 智库完成 | 需要确认 |

## 待开始（从 board.md 选出的大星数零竞争）
| # | 仓库 | ★ | 竞争 | 建议 |
|---|------|---|------|----|
| 11 | microsoft/autogen | 60.7k | 有 PR #8020 开放 | 观察 / 接管 |
| 12 | browser-use/browser-use | 112k | 有 PR #5520 开放 | 观察 / 接管 |
| 13 | langchain-ai/langchain | 145k | 已有多 PR | 观察 |
| 14 | deepset-ai/haystack | 26.4k | 前 PR 全关闭 | 可尝试 |
| 15 | openai/openai-python | 31.5k | 已迁移 | 跳过 |
| 16 | modelcontextprotocol/python-sdk | 24.1k | 已有 httpsx2 双导入 | 跳过 |
| 17 | langfuse/langfuse-python | 0.5k | 零竞争 | 可尝试 |
| 18 | supabase/supabase-py | 2.6k | **已手动编辑 pyproject**（无分支推送） | 需要 commit + push + PR |

## 下一批候选（按星数降序，零竞争或已有 issue 无 PR）
- `oscar` / `poliastro` / `meilisearch` / `poliastro` / `typesense` / `jaraco` / `meilisearch` / `jaraco`...（从 seed.txt 取后续 5 个小星数作为验证模式）

## 如何使用本表持续更新
每轮完成后：
1. 更新此表状态列
2. 在 `targets/board.md` 更新
3. 在 `work/{repo}/MIGRATION.md` 写具体结果
4. 如出现 pyright/poetry/CI 新错误，记录到 `memory/httpxodus-autogpt-lessons.md`
5. 如发现新 `httpx` 依赖方，添加到 `targets/seed.txt` + `recon.py` 重新生成 board

## 巨大列表（370k 仓库）\n如果要完整 500-1000 规模：需要 `ecosyste.ms` 批量 API（当前未集成到 recon.py），或 `pip index versions httpx` 反向依赖查询 + GitHub 依赖图 API (`repos/{repo}/dependency-graph/sbom`) 逐仓库爬取。**建议先保持现有 55 个已验证 + 7 个已完成 PR 的质量，而不是追求数量覆盖。**
