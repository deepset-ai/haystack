# HTTPXodus

> Exodus（出埃及记）+ httpx：带领 Python 生态离开停滞的 `httpx`，走向由 Pydantic 官方维护的 `httpx2`。

## 战役目标

系统性地找出仍在依赖 `httpx` 的最大开源项目，为它们：

1. 提交说明详尽的 migration issue（每个 issue 带 HTTPXodus 署名）
2. Fork → clone → 迁移 → 跑通测试 → 提交 PR

## 背景事实（已核实，2026-09-02）

| 事实 | 数据 |
|------|------|
| httpx 最后一个稳定版 | 0.28.1（2024-12-06） |
| httpx 稳定版停滞时长 | 约 21 个月 |
| httpx 1.0 dev 版 | dev4/dev5/dev6 于 2026-08-19 ~ 08-31 密集发布（1.0 在酝酿中，但无稳定版时间表） |
| httpx 依赖方数量 | 370,630 个仓库（ecosyste.ms） |
| httpx2 维护方 | Pydantic Services Inc.（原作者 Tom Christie 参与） |
| httpx2 当前版本 | 2.12.0（16 个版本，发布节奏约每周一个） |
| 已完成迁移的项目 | Starlette、Anthropic SDK、MCP Python SDK、OpenAI SDK（optional extra） |

## 铁律

1. **先发 issue，后发 PR** —— 大项目惯例，先讨论方案。
2. **先查重，再动手** —— 已有 migration issue/PR 的项目标记为 `duplicate-skip`，绝不重复提交。
3. **诚实叙事** —— issue 中必须如实说明 httpx 1.0 dev 版正在活跃开发，不得夸大"已死"。
4. **每个 PR 发出前必须经本人审核** —— 绝不擅自提交外向操作。
5. **库用双导入，应用才硬切** —— library 类项目优先 `try: import httpx2 as httpx` 双导入模式；httpx2 的 TLS 改用 OS trust store，对容器/企业代理环境是行为变更，issue 中必须提示。
6. **被拒要优雅** —— 维护者拒绝就在看板标记 `declined`，不纠缠。

## 目录结构

```
httpxodus/
├── README.md            # 本文件，战役宪章
├── docs/
│   └── issue-template.md  # issue 统一模板（所有 issue 以此为准）
├── targets/
│   ├── seed.txt         # 候选目标种子清单（人工策划）
│   └── board.md         # 侦察生成的目标看板（自动更新）
├── scripts/
│   └── recon.py         # 侦察脚本：验证依赖、查星数、查重
└── issues/              # 每个目标的 issue 草稿（审核后发出）
```

## 工作流

```
种子清单 → recon.py 验证排名 → 人工挑选目标
        → 写 issue 草稿(issues/{repo}.md) → 审核 → gh 发 issue
        → fork + clone → 迁移 → 测试 → 审核 → 发 PR
        → 看板跟踪状态直至合入或关闭
```
