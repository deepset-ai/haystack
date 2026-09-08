<div align="center">
  <a href="https://haystack.deepset.ai/"><img src="https://raw.githubusercontent.com/deepset-ai/haystack/main/images/banner.png" alt="Blue banner with the Haystack logo and the text ‘haystack by deepset – The Open Source AI Framework for Production Ready RAG & Agents’ surrounded by abstract icons representing search, documents, agents, pipelines, and cloud systems."></a>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CI/CD   | [![Tests](https://github.com/deepset-ai/haystack/actions/workflows/tests.yml/badge.svg)](https://github.com/deepset-ai/haystack/actions/workflows/tests.yml) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) [![Coverage badge](https://raw.githubusercontent.com/deepset-ai/haystack/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/deepset-ai/haystack/blob/python-coverage-comment-action-data/htmlcov/index.html) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Docs    | [![Website](https://img.shields.io/website?label=documentation&up_message=online&url=https%3A%2F%2Fdocs.haystack.deepset.ai)](https://docs.haystack.deepset.ai)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Package | [![PyPI](https://img.shields.io/pypi/v/haystack-ai)](https://pypi.org/project/haystack-ai/) ![PyPI - Downloads](https://img.shields.io/pypi/dm/haystack-ai?color=blue&logo=pypi&logoColor=gold) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/haystack-ai?logo=python&logoColor=gold) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/haystack-ai.svg)](https://anaconda.org/conda-forge/haystack-ai) [![GitHub](https://img.shields.io/github/license/deepset-ai/haystack?color=blue)](LICENSE) [![License Compliance](https://github.com/deepset-ai/haystack/actions/workflows/license_compliance.yml/badge.svg)](https://github.com/deepset-ai/haystack/actions/workflows/license_compliance.yml) [![HVTrust](https://hvtracker.net/badge/haystack.svg)](https://hvtracker.net/agents/haystack/) [![Evidence Grade](https://hvtracker.net/badge/haystack-grade.svg)](https://hvtracker.net/agents/haystack/) [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/13067/badge)](https://www.bestpractices.dev/projects/13067)|
| Meta    | [![Discord](https://img.shields.io/discord/993534733298450452?logo=discord)](https://discord.com/invite/qZxjM4bAHU) [![Twitter Follow](https://twitter.com/haystack_ai)](https://twitter.com/haystack_ai)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
</div>

<div align="center">

# 🎉🎊✨ &nbsp; Haystack 3.0 正式发布！ &nbsp; ✨🎊🎉

### 点击[此处](https://haystack.deepset.ai/blog/haystack-3-release)阅读官方发布公告！

## 🥳 🎈 🎆 🪅 🎇 🍾 🥂 🎁 🌈

</div>

[Haystack](https://haystack.deepset.ai/) 是一套开源的 AI 编排框架，专为使用 Python 构建生产级大语言模型（LLM）应用程序而设计。

支持构建模块化管道（Pipelines）与智能体工作流（Agent Workflows），对检索、路由分发、上下文记忆和模型生成拥有完全显式的控制力。构建高可扩展的 RAG 系统、多模态应用、语义搜索、智能问答以及自主智能体——其高度透明的架构支持深度定制、自由实验与稳健投产。

## 目录

- [安装指南](#安装指南)
- [官方文档](#官方文档)
- [核心特性](#核心特性)
- [Haystack 企业级服务：技术支持与平台](#haystack-企业级服务技术支持与平台)
- [遥测说明](#遥测说明)
- [🖖 社区与交流](#-社区与交流)
- [参与贡献](#参与贡献)
- [采用 Haystack 的企业与组织](#采用-haystack-的企业与组织)


## 安装指南

获取 Haystack 最简单的方式是通过 pip：

```sh
pip install haystack-ai
```

安装 nightly 抢先体验版以试用最新特性：
```sh
pip install --pre haystack-ai
```

Haystack 支持多种安装方式，包括 Docker 镜像。如需查看完整安装指南，请参阅[官方文档](https://docs.haystack.deepset.ai/docs/installation)。

## 官方文档

如果你是初次接触本项目，建议先阅读[“什么是 Haystack？”](https://haystack.deepset.ai/overview/intro)，随后按照[“快速上手指南”](https://haystack.deepset.ai/overview/quick-start)的指引，只需几分钟即可构建你的第一个 LLM 应用程序。通过[实战教程](https://haystack.deepset.ai/tutorials)持续深入探索。对于更高级的用例，或为了获取实现灵感，欢迎浏览 [Cookbook 菜谱库](https://haystack.deepset.ai/cookbook)中提供的 Haystack 最佳实践代码。

你可随时查阅[官方技术文档](https://docs.haystack.deepset.ai/docs/intro)，深入了解 Haystack 的各项功能、使用方式及其背后的底层技术原理。

## 核心特性

**专为生产环境打造的智能体（Agents built for production）**  
通过生命周期钩子（Lifecycle Hooks，如 `before_llm`、`before_tool`、`on_exit` 等）扩展 Agent 行为，轻松嵌入安全护栏（Guardrails）与自定义业务逻辑；开箱即用地原生追踪 `step_count`（步骤数）、`token_usage`（Token 用量）与工具调用详情，实现全流程监控与精准成本控制。利用来自 [Agent Pack](https://github.com/deepset-ai/haystack-core-integrations/tree/main/integrations/agent_pack) 的开箱即用型智能体快速启动项目（例如深度研究 Agent 或高级 RAG Agent），或通过 `SkillToolset` 为自定义智能体赋予“渐进式技能发现（Progressive Skill Discovery）”能力，让工具/技能的描述信息仅在必要时才注入上下文窗口。

**专为上下文工程构建（Built for context engineering）**  
构建高灵活性的系统架构，对信息在送入模型之前的检索、重排（Ranking）、过滤、合并、结构化与路由分发流程进行显式精确掌控。定义透明可溯的管道与智能体工作流，确保检索、记忆、工具与生成的每一步都清晰可追踪。

**原生异步支持（Native Async Support）**  
同一条 `Pipeline` 可以同步或异步执行，并支持逐 Token 流式输出。`Agent` 能够并行执行并发工具调用。

**高度模块化与灵活定制（Modular and customizable）**  
利用内置组件实现检索、索引、工具调用、上下文记忆与系统评估，或轻松创建自定义组件。灵活添加循环（Loops）、分支（Branches）与条件逻辑，精准控制上下文在管道与 Agent 工作流中的流转路径。

**与模型及供应商完全解耦（Model- and vendor-agnostic）**  
广泛兼容 OpenAI、Mistral、Anthropic、Cohere、Hugging Face、Google、Azure OpenAI、AWS Bedrock 以及各类本地运行的模型。随心更换模型底座或基础设施组件，无需重写应用逻辑。

**可扩展的开源生态（Extensible ecosystem）**  
通过高度一致的标准化接口构建与共享自定义组件，方便社区与第三方生态全面扩展 Haystack，共建繁荣的开放生态。

> [!TIP]
>
> 想要将 Haystack 管道作为 **REST API** 或 **MCP 服务器（Model Context Protocol）** 部署与服务化吗？[Hayhooks](https://github.com/deepset-ai/hayhooks) 提供了便捷的封装途径，允许你使用自定义逻辑包装管道和智能体，并通过 HTTP 端点或 MCP 协议对外暴露。它还全面支持与 OpenAI 兼容的聊天补全端点，并能无缝接入 [open-webui](https://openwebui.com/) 等前端界面。

## Haystack 企业级服务：技术支持与平台

通过 **Haystack Enterprise Starter** 获取来自 Haystack 官方团队的专家级技术支持，利用企业级参考模板加速研发，并通过针对云端和私有化环境的部署指南实现安全可靠的规模化扩展。详情请参阅[官方发布博客](https://haystack.deepset.ai/blog/announcing-haystack-enterprise)。

👉 [获取 Haystack Enterprise Starter](https://www.deepset.ai/products-and-services/haystack-enterprise-starter?utm_source=github.com&utm_medium=referral&utm_campaign=haystack_enterprise)

需要针对 Haystack 的全托管生产环境？**Haystack Enterprise Platform** 提供了内置的可观测性（Observability）、团队协作、合规治理与访问控制体系，全面协助你构建、测试、部署与运维 Haystack 管道。它支持全托管云服务或私有化自托管部署。

👉 了解关于 [Haystack Enterprise Platform](https://www.deepset.ai/products-and-services/haystack-enterprise-platform?utm_campaign=developer-relations&utm_source=haystack&utm_medium=readme) 的更多信息或[申请免费试用](https://www.deepset.ai/haystack-enterprise-platform-trial?utm_campaign=developer-relations&utm_source=haystack&utm_medium=readme)

## 遥测说明

Haystack 会收集管道组件的**匿名**使用统计信息。每当这些组件被初始化时，我们会收到一条事件通知。通过这种方式，我们能够了解哪些组件对社区最具实用价值。

关于 Haystack 遥测机制的详细说明以及如何选择退出（Opt-out），请参阅 [Haystack 官方文档](https://docs.haystack.deepset.ai/docs/telemetry)。

## 🖖 社区与交流

如果你有新功能建议或发现了缺陷（Bug），欢迎随时在 [GitHub Issues](https://github.com/deepset-ai/haystack/issues) 提交工单。我们定期审查这些反馈并会迅速回应。如果你想就某个主题展开深入讨论，或就如何将 Haystack 应用于你的具体项目获取更广泛的建议，可以在 [Github Discussions](https://github.com/deepset-ai/haystack/discussions) 发起主题，或加入我们的 [Discord 频道](https://discord.com/invite/VBpFzsgRVF)。我们也活跃在 [𝕏 (Twitter)](https://twitter.com/haystack_ai) 与 [Stack Overflow](https://stackoverflow.com/questions/tagged/haystack)。

## 参与贡献

我们非常欢迎来自社区的贡献——无论是修正排版拼写错误，还是贡献全新的功能特性！你无需成为 Haystack 专家即可提供有价值的改进。若要了解如何开启贡献，请先查阅我们的[贡献者指南 (Contributor Guidelines)](https://github.com/deepset-ai/haystack/blob/main/CONTRIBUTING.md)。

你可以通过多种方式参与 Haystack 建设：
- 为 Haystack 核心项目贡献代码与改进
- 在 [haystack-core-integrations](https://github.com/deepset-ai/haystack-core-integrations) 仓库贡献生态组件与集成
- 在 [haystack/docs-website](https://github.com/deepset-ai/haystack/tree/main/docs-website) 完善文档页面

> [!TIP]
> 👉 **[查看当前所有面向社区开放认领的 Issues 清单](https://github.com/orgs/deepset-ai/projects/14)**

## 采用 Haystack 的企业与组织

Haystack 已被各行各业数以千计的工程团队采用，用于构建生产级 AI 系统，其中包括：

- **科技与 AI 基础设施**：[Apple](https://www.apple.com/)、[Meta](https://www.meta.com/about)、[Databricks](https://www.databricks.com/)、[NVIDIA](https://developer.nvidia.com/blog/reducing-development-time-for-intelligent-virtual-assistants-in-contact-centers/)、[Intel](https://github.com/intel/open-domain-question-and-answer#readme)
- **公共部门 AI 倡议**：[欧盟委员会 (European Commission)](https://commission.europa.eu/index_en)、[德国联邦教育与研究部 (BMFTR)](https://www.deepset.ai/case-studies/german-federal-ministry-research-technology-space-bmftr)、[巴登-符腾堡州 PD 咨询](https://www.pd-g.de/)
- **企业级与工业 AI 应用**：[空中客车 (Airbus)](https://www.deepset.ai/case-studies/airbus)、[汉莎行业解决方案 (Lufthansa Industry Solutions)](https://haystack.deepset.ai/blog/lufthansa-user-story)、[英飞凌 (Infineon)](https://www.infineon.com/)、[乐高 (LEGO)](https://github.com/larsbaunwall/bricky#readme)、[康卡斯特 (Comcast)](https://arxiv.org/html/2405.00801v2)、[埃森哲 (Accenture)](https://www.accenture.com/)、[TELUS 农业与消费品](https://www.telus.com/agcg/en)
- **知识与内容平台**：[Netflix](https://netflix.com)、[时代在线 (ZEIT Online)](https://www.deepset.ai/case-studies/zeit-online)、[乐天 (Rakuten)](https://www.rakuten.com/)、[牛津大学出版社 (Oxford University Press)](https://corp.oup.com/)、[Manz](https://www.deepset.ai/case-studies/manz)、[YPulse](https://www.deepset.ai/case-studies/ypulse)


你也在生产中使用 Haystack 吗？欢迎提交 PR 或[在此分享你的实践故事](https://forms.gle/Mm3G1aEST3GAH2rn8)。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年9月8日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
