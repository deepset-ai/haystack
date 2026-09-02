# HTTPXodus 目标看板

生成时间：由 recon.py 生成，共 55 个候选

## 🎯 战役进行状态（人工维护）

| 仓库 | 状态 | Issue | PR | 备注 |
|------|------|-------|----|------|
| Significant-Gravitas/AutoGPT | 🔄 PR 已发，CI 修复中 | [#14268](https://github.com/Significant-Gravitas/AutoGPT/issues/14268) | [#14271](https://github.com/Significant-Gravitas/AutoGPT/pull/14271) | poetry.lock 版本不匹配，用 poetry 2.2.1 重新生成 |

## 📋 待完成队列（按优先级）

| # | 仓库 | ★ | httpx 状态 | 说明 |
|---|------|---|-----------|------|
| 1 | [mem0ai/mem0](https://github.com/mem0ai/mem0) | 64.5k | uses-httpx | AI 记忆层，零竞争 |
| 2 | [lm-sys/fastchat](https://github.com/lm-sys/fastchat) | 39.5k | uses-httpx | LLM 评测老项目，零竞争 |
| 3 | [chroma-core/chroma](https://github.com/chroma-core/chroma) | 29.2k | uses-httpx | 向量数据库标准件，零竞争 |
| 4 | [reflex-dev/reflex](https://github.com/reflex-dev/reflex) | 28.9k | uses-httpx | Python Web 框架，零竞争 |
| 5 | [flet-dev/flet](https://github.com/flet-dev/flet) | 16.6k | uses-httpx | Flutter Python 封装，零竞争 |

## 侦察结果（自动生成）

| # | 仓库 | ★ | httpx 状态 | 已有 httpx2 issue/PR | 证据 |
|---|------|---|-----------|---------------------|------|
| 1 | [Significant-Gravitas/AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) | 187064 | uses-httpx | — | `autogpt_platform/autogpt_libs/pyproject.toml:22,autogpt_plat` |
| 2 | [microsoft/markitdown](https://github.com/microsoft/markitdown) | 177638 | unknown | — | `no manifests found` |
| 3 | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 145472 | uses-httpx | PR#39888(closed), PR#39613(closed), PR#39887(closed), PR#40130(open), PR#40123(open) | `libs/partners/mistralai/pyproject.toml:28,libs/partners/mist` |
| 4 | [browser-use/browser-use](https://github.com/browser-use/browser-use) | 112009 | uses-httpx | PR#5520(open), PR#5511(open), issue#5333(open) | `pyproject.toml:21` |
| 5 | [fastapi/fastapi](https://github.com/fastapi/fastapi) | 102012 | dual | PR#16285(closed), PR#16121(closed), PR#15729(closed), issue#16254(closed), PR#15827(closed) | `httpx2@pyproject.toml:149; httpx@pyproject.toml:64,pyproject` |
| 6 | [OpenBB-finance/OpenBB](https://github.com/OpenBB-finance/OpenBB) | 72590 | unknown | — | `no manifests found` |
| 7 | [mem0ai/mem0](https://github.com/mem0ai/mem0) | 64553 | uses-httpx | — | `pyproject.toml:20` |
| 8 | [microsoft/autogen](https://github.com/microsoft/autogen) | 60751 | uses-httpx | issue#8014(open), PR#8020(open) | `python/packages/autogen-ext/pyproject.toml:143,python/packag` |
| 9 | [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | 57979 | uses-httpx | issue#6750(open) | `pyproject.toml:231` |
| 10 | [BerriAI/litellm](https://github.com/BerriAI/litellm) | 57787 | uses-httpx | issue#38075(open), issue#35306(open), PR#31230(open), PR#30551(closed), PR#33902(closed) | `pyproject.toml:18,pyproject.toml:265` |
| 11 | [run-llama/llama_index](https://github.com/run-llama/llama_index) | 51975 | uses-httpx | PR#22557(closed), issue#22515(closed), PR#22535(closed), PR#22922(closed) | `llama-index-utils/llama-index-utils-qianfan/pyproject.toml:3` |
| 12 | [gradio-app/gradio](https://github.com/gradio-app/gradio) | 43452 | uses-httpx | issue#13705(open) | `requirements.txt:8,requirements.txt:21` |
| 13 | [agno-agi/agno](https://github.com/agno-agi/agno) | 42004 | uses-httpx | PR#9269(open), issue#9267(open), PR#9686(closed), PR#8210(closed) | `libs/agnoctl/pyproject.toml:27,libs/agno_infra/pyproject.tom` |
| 14 | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | 40891 | uses-httpx | PR#8104(closed), PR#8105(closed), PR#7971(closed), PR#8251(closed) | `libs/cli/pyproject.toml:16,libs/sdk-py/pyproject.toml:15,lib` |
| 15 | [lm-sys/fastchat](https://github.com/lm-sys/fastchat) | 39522 | uses-httpx | — | `pyproject.toml:16` |
| 16 | [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | 37716 | no-manifest-hit | — | `pyproject.toml` |
| 17 | [openai/openai-python](https://github.com/openai/openai-python) | 31535 | migrated | — | `pyproject.toml:12` |
| 18 | [python-telegram-bot/python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) | 29443 | uses-httpx | issue#5258(open) | `pyproject.toml:42,pyproject.toml:43,pyproject.toml:75,pyproj` |
| 19 | [assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher) | 29247 | uses-httpx | PR#2086(open) | `pyproject.toml:69,pyproject.toml:70,pyproject.toml:71,requir` |
| 20 | [chroma-core/chroma](https://github.com/chroma-core/chroma) | 29204 | uses-httpx | — | `pyproject.toml:10,requirements.txt:4` |
| 21 | [reflex-dev/reflex](https://github.com/reflex-dev/reflex) | 28869 | uses-httpx | — | `pyproject.toml:25` |
| 22 | [deepset-ai/haystack](https://github.com/deepset-ai/haystack) | 26393 | uses-httpx | PR#12336(closed), PR#12348(closed), PR#12332(closed) | `pyproject.toml:57,pyproject.toml:126` |
| 23 | [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) | 24179 | dual | issue#3332(open), PR#3338(open), PR#3397(open), issue#3238(open), PR#3098(open) | `httpx2@pyproject.toml:133; httpx@pyproject.toml:110` |
| 24 | [gventuri/pandas-ai](https://github.com/gventuri/pandas-ai) | 23789 | no-manifest-hit | — | `pyproject.toml` |
| 25 | [PrefectHQ/prefect](https://github.com/PrefectHQ/prefect) | 23761 | dual | issue#22841(open), PR#22475(open), PR#22683(closed), PR#22681(closed), PR#22673(closed) | `httpx2@pyproject.toml:338,pyproject.toml:339; httpx@pyprojec` |
| 26 | [marimo-team/marimo](https://github.com/marimo-team/marimo) | 22592 | dual | PR#10587(closed), PR#10427(closed), PR#10581(closed), issue#1(open) | `httpx2@pyproject.toml:120; httpx@pyproject.toml:36,pyproject` |
| 27 | [jina-ai/jina](https://github.com/jina-ai/jina) | 21860 | no-manifest-hit | — | `pyproject.toml,requirements.txt,setup.py` |
| 28 | [pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai) | 19652 | uses-httpx | PR#7079(open), issue#7816(open), issue#7805(open), issue#7808(open), issue#6661(open) | `pyproject.toml:193` |
| 29 | [flet-dev/flet](https://github.com/flet-dev/flet) | 16633 | uses-httpx | — | `sdk/python/packages/flet/pyproject.toml:13,sdk/python/exampl` |
| 30 | [unstructured-io/unstructured](https://github.com/unstructured-io/unstructured) | 15378 | no-manifest-hit | — | `pyproject.toml` |
| 31 | [instructor-ai/instructor](https://github.com/instructor-ai/instructor) | 13820 | uses-httpx | — | `requirements.txt:15,requirements.txt:22,requirements.txt:39,` |
| 32 | [encode/starlette](https://github.com/encode/starlette) | 12584 | dual | — | `httpx2@pyproject.toml:46,pyproject.toml:53; httpx@pyproject.` |
| 33 | [microsoft/promptflow](https://github.com/microsoft/promptflow) | 11234 | uses-httpx | — | `src/promptflow-devkit/pyproject.toml:48` |
| 34 | [ollama/ollama-python](https://github.com/ollama/ollama-python) | 10477 | uses-httpx | issue#722(open) | `pyproject.toml:10,requirements.txt:11,requirements.txt:15,re` |
| 35 | [anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) | 3876 | dual | PR#1768(open), issue#1755(closed), PR#1864(closed), PR#1866(closed) | `httpx2@pyproject.toml:12,pyproject.toml:90; httpx@pyproject.` |
| 36 | [roman-right/beanie](https://github.com/roman-right/beanie) | 2695 | uses-httpx | — | `pyproject.toml:50` |
| 37 | [supabase/supabase-py](https://github.com/supabase/supabase-py) | 2574 | uses-httpx | — | `src/auth/pyproject.toml:20,src/functions/pyproject.toml:16,s` |
| 38 | [scholarly-python-package/scholarly](https://github.com/scholarly-python-package/scholarly) | 1878 | uses-httpx | — | `requirements.txt:7,setup.py:33` |
| 39 | [qdrant/qdrant-client](https://github.com/qdrant/qdrant-client) | 1353 | uses-httpx | — | `pyproject.toml:21` |
| 40 | [langchain-ai/langsmith-sdk](https://github.com/langchain-ai/langsmith-sdk) | 1043 | dual | issue#3336(closed), PR#3472(closed), PR#3406(closed), PR#3473(closed), PR#3341(closed) | `httpx2@python/pyproject.toml:30; httpx@python/pyproject.toml` |
| 41 | [poliastro/poliastro](https://github.com/poliastro/poliastro) | 1013 | uses-httpx | — | `pyproject.toml:75` |
| 42 | [replicate/replicate-python](https://github.com/replicate/replicate-python) | 911 | uses-httpx | — | `pyproject.toml:14` |
| 43 | [mistralai/client-python](https://github.com/mistralai/client-python) | 765 | uses-httpx | issue#604(open) | `pyproject.toml:10,pyproject.toml:71` |
| 44 | [groq/groq-python](https://github.com/groq/groq-python) | 611 | uses-httpx | — | `pyproject.toml:12,pyproject.toml:45` |
| 45 | [meilisearch/meilisearch-python](https://github.com/meilisearch/meilisearch-python) | 602 | no-manifest-hit | — | `pyproject.toml` |
| 46 | [modal-labs/modal-client](https://github.com/modal-labs/modal-client) | 513 | uses-httpx | — | `py/pyproject.toml:46` |
| 47 | [langfuse/langfuse-python](https://github.com/langfuse/langfuse-python) | 462 | uses-httpx | — | `pyproject.toml:11` |
| 48 | [pinecone-io/pinecone-python-client](https://github.com/pinecone-io/pinecone-python-client) | 449 | uses-httpx | — | `pyproject.toml:35,pyproject.toml:36,pyproject.toml:37,pyproj` |
| 49 | [cohere-ai/cohere-python](https://github.com/cohere-ai/cohere-python) | 397 | uses-httpx | — | `pyproject.toml:43,pyproject.toml:44,pyproject.toml:68,pyproj` |
| 50 | [weaviate/weaviate-python-client](https://github.com/weaviate/weaviate-python-client) | 227 | uses-httpx | — | `setup.cfg:37` |
| 51 | [jaraco/wolframalpha](https://github.com/jaraco/wolframalpha) | 155 | uses-httpx | — | `pyproject.toml:29` |
| 52 | [sanic-org/sanic-testing](https://github.com/sanic-org/sanic-testing) | 36 | uses-httpx | — | `setup.py:51` |
| 53 | [pythongssapi/httpx-gssapi](https://github.com/pythongssapi/httpx-gssapi) | 14 | uses-httpx | issue#57(open) | `setup.cfg:2,setup.cfg:3,setup.cfg:6,setup.cfg:19,setup.cfg:2` |
| 54 | [ulodciv/httpx-ntlm](https://github.com/ulodciv/httpx-ntlm) | 13 | uses-httpx | — | `requirements.txt:1,setup.py:7,setup.py:9,setup.py:11,setup.p` |
| 55 | typesense/typesense-py | - | ❓ 仓库不可达 | | |
