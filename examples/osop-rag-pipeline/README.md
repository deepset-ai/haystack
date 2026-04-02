# Haystack RAG Pipeline — OSOP Workflow Example

This directory contains a portable [OSOP](https://github.com/osopcloud/osop-spec) workflow definition for a typical Haystack RAG (Retrieval-Augmented Generation) pipeline.

## What is OSOP?

**OSOP** (Open Standard for Orchestration Protocols) is a YAML-based format for describing multi-step workflows in a tool-agnostic way. It lets you define pipelines, agent workflows, and automation flows that can be understood by any compatible runtime — including Haystack, LangChain, Prefect, and others.

Think of it as the **OpenAPI of workflows**: a single `.osop` file describes what your pipeline does, so teams can share, review, and port workflows across tools.

## Pipeline Overview

The `haystack-rag-pipeline.osop` file describes a standard RAG pipeline:

```
User Query → Query Embedder → Document Retriever → Prompt Builder → Answer Generator → Response
```

| Step | OSOP Node Type | Haystack Equivalent |
|------|---------------|---------------------|
| User Query | `human` | Pipeline input |
| Query Embedder | `agent` | `OpenAITextEmbedder` |
| Document Retriever | `db` | `QdrantEmbeddingRetriever` |
| Prompt Builder | `system` | `PromptBuilder` |
| Answer Generator | `agent` | `OpenAIGenerator` |
| Response | `api` | Pipeline output |

## Usage

The `.osop` file is a standalone YAML document. You can:

- **Read it** to understand the pipeline at a glance
- **Validate it** with the [OSOP CLI](https://github.com/osopcloud/osop): `osop validate haystack-rag-pipeline.osop`
- **Visualize it** with the [OSOP Editor](https://github.com/osopcloud/osop-editor)
- **Use it as a reference** when building the equivalent Haystack pipeline in Python

## Links

- [OSOP Spec](https://github.com/osopcloud/osop-spec)
- [OSOP CLI](https://github.com/osopcloud/osop)
- [Haystack Documentation](https://docs.haystack.deepset.ai/)
