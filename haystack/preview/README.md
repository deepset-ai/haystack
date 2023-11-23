# Haystack 2.0 - Preview Features

[![PyPI - Version](https://img.shields.io/pypi/v/haystack-ai.svg)](https://pypi.org/project/haystack-ai)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/haystack-ai.svg)](https://pypi.org/project/haystack-ai)

Since Haystack 1.15, we’ve been slowly introducing new components and features to Haystack in the background in preparation for Haystack 2.0. In this `preview` module, you can find what’s been implemented so far regarding Haystack 2.0. **Keep in mind that Haystack 2.0 is still a work in progress.** Read more about Haystack 2.0 in [Shaping Haystack 2.0](https://github.com/deepset-ai/haystack/discussions/5568).

## 💾 Installation

**Install `haystack-ai`**

There is a separate PyPI package that only ships the code in `preview` module. You can install `haystack-ai` using pip:
```sh
pip install haystack-ai
```
The `haystack-ai` package is built on the `main` branch, so it's highly unstable, but it's useful if you want to try the new features as soon as they are merged.

**Install `farm-haystack`**

As an alternative way, you can install `farm-haystack`:
```sh
pip install farm-haystack
```
The `farm-haystack` package includes all new features of Haystack 2.0. Note that updates to this package occur less frequently compared to `haystack-ai`. So, you might not get the all latest Haystack 2.0 features immediately when using `farm-haystack`.

## 🚗 Getting Started

In our **end 2 end tests** you can find example code for the following pipelines:
- [RAG pipeline](https://github.com/deepset-ai/haystack/blob/main/e2e/preview/pipelines/test_rag_pipelines.py)
- [Extractive QA pipeline](https://github.com/deepset-ai/haystack/blob/main/e2e/preview/pipelines/test_extractive_qa_pipeline.py)
- more to come, check out the [folder](https://github.com/deepset-ai/haystack/blob/main/e2e/preview/)

## 💙 Stay Updated
To learn how and when components will be migrated to the new major version, have a look at the [Migrate Components to Pipeline v2](https://github.com/deepset-ai/haystack/issues/5265) roadmap item, where we keep track of issues and PRs about Haystack 2.0. When you have questions, you can always contact us using the [Shaping Haystack 2.0](https://github.com/deepset-ai/haystack/discussions/5568) discussion or [Haystack Discord server](https://discord.com/channels/993534733298450452/1141683185458094211).
