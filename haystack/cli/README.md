# Haystack CLI

<p align="center" float="left">
  <img alt="" src="https://raw.githubusercontent.com/deepset-ai/.github/main/deepset-logo-colored.png" width="30%"/>
  <img alt="" src="https://raw.githubusercontent.com/deepset-ai/.github/main/haystack-logo-colored-on-dark.png#gh-dark-mode-only" width="30%"/>
  <img alt="" src="https://raw.githubusercontent.com/deepset-ai/.github/main/haystack-logo-colored.png#gh-light-mode-only" width="30%"/>
</p>

<strong><a href="https://github.com/deepset-ai/haystack">Haystack</a></strong> is an open source NLP framework by <strong><a href="https://deepset.ai">deepset</a></strong> to help you build production ready search systems or applications powered by various NLP tasks such as Question Answering. Haystack is designed to help you build systems that work intelligently over large document collections. It achieves this with the concept of <strong>Pipelines</strong> consisting of various <strong>Nodes</strong> such as a <strong>DocumentStore</strong>, a <strong>Retriever</strong> and a <strong>Reader</strong>.


This is the repository where we keep the code for the Haystack CLI.

To contribute to the tutorials please check out our [Contributing Guidelines](./Contributing.md)

## Available commands

### `haystack prompt fetch`

```
Usage: haystack prompt fetch [OPTIONS] [PROMPT_NAME]...

  Downloads a prompt from the official Haystack PromptHub and saves it locally
  to ease use in environments with no network.

  PROMPT_NAME can be specified multiple times.

  PROMPTHUB_CACHE_PATH environment variable can be set to change the default
  folder in which the prompts will be saved in.

  If a custom PROMPTHUB_CACHE_PATH is used, remember to also use it for
  Haystack invocations.

  The Haystack PromptHub is https://prompthub.deepset.ai/

Options:
  --help  Show this message and exit.
```

Example usage:

```
haystack prompt fetch deepset/conversational-agent-with-tools deepset/summarization
```
