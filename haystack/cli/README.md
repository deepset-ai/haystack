# Haystack CLI

<p align="center" float="left">
  <img alt="" src="https://raw.githubusercontent.com/deepset-ai/.github/main/deepset-logo-colored.png" width="30%"/>
  <img alt="" src="https://raw.githubusercontent.com/deepset-ai/.github/main/haystack-logo-colored-on-dark.png#gh-dark-mode-only" width="30%"/>
  <img alt="" src="https://raw.githubusercontent.com/deepset-ai/.github/main/haystack-logo-colored.png#gh-light-mode-only" width="30%"/>
</p>

<strong><a href="https://github.com/deepset-ai/haystack">Haystack</a></strong> is an open source NLP framework by <strong><a href="https://deepset.ai">deepset</a></strong> to help you build production ready search systems or applications powered by various NLP tasks such as Question Answering. Haystack is designed to help you build systems that work intelligently over large document collections. It achieves this with the concept of <strong>Pipelines</strong> consisting of various <strong>Nodes</strong> such as a <strong>DocumentStore</strong>, a <strong>Retriever</strong> and a <strong>Reader</strong>.


This is the repository where we keep the code for the Haystack CLI.

To contribute to the tutorials please check out our [Contributing Guidelines](./Contributing.md)

## Available CLI commands

### `haystack prompt fetch`

**Usage:**

```
haystack prompt fetch [OPTIONS] [PROMPT_NAME]
```

Download a prompt from the official [Haystack PromptHub](https://prompthub.deepset.ai/) and save it locally
for easier use in environments with no network.

You can specify multiple prompts to fetch at the same time.

PROMPTHUB_CACHE_PATH environment variable can be set to change the default
folder in which the prompts will be saved in. You can find the default cache path on your machine by running the following code:

  ``` python
  from haystack.nodes.prompt.prompt_template import PROMPTHUB_CACHE_PATH
  print(PROMPTHUB_CACHE_PATH)
  ```

If you set a custom PROMPTHUB_CACHE_PATH environment variable, remember to set it to the same value in your console before running Haystack.

**Example:**

```
haystack prompt fetch deepset/conversational-agent-with-tools deepset/summarization
```

**Options:**

`--help`  Show options and exit.

### `haystack --version`

  Show your current Haystack version and exit.