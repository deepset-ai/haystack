# Canals

<p align="center" float="left">
  <img alt="" src="https://raw.githubusercontent.com/deepset-ai/.github/main/deepset-logo-colored.png" width="30%"/>
  <img alt="" src="https://raw.githubusercontent.com/deepset-ai/.github/main/haystack-logo-colored-on-dark.png#gh-dark-mode-only" width="30%"/>
  <img alt="" src="https://raw.githubusercontent.com/deepset-ai/.github/main/haystack-logo-colored.png#gh-light-mode-only" width="30%"/>
</p>

[![PyPI - Version](https://img.shields.io/pypi/v/canals.svg)](https://pypi.org/project/canals)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/canals.svg)](https://pypi.org/project/canals)

Canals is a nodes orchestration engine. Nodes are classes that can execute a task, like reading a file, performing calculations, or making API calls. Canals connects these objects together: it builds a graph of dependencies and takes care of managing their execution order, making sure that each object receives the input it expects from the other nodes of the pipeline.

Canals powers version 2.0 of the [Haystack framework](https://github.com/deepset-ai/haystack).

## Installation

```console
pip install canals
```

gets you the bare minimum necessary to run Canals.

To be able to draw pipelines, please make sure you have [graphviz](https://graphviz.org/download/) installed:

```console
sudo apt install graphviz
# You may need `graphviz-dev` too: sudo apt install graphviz-dev

pip install canals[draw]
```
