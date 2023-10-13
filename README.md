# Canals

<p align="center" float="left">
  <img alt="" src="https://raw.githubusercontent.com/deepset-ai/canals/main/images/canals-logo-light.png#gh-dark-mode-only"/>
  <img alt="" src="https://raw.githubusercontent.com/deepset-ai/canals/main/images/canals-logo-dark.png#gh-light-mode-only"/>
</p>

[![PyPI - Version](https://img.shields.io/pypi/v/canals.svg)](https://pypi.org/project/canals)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/canals.svg)](https://pypi.org/project/canals)
<a href="https://github.com/deepset-ai/canals/actions/workflows/tests.yml">
    <img alt="Tests" src="https://github.com/deepset-ai/canals/workflows/Tests/badge.svg?branch=main">
</a>
[![Coverage Status](https://coveralls.io/repos/github/deepset-ai/canals/badge.svg?branch=main)](https://coveralls.io/github/deepset-ai/canals?branch=main)
<a href="https://deepset-ai.github.io/canals/">
    <img alt="Documentation" src="https://img.shields.io/website?label=documentation&up_message=online&url=https%3A%2F%2Fdeepset-ai.github.io/canals/">
</a>
<a href="https://github.com/deepset-ai/canals/commits/main">
    <img alt="Last commit" src="https://img.shields.io/github/last-commit/deepset-ai/canals">
</a>
<a href="https://pepy.tech/project/canals">
    <img alt="Monthly Downloads" src="https://pepy.tech/badge/canals/month">
</a>
<a href="https://github.com/deepset-ai/canals">
    <img alt="Stars" src="https://shields.io/github/stars/deepset-ai/canals?style=social">
</a>
<a href="https://ossinsight.io/analyze/deepset-ai/canals">
    <img alt="Stats" src="https://img.shields.io/badge/Stats-updated-blue">
</a>

Canals is a **component orchestration engine**. Components are Python objects that can execute a task, like reading a file, performing calculations, or making API calls. Canals connects these objects together: it builds a graph of components and takes care of managing their execution order, making sure that each object receives the input it expects from the other components of the pipeline.

Canals powers version 2.0 of the [Haystack framework](https://github.com/deepset-ai/haystack).

## Installation

To install Canals, run:

```console
pip install canals
```

To be able to draw pipelines (`Pipeline.draw()` method), please make sure you have either an internet connection (to reach the Mermaid graph renderer at `https://mermaid.ink`) or [graphviz](https://graphviz.org/download/) (version 2.49.0) installed. If you
plan to use Mermaid there is no additional steps to take, while for graphviz
you need to do:

### GraphViz
```console
sudo apt install graphviz  # You may need `graphviz-dev` too
pip install pygraphviz
```
