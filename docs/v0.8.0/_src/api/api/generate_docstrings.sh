#!/usr/bin/env bash

# Purpose : Automate the generation of docstrings

pydoc-markdown pydoc-markdown-document-store.yml
pydoc-markdown pydoc-markdown-file-converters.yml
pydoc-markdown pydoc-markdown-preprocessor.yml
pydoc-markdown pydoc-markdown-reader.yml
pydoc-markdown pydoc-markdown-generator.yml
pydoc-markdown pydoc-markdown-retriever.yml
pydoc-markdown pydoc-markdown-pipelines.yml