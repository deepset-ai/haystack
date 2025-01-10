# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from haystack.lazy_imports import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from haystack.components.retrievers.filter_retriever import FilterRetriever
    from haystack.components.retrievers.in_memory.bm25_retriever import InMemoryBM25Retriever
    from haystack.components.retrievers.in_memory.embedding_retriever import InMemoryEmbeddingRetriever
    from haystack.components.retrievers.sentence_window_retriever import SentenceWindowRetriever

_lazy_imports = {
    "FilterRetriever": "haystack.components.retrievers.filter_retriever",
    "InMemoryBM25Retriever": "haystack.components.retrievers.in_memory.bm25_retriever",
    "InMemoryEmbeddingRetriever": "haystack.components.retrievers.in_memory.embedding_retriever",
    "SentenceWindowRetriever": "haystack.components.retrievers.sentence_window_retriever",
}

__all__ = list(_lazy_imports.keys())


def __getattr__(name):
    return lazy_getattr(name, _lazy_imports, __name__)


def __dir__():
    return lazy_dir(_lazy_imports)
