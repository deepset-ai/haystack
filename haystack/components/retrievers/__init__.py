# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "auto_merging_retriever": ["AutoMergingRetriever"],
    "filter_retriever": ["FilterRetriever"],
    "in_memory": ["InMemoryBM25Retriever", "InMemoryEmbeddingRetriever"],
    "sentence_window_retriever": ["SentenceWindowRetriever"],
}

if TYPE_CHECKING:
    from .auto_merging_retriever import AutoMergingRetriever
    from .filter_retriever import FilterRetriever
    from .in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
    from .sentence_window_retriever import SentenceWindowRetriever

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
