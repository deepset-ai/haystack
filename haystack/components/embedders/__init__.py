# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "azure_document_embedder": ["AzureOpenAIDocumentEmbedder"],
    "azure_text_embedder": ["AzureOpenAITextEmbedder"],
    "mock_document_embedder": ["MockDocumentEmbedder"],
    "mock_text_embedder": ["MockTextEmbedder"],
    "openai_document_embedder": ["OpenAIDocumentEmbedder"],
    "openai_text_embedder": ["OpenAITextEmbedder"],
}

if TYPE_CHECKING:
    from .azure_document_embedder import AzureOpenAIDocumentEmbedder as AzureOpenAIDocumentEmbedder
    from .azure_text_embedder import AzureOpenAITextEmbedder as AzureOpenAITextEmbedder
    from .mock_document_embedder import MockDocumentEmbedder as MockDocumentEmbedder
    from .mock_text_embedder import MockTextEmbedder as MockTextEmbedder
    from .openai_document_embedder import OpenAIDocumentEmbedder as OpenAIDocumentEmbedder
    from .openai_text_embedder import OpenAITextEmbedder as OpenAITextEmbedder

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
