# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "azure_document_embedder": ["AzureOpenAIDocumentEmbedder"],
    "azure_text_embedder": ["AzureOpenAITextEmbedder"],
    "hugging_face_api_document_embedder": ["HuggingFaceAPIDocumentEmbedder"],
    "hugging_face_api_text_embedder": ["HuggingFaceAPITextEmbedder"],
    "openai_document_embedder": ["OpenAIDocumentEmbedder"],
    "openai_text_embedder": ["OpenAITextEmbedder"],
    "sentence_transformers_document_embedder": ["SentenceTransformersDocumentEmbedder"],
    "sentence_transformers_text_embedder": ["SentenceTransformersTextEmbedder"],
}

if TYPE_CHECKING:
    from .azure_document_embedder import AzureOpenAIDocumentEmbedder
    from .azure_text_embedder import AzureOpenAITextEmbedder
    from .hugging_face_api_document_embedder import HuggingFaceAPIDocumentEmbedder
    from .hugging_face_api_text_embedder import HuggingFaceAPITextEmbedder
    from .openai_document_embedder import OpenAIDocumentEmbedder
    from .openai_text_embedder import OpenAITextEmbedder
    from .sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
    from .sentence_transformers_text_embedder import SentenceTransformersTextEmbedder

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
