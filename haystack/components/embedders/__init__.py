# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from haystack.lazy_imports import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from haystack.components.embedders.azure_document_embedder import AzureOpenAIDocumentEmbedder
    from haystack.components.embedders.azure_text_embedder import AzureOpenAITextEmbedder
    from haystack.components.embedders.hugging_face_api_document_embedder import HuggingFaceAPIDocumentEmbedder
    from haystack.components.embedders.hugging_face_api_text_embedder import HuggingFaceAPITextEmbedder
    from haystack.components.embedders.openai_document_embedder import OpenAIDocumentEmbedder
    from haystack.components.embedders.openai_text_embedder import OpenAITextEmbedder
    from haystack.components.embedders.sentence_transformers_document_embedder import (
        SentenceTransformersDocumentEmbedder,
    )
    from haystack.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder


_lazy_imports = {
    "AzureOpenAIDocumentEmbedder": "haystack.components.embedders.azure_document_embedder",
    "AzureOpenAITextEmbedder": "haystack.components.embedders.azure_text_embedder",
    "HuggingFaceAPIDocumentEmbedder": "haystack.components.embedders.hugging_face_api_document_embedder",
    "HuggingFaceAPITextEmbedder": "haystack.components.embedders.hugging_face_api_text_embedder",
    "OpenAIDocumentEmbedder": "haystack.components.embedders.openai_document_embedder",
    "OpenAITextEmbedder": "haystack.components.embedders.openai_text_embedder",
    "SentenceTransformersDocumentEmbedder": "haystack.components.embedders.sentence_transformers_document_embedder",
    "SentenceTransformersTextEmbedder": "haystack.components.embedders.sentence_transformers_text_embedder",
}

__all__ = list(_lazy_imports.keys())


def __getattr__(name):
    return lazy_getattr(name, _lazy_imports, __name__)


def __dir__():
    return lazy_dir(_lazy_imports)
