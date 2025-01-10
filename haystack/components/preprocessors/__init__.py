# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from haystack.lazy_imports import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from haystack.components.preprocessors.document_cleaner import DocumentCleaner
    from haystack.components.preprocessors.document_splitter import DocumentSplitter
    from haystack.components.preprocessors.nltk_document_splitter import NLTKDocumentSplitter
    from haystack.components.preprocessors.sentence_tokenizer import SentenceSplitter
    from haystack.components.preprocessors.text_cleaner import TextCleaner


_lazy_imports = {
    "DocumentCleaner": "haystack.components.preprocessors.document_cleaner",
    "DocumentSplitter": "haystack.components.preprocessors.document_splitter",
    "NLTKDocumentSplitter": "haystack.components.preprocessors.nltk_document_splitter",
    "SentenceSplitter": "haystack.components.preprocessors.sentence_tokenizer",
    "TextCleaner": "haystack.components.preprocessors.text_cleaner",
}

__all__ = list(_lazy_imports.keys())


def __getattr__(name):
    return lazy_getattr(name, _lazy_imports, __name__)


def __dir__():
    return lazy_dir(_lazy_imports)
