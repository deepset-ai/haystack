# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional

from haystack import component, logging
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install langdetect'") as langdetect_import:
    import langdetect


@component
class TextLanguageRouter:
    """
    Routes a text input onto one of different output connections depending on its language.

    The set of supported languages can be specified.
    For routing Documents based on their language use the `DocumentLanguageClassifier` component to first
    classify the documents and then the `MetaDataRouter` to route them.

    Usage example in a retrieval pipeline that passes only English language queries to the retriever:

    ```python
    from haystack import Pipeline, Document
    from haystack.components.routers import TextLanguageRouter
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

    document_store = InMemoryDocumentStore()
    document_store.write_documents([Document(content="Elvis Presley was an American singer and actor.")])

    p = Pipeline()
    p.add_component(instance=TextLanguageRouter(languages=["en"]), name="text_language_router")
    p.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="retriever")
    p.connect("text_language_router.en", "retriever.query")

    result = p.run({"text_language_router": {"text": "Who was Elvis Presley?"}})
    assert result["retriever"]["documents"][0].content == "Elvis Presley was an American singer and actor."

    result = p.run({"text_language_router": {"text": "ένα ελληνικό κείμενο"}})
    assert result["text_language_router"]["unmatched"] == "ένα ελληνικό κείμενο"
    ```
    """

    def __init__(self, languages: Optional[List[str]] = None):
        """
        Initialize the TextLanguageRouter component.

        :param languages: A list of languages in ISO code, each corresponding to a different output connection.
            For supported languages, see the [`langdetect` documentation](https://github.com/Mimino666/langdetect#languages).
            If not specified, the default is `["en"]`.
        """
        langdetect_import.check()
        if not languages:
            languages = ["en"]
        self.languages = languages
        component.set_output_types(self, unmatched=str, **{language: str for language in languages})

    def run(self, text: str) -> Dict[str, str]:
        """
        Route the text to one of different output connections based on its language.

        If the text does not match any of the languages specified at initialization, it is routed to
        a connection named "unmatched".

        :param text: A string to route to different edges based on its language.

        :returns: A dictionary of length one in which the key is the language (or `"unmatched"`)
            and the value is the text.

        :raises TypeError: If the input is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(
                "TextLanguageRouter expects a str as input. In case you want to classify a document, please use the DocumentLanguageClassifier and MetaDataRouter."
            )

        output: Dict[str, str] = {}

        detected_language = self._detect_language(text)
        if detected_language in self.languages:
            output[detected_language] = text
        else:
            output["unmatched"] = text

        return output

    def _detect_language(self, text: str) -> Optional[str]:
        try:
            language = langdetect.detect(text)
        except langdetect.LangDetectException as exception:
            logger.warning("Langdetect cannot detect the language of text. Error: {error}", error=exception)
            # Only log the text in debug mode, as it might contain sensitive information
            logger.debug("Langdetect cannot detect the language of text: {text}", text=text)
            language = None
        return language
