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
    Routes text strings to different output connections based on their language.

    Provide a list of languages during initialization. If the document's text doesn't match any of the
    specified languages, the metadata value is set to "unmatched".
    For routing documents based on their language, use the DocumentLanguageClassifier component,
    followed by the MetaDataRouter.

    ### Usage example

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

        :param languages: A list of ISO language codes.
            See the supported languages in [`langdetect` documentation](https://github.com/Mimino666/langdetect#languages).
            If not specified, defaults to ["en"].
        """
        langdetect_import.check()
        if not languages:
            languages = ["en"]
        self.languages = languages
        component.set_output_types(self, unmatched=str, **{language: str for language in languages})

    def run(self, text: str) -> Dict[str, str]:
        """
        Routes the text strings to different output connections based on their language.

        If the document's text doesn't match any of the specified languages, the metadata value is set to "unmatched".

        :param text: A text string to route.

        :returns: A dictionary in which the key is the language (or `"unmatched"`),
            and the value is the text.

        :raises TypeError: If the input is not a string.
        """
        if not isinstance(text, str):
            msg = (
                "TextLanguageRouter expects a string as input. In case you want to classify a document, please use "
                "the DocumentLanguageClassifier and MetaDataRouter."
            )
            raise TypeError(msg)

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
