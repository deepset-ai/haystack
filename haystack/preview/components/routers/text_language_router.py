import logging
from typing import List, Dict, Optional

from haystack.preview import component
from haystack.preview.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install langdetect'") as langdetect_import:
    import langdetect


@component
class TextLanguageRouter:
    """
    Routes a text input onto one of different output connections depending on its language.
    This is useful for routing queries to different models in a pipeline depending on their language.
    The set of supported languages can be specified.
    For routing Documents based on their language use the related DocumentLanguageClassifier component to first
    classify the documents and then the MetaDataRouter to route them.

    Example usage in a retrieval pipeline that passes only English language queries to the retriever:

    ```python
    document_store = InMemoryDocumentStore()
    p = Pipeline()
    p.add_component(instance=TextLanguageRouter(), name="text_language_router")
    p.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="retriever")
    p.connect("text_language_router.en", "retriever.query")
    p.run({"text_language_router": {"text": "What's your query?"}})
    ```
    """

    def __init__(self, languages: Optional[List[str]] = None):
        """
        :param languages: A list of languages in ISO code, each corresponding to a different output connection (see [langdetect` documentation](https://github.com/Mimino666/langdetect#languages)). By default, only ["en"] is supported and texts of any other language are routed to "unmatched".
        """
        langdetect_import.check()
        if not languages:
            languages = ["en"]
        self.languages = languages
        component.set_output_types(self, unmatched=str, **{language: str for language in languages})

    def run(self, text: str) -> Dict[str, str]:
        """
        Run the TextLanguageRouter. This method routes the text one of different edges based on its language.
        If the text does not match any of the languages specified at initialization, it is routed to
        a connection named "unmatched".

        :param text: A str to route to one of different edges.
        """
        if not isinstance(text, str):
            raise TypeError(
                "TextLanguageRouter expects a str as input. In case you want to classify a document, please use the DocumentLanguageClassifier and MetaDataRouter."
            )

        output: Dict[str, str] = {}

        detected_language = self.detect_language(text)
        if detected_language in self.languages:
            output[detected_language] = text
        else:
            output["unmatched"] = text

        return output

    def detect_language(self, text: str) -> Optional[str]:
        try:
            language = langdetect.detect(text)
        except langdetect.LangDetectException:
            logger.warning("Langdetect cannot detect the language of text: %s", text)
            language = None
        return language
