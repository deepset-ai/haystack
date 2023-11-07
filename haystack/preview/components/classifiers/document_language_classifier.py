import logging
from typing import List, Dict, Optional

from haystack.preview import component, Document
from haystack.preview.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install langdetect'") as langdetect_import:
    import langdetect


@component
class DocumentLanguageClassifier:
    """
    Routes documents onto different output connections depending on their language.
    This is useful for routing documents to different models in a pipeline depending on their language.
    The set of supported languages can be specified.
    For routing plain text using the same logic, use the related TextLanguageClassifier component instead.

    Example usage within an indexing pipeline, storing in a Document Store
    only documents written in English:

    ```python
    document_store = InMemoryDocumentStore()
    p = Pipeline()
    p.add_component(instance=TextFileToDocument(), name="text_file_converter")
    p.add_component(instance=DocumentLanguageClassifier(), name="language_classifier")
    p.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    p.connect("text_file_converter.documents", "language_classifier.documents")
    p.connect("language_classifier.en", "writer.documents")
    ```
    """

    def __init__(self, languages: Optional[List[str]] = None):
        """
        :param languages: A list of languages in ISO code, each corresponding to a different output connection
            (see [langdetect` documentation](https://github.com/Mimino666/langdetect#languages)).
            By default, only ["en"] is supported and Documents of any other language are routed to "unmatched".
        """
        langdetect_import.check()
        if not languages:
            languages = ["en"]
        self.languages = languages
        component.set_output_types(
            self, unmatched=List[Document], **{language: List[Document] for language in languages}
        )

    def run(self, documents: List[Document]):
        """
        Run the DocumentLanguageClassifier. This method routes the documents to different edges based on their language.
        If a Document's text does not match any of the languages specified at initialization, it is routed to
        a connection named "unmatched".

        :param documents: A list of documents to route to different edges.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "DocumentLanguageClassifier expects a list of Document as input. "
                "In case you want to classify a text, please use the TextLanguageClassifier."
            )

        output: Dict[str, List[Document]] = {language: [] for language in self.languages}
        output["unmatched"] = []

        for document in documents:
            detected_language = self.detect_language(document)
            if detected_language in self.languages:
                output[detected_language].append(document)
            else:
                output["unmatched"].append(document)

        return output

    def detect_language(self, document: Document) -> Optional[str]:
        try:
            language = langdetect.detect(document.content)
        except langdetect.LangDetectException:
            logger.warning("Langdetect cannot detect the language of Document with id: %s", document.id)
            language = None
        return language
