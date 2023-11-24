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
    Classify the language of documents and add the detected language to their metadata.
    A MetadataRouter can then route them onto different output connections depending on their language.
    This is useful to route documents to different models in a pipeline depending on their language.
    The set of supported languages can be specified.
    For routing plain text using the same logic, use the related TextLanguageRouter component instead.

    Example usage within an indexing pipeline, storing in a Document Store
    only documents written in English:

    ```python
    document_store = InMemoryDocumentStore()
    p = Pipeline()
    p.add_component(instance=TextFileToDocument(), name="text_file_converter")
    p.add_component(instance=DocumentLanguageClassifier(), name="language_classifier")
    p.add_component(instance=MetadataRouter(rules={"en": {"language": {"$eq": "en"}}}), name="router")
    p.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    p.connect("text_file_converter.documents", "language_classifier.documents")
    p.connect("language_classifier.documents", "router.documents")
    p.connect("router.en", "writer.documents")
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

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Run the DocumentLanguageClassifier. This method classifies the documents' language and adds it to their metadata.
        If a Document's text does not match any of the languages specified at initialization, the metadata value "unmatched" will be stored.

        :param documents: A list of documents to classify their language.
        :return: List of Documents with an added metadata field called language.
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
                document.meta["language"] = detected_language
            else:
                document.meta["language"] = "unmatched"

        return {"documents": documents}

    def detect_language(self, document: Document) -> Optional[str]:
        try:
            language = langdetect.detect(document.content)
        except langdetect.LangDetectException:
            logger.warning("Langdetect cannot detect the language of Document with id: %s", document.id)
            language = None
        return language
