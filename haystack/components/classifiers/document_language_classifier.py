# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional

from haystack import Document, component, logging
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install langdetect'") as langdetect_import:
    import langdetect


@component
class DocumentLanguageClassifier:
    """
    Classify the language of documents and add the detected language to their metadata.

    A `MetadataRouter` can then route them onto different output connections depending on their language.
    The set of supported languages can be specified.
    For routing plain text using the same logic, use the related `TextLanguageRouter` component instead.

    Usage example within an indexing pipeline, storing in a Document Store
    only documents written in English:

    ```python
    from haystack import Document, Pipeline
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.classifiers import DocumentLanguageClassifier
    from haystack.components.routers import MetadataRouter
    from haystack.components.writers import DocumentWriter

    docs = [Document(id="1", content="This is an English document"),
            Document(id="2", content="Este es un documento en español")]

    document_store = InMemoryDocumentStore()

    p = Pipeline()
    p.add_component(instance=DocumentLanguageClassifier(languages=["en"]), name="language_classifier")
    p.add_component(instance=MetadataRouter(rules={"en": {"language": {"$eq": "en"}}}), name="router")
    p.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    p.connect("language_classifier.documents", "router.documents")
    p.connect("router.en", "writer.documents")

    p.run({"language_classifier": {"documents": docs}})

    written_docs = document_store.filter_documents()
    assert len(written_docs) == 1
    assert written_docs[0] == Document(id="1", content="This is an English document", meta={"language": "en"})
    ```
    """

    def __init__(self, languages: Optional[List[str]] = None):
        """
        Initialize the DocumentLanguageClassifier.

        :param languages: A list of languages in ISO code, each corresponding to a different output connection.
            For supported languages, see the [`langdetect` documentation](https://github.com/Mimino666/langdetect#languages).
            If not specified, the default is ["en"].
        """
        langdetect_import.check()
        if not languages:
            languages = ["en"]
        self.languages = languages

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        This method classifies the documents' language and adds it to their metadata.

        If a Document's text does not match any of the languages specified at initialization,
        the metadata value "unmatched" will be stored.

        :param documents: A list of documents to classify their language.

        :returns: A dictionary with the following key:
            - `documents`: List of Documents with an added metadata field called `language`.

        :raises TypeError: if the input is not a list of Documents.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "DocumentLanguageClassifier expects a list of Document as input. "
                "In case you want to classify a text, please use the TextLanguageClassifier."
            )

        output: Dict[str, List[Document]] = {language: [] for language in self.languages}
        output["unmatched"] = []

        for document in documents:
            detected_language = self._detect_language(document)
            if detected_language in self.languages:
                document.meta["language"] = detected_language
            else:
                document.meta["language"] = "unmatched"

        return {"documents": documents}

    def _detect_language(self, document: Document) -> Optional[str]:
        try:
            language = langdetect.detect(document.content)
        except langdetect.LangDetectException:
            logger.warning(
                "Langdetect cannot detect the language of Document with id: {document_id}", document_id=document.id
            )
            language = None
        return language
