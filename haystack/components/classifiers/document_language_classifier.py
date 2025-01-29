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
    Classifies the language of each document and adds it to its metadata.

    Provide a list of languages during initialization. If the document's text doesn't match any of the
    specified languages, the metadata value is set to "unmatched".
    To route documents based on their language, use the MetadataRouter component after DocumentLanguageClassifier.
    For routing plain text, use the TextLanguageRouter component instead.

    ### Usage example

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
        Initializes the DocumentLanguageClassifier component.

        :param languages: A list of ISO language codes.
            See the supported languages in [`langdetect` documentation](https://github.com/Mimino666/langdetect#languages).
            If not specified, defaults to ["en"].
        """
        langdetect_import.check()
        if not languages:
            languages = ["en"]
        self.languages = languages

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Classifies the language of each document and adds it to its metadata.

        If the document's text doesn't match any of the languages specified at initialization,
        sets the metadata value to "unmatched".

        :param documents: A list of documents for language classification.

        :returns: A dictionary with the following key:
            - `documents`: A list of documents with an added `language` metadata field.

        :raises TypeError: if the input is not a list of Documents.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "DocumentLanguageClassifier expects a list of Document as input. "
                "In case you want to classify and route a text, please use the TextLanguageRouter."
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
