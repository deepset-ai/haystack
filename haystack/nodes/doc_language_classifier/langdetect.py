import logging
from typing import List, Optional, Dict, Any

from langdetect import LangDetectException, detect

from haystack.nodes.base import Document
from haystack.nodes.doc_language_classifier.base import BaseDocumentLanguageClassifier

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGES = ["en", "de", "es", "cs", "nl"]


class LangdetectDocumentLanguageClassifier(BaseDocumentLanguageClassifier):
    outgoing_edges = len(DEFAULT_LANGUAGES)

    @classmethod
    def _calculate_outgoing_edges(cls, component_params: Dict[str, Any]) -> int:
        route_by_language = component_params.get("route_by_language", True)
        if route_by_language is False:
            return 1
        languages_to_route = component_params.get("languages_to_route", DEFAULT_LANGUAGES)
        return len(languages_to_route)

    def __init__(
        self,
        add_language_to_meta: bool = True,
        route_by_language: bool = True,
        languages_to_route: Optional[List[str]] = None,
    ):
        """
        Node that sends out Documents on a different output edge depending on the language the document is written in.
        :param languages: languages that this node can distinguish (ISO code, see `langdetect` documentation).
        """
        super().__init__()

        if languages_to_route is None:
            languages_to_route = DEFAULT_LANGUAGES

        if len(set(languages_to_route)) != len(languages_to_route):
            duplicates = {lang for lang in languages_to_route if languages_to_route.count(lang) > 1}
            raise ValueError(f"languages_to_route parameter can't contain duplicate values ({duplicates}).")

        self.add_language_to_meta = add_language_to_meta
        self.route_by_language = route_by_language
        self.languages_to_route = languages_to_route

    def predict(self, documents: List[Document]) -> List[Document]:
        """
        Return the code of the language of the document.
        """
        if len(documents) == 0:
            raise AttributeError("DocumentLanguageClassifier needs at least one document to predict the language.")

        documents_with_language = []
        for document in documents:
            try:
                language = detect(document.content)
            except LangDetectException:
                logger.warning("Langdetect cannot detect the language of document: %s", document)
                language = None
            document.meta["language"] = language
            documents_with_language.append(document)
        return documents_with_language

    def predict_batch(self, documents: List[List[Document]], batch_size: Optional[int] = None) -> List[List[Document]]:
        """
        Produce the summarization from the supplied documents.
        These documents can for example be retrieved via the Retriever.

        :param documents: Single list of related documents or list of lists of related documents
                          (e.g. coming from a retriever) that the answer shall be conditioned on.
        :param batch_size: Number of Documents to process at a time.
        """

        if len(documents) == 0 or all(len(docs_list) == 0 for docs_list in documents):
            raise AttributeError("DocumentLanguageClassifier needs at least one document to predict the language.")

        # TODO:  if batch_size is None:
        #     batch_size = self.batch_size

        return [self.predict(documents=docs_list) for docs_list in documents]
