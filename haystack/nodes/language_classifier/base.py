from abc import abstractmethod
from haystack.nodes.base import BaseComponent, Document
from typing import Optional, List, Dict, Any, Union
import logging
from langdetect import detect, LangDetectException


logger = logging.getLogger(__name__)

DEFAULT_LANGUAGES = ["en", "de", "es", "cs", "nl"]


class BaseDocumentLanguageClassifier(BaseComponent):
    outgoing_edges = len(DEFAULT_LANGUAGES)

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
            raise ValueError("languages_to_route parameter can't contain duplicate values.")

        self.add_language_to_meta = add_language_to_meta
        self.route_by_language = route_by_language
        self.languages_to_route = languages_to_route

    @classmethod
    def _calculate_outgoing_edges(cls, component_params: Dict[str, Any]) -> int:
        route_by_language = component_params["route_by_language"]
        if route_by_language is False:
            return 1
        languages_to_route = component_params.get("languages_to_route", DEFAULT_LANGUAGES)
        return len(languages_to_route)

    def detect_language(self, document: Document) -> Optional[str]:
        """
        Return the code of the language of the document.
        """
        try:
            return detect(document.content)
        except LangDetectException:
            logger.warning("Langdetect cannot detect the language of document: %s", document)
            return None

    # def _get_language(self, documents: List[Document]) -> bool:
    #     """
    #     Checks whether all the documents passed are written in the same language and returns its code.
    #     """
    #     languages = {self.detect_language(document) for document in documents}
    #     if self.route_by_language is True:
    #     if len(languages) > 1:
    #         raise ValueError(
    #             f"Documents of multiple languages ({languages}) are not allowed together. "
    #             "Please call Pipeline.run() once for each file, or consider using Pipeline.run_batch()."
    #         )
    #     if not languages[0]:
    #         logging.warning(
    #             f"Langdetect could not understand the language of any of the documents. "
    #             f"Defaulting to the first language in the supported languages list: {self.languages[0]}"
    #         )
    #         return self.languages[0]
    #     return languages[0]

    def run(self, documents: List[Document]):  # type: ignore
        """
        Sends out documents on a different output edge depending on their language.
        :param documents: list of documents to classify.
        """
        languages = [self.detect_language(document) for document in documents]

        if self.add_language_to_meta is True:
            for doc, lang in zip(documents, languages):
                doc.meta["language"] = lang

        output = {"documents": documents}

        if self.route_by_language is False:
            return output, "output_1"

        # route_by_language is True
        unique_languages = list(set(languages))
        if len(unique_languages) > 1:
            raise ValueError(
                f"If route_by_language parameter is True, Documents of multiple languages ({unique_languages}) are not allowed together. "
                "If you want to route documents by language, you can call Pipeline.run() once for each file, or consider using Pipeline.run_batch()."
            )
        if unique_languages[0] is None:
            logging.warning(
                "The model cannot detect the language of any of the documents."
                "The first language in the list of supported languages ​​will be used to route the document: %s",
                self.languages_to_route[0],
            )
            language: Optional[str] = self.languages_to_route[0]
        language = unique_languages[0]
        if language not in self.languages_to_route:
            raise ValueError(
                f"'{language}' is not in the list of languages to route ({', '.join(self.languages_to_route)})."
                f"You should specify them when initializing the node, using the parameter languages_to_route."
            )

        return output, f"output_{self.languages_to_route.index(language) + 1}"

    def run_batch(self, documents: Union[List[Document], List[List[Document]]]):  # type: ignore
        """
        Sends out documents on a different output edge depending on their language, in batches.
        :param documents: list of lists of documents to classify.
        """
        if isinstance(documents[0], Document):
            flat_doc_list: List[Document] = [doc for doc in documents if isinstance(doc, Document)]
            return self.run(documents=flat_doc_list)

        split: Dict = {}
        if self.route_by_language is True:
            split = {f"output_{pos}": [] for pos in range(len(self.languages_to_route))}

        nested_doc_list: List[List[Document]] = [lst for lst in documents if isinstance(lst, list)]
        for documents_list in nested_doc_list:
            output, edge_name = self.run(documents=documents_list)
            split[edge_name]["documents"].append(output)
        return split, "split"
