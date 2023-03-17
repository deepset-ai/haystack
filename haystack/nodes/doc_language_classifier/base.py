import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Any

from haystack.nodes.base import BaseComponent, Document


logger = logging.getLogger(__name__)

DEFAULT_LANGUAGES = ["en", "de", "es", "cs", "nl"]


class BaseDocumentLanguageClassifier(BaseComponent):
    """
    Abstract class for Document Language Classifiers.
    """

    outgoing_edges = len(DEFAULT_LANGUAGES)

    @classmethod
    def _calculate_outgoing_edges(cls, component_params: Dict[str, Any]) -> int:
        route_by_language = component_params.get("route_by_language", True)
        if route_by_language is False:
            return 1
        languages_to_route = component_params.get("languages_to_route", DEFAULT_LANGUAGES)
        return len(languages_to_route)

    def __init__(self, route_by_language: bool = True, languages_to_route: Optional[List[str]] = None):
        """
        :param route_by_language: Routes Documents to a different output edge depending on their language.
        :param languages_to_route: A list of languages in ISO code, each corresponding to a different output edge (see [langdetect documentation](https://github.com/Mimino666/langdetect#languages)).
        """
        super().__init__()

        if languages_to_route is None:
            languages_to_route = DEFAULT_LANGUAGES
            if route_by_language is True:
                logger.info(
                    "The languages_to_route list is not defined. The default list will be used: %s", languages_to_route
                )

        if len(set(languages_to_route)) != len(languages_to_route):
            duplicates = {lang for lang in languages_to_route if languages_to_route.count(lang) > 1}
            raise ValueError(f"The languages_to_route parameter can't contain duplicate values ({duplicates}).")

        self.route_by_language = route_by_language
        self.languages_to_route = languages_to_route

    @abstractmethod
    def predict(self, documents: List[Document], batch_size: Optional[int] = None) -> List[Document]:
        pass

    @abstractmethod
    def predict_batch(self, documents: List[List[Document]], batch_size: Optional[int] = None) -> List[List[Document]]:
        pass

    def _get_edge_from_language(self, language: str) -> str:
        return f"output_{self.languages_to_route.index(language) + 1}"

    def run(self, documents: List[Document]) -> Tuple[Dict[str, List[Document]], str]:  # type: ignore
        """
        Run language document classifier on a list of documents.

        :param documents: A list of documents whose language you want to detect.
        """
        docs_with_languages = self.predict(documents=documents)
        output = {"documents": docs_with_languages}

        if self.route_by_language is False:
            return output, "output_1"

        # self.route_by_language is True
        languages = [doc.meta["language"] for doc in docs_with_languages]
        unique_languages = list(set(languages))
        if len(unique_languages) > 1:
            raise ValueError(
                f"If the route_by_language parameter is True, Documents of multiple languages ({unique_languages}) are not allowed together. "
                "If you want to route documents by language, you can call Pipeline.run() once for each Document."
            )
        language = unique_languages[0]
        if language is None:
            logger.warning(
                "The model cannot detect the language of any of the documents."
                "The first language in the list of supported languages will be used to route the document: %s",
                self.languages_to_route[0],
            )
            language = self.languages_to_route[0]
        if language not in self.languages_to_route:
            raise ValueError(
                f"'{language}' is not in the list of languages to route ({', '.join(self.languages_to_route)})."
                f"You should specify them when initializing the node, using the parameter languages_to_route."
            )
        return output, self._get_edge_from_language(str(language))

    def run_batch(self, documents: List[List[Document]], batch_size: Optional[int] = None) -> Tuple[Dict, str]:  # type: ignore
        """
        Run language document classifier on batches of documents.

        :param documents: A list of lists of documents whose language you want to detect.
        """
        docs_lists_with_languages = self.predict_batch(documents=documents, batch_size=batch_size)

        if self.route_by_language is False:
            output = {"documents": docs_lists_with_languages}
            return output, "output_1"

        # self.route_by_language is True
        split: Dict[str, Dict[str, List[List[Document]]]] = {
            f"output_{pos}": {"documents": []} for pos in range(1, len(self.languages_to_route) + 1)
        }

        for docs_list in docs_lists_with_languages:
            languages = [doc.meta["language"] for doc in docs_list]
            unique_languages = list(set(languages))
            if len(unique_languages) > 1:
                raise ValueError(
                    f"If the route_by_language parameter is True, Documents of multiple languages ({unique_languages}) are not allowed together. "
                    "If you want to route documents by language, you can call Pipeline.run() once for each Document."
                )
            if unique_languages[0] is None:
                logger.warning(
                    "The model cannot detect the language of some of the documents."
                    "The first language in the list of supported languages will be used to route the documents: %s",
                    self.languages_to_route[0],
                )
                language: Optional[str] = self.languages_to_route[0]
            language = unique_languages[0]
            if language not in self.languages_to_route:
                raise ValueError(
                    f"'{language}' is not in the list of languages to route ({', '.join(self.languages_to_route)})."
                    f"Specify them when initializing the node, using the parameter languages_to_route."
                )

            edge_name = self._get_edge_from_language(str(language))
            split[edge_name]["documents"].append(docs_list)
        return split, "split"
