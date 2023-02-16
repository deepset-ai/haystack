import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

from haystack.nodes.base import BaseComponent, Document


logger = logging.getLogger(__name__)


class BaseDocumentLanguageClassifier(BaseComponent):
    outgoing_edges = 1
    route_by_language = False
    languages_to_route: List[str] = []

    @abstractmethod
    def predict(self, documents: List[Document]) -> List[Document]:
        pass

    @abstractmethod
    def predict_batch(self, documents: List[List[Document]], batch_size: Optional[int] = None) -> List[List[Document]]:
        pass

    def _get_edge_from_language(self, language: str) -> str:
        return f"output_{self.languages_to_route.index(language) + 1}"

    def run(self, documents: List[Document]) -> Tuple[Dict[str, List[Document]], str]:  # type: ignore
        """
        Sends out documents on a different output edge depending on their language.
        :param documents: list of documents to classify.
        """
        docs_with_languages = self.predict(documents=documents)

        if self.route_by_language is False:
            output = {"documents": docs_with_languages}
            return output, "output_1"

        # route_by_language is True
        languages = [doc.meta["language"] for doc in docs_with_languages]
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
        return output, self._get_edge_from_language(str(language))

    def run_batch(self, documents: List[List[Document]]) -> Tuple[Dict, str]:  # type: ignore
        """
        Sends out documents on a different output edge depending on their language, in batches.
        :param documents: list of lists of documents to classify.
        """
        docs_lists_with_languages = self.predict_batch(documents=documents)

        if self.route_by_language is False:
            output = {"documents": docs_lists_with_languages}
            return output, "output_1"

        # route_by_language is True
        split: Dict[str, Dict[str, List[List[Document]]]] = {
            f"output_{pos}": {"documents": []} for pos in range(len(self.languages_to_route))
        }

        for docs_list in docs_lists_with_languages:
            languages = [doc.meta["language"] for doc in docs_list]
            unique_languages = list(set(languages))
            if len(unique_languages) > 1:
                raise ValueError(
                    f"If route_by_language parameter is True, Documents of multiple languages ({unique_languages}) are not allowed together. "
                    "If you want to route documents by language, you can call Pipeline.run() once for each file."
                )
            if unique_languages[0] is None:
                logging.warning(
                    "The model cannot detect the language of some of the documents."
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

            edge_name = self._get_edge_from_language(str(language))
            split[edge_name]["documents"].append(docs_list)
        return split, "split"
