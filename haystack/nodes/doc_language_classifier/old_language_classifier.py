from pydoc import doc
from haystack.nodes.base import BaseComponent, Document
from typing import *
import logging

try:
    from langdetect import detect, LangDetectException
except ImportError as ie:
    logging.debug("Failed to import 'langdetect'.")

DEFAULT_LANGUAGES = ["en", "de", "es", "cs", "nl"]


class DocumentLanguageClassifier(BaseComponent):
    outgoing_edges = len(DEFAULT_LANGUAGES)

    def __init__(self, languages: List[str] = DEFAULT_LANGUAGES):
        """
        Node that sends out Documents on a different output edge depending on the language the document is written in.
        :param languages: languages that this node can distinguish (ISO code, see `langdetect` documentation).
        """
        super().__init__()

        if len(set(languages)) != len(languages):
            duplicates = languages
            for item in set(languages):
                duplicates.remove(item)
            raise ValueError(f"languages can't contain duplicate values ({duplicates}).")

        self.languages = languages

    @classmethod
    def _calculate_outgoing_edges(cls, component_params: Dict[str, Any]) -> int:
        languages = component_params.get("languages", DEFAULT_LANGUAGES)
        return len(languages)

    def _detect_language(self, document: Document) -> str:
        """
        Return the code of the language of the document.
        """
        try:
            return detect(document.content)
        except LangDetectException:
            logging.warning(
                f"Langdetect could not understand the language of doc with id: {document.id}. "
                f"Content: '{document.content}'"
            )
            return None

    def _get_language(self, documents: List[Document]) -> bool:
        """
        Checks whether all the documents passed are written in the same language and returns its code.
        """
        languages = {self._detect_language(document) for document in documents}
        if len(languages) > 1:
            raise ValueError(
                f"Documents of multiple languages ({languages}) are not allowed together. "
                "Please call Pipeline.run() once for each file, or consider using Pipeline.run_batch()."
            )
        if not languages[0]:
            logging.warning(
                f"Langdetect could not understand the language of any of the documents. "
                f"Defaulting to the first language in the supported languages list: {self.languages[0]}"
            )
            return self.languages[0]
        return languages[0]

    def run(self, documents: List[Document]):
        """
        Sends out documents on a different output edge depending on their language.
        :param documents: list of documents to classify.
        """
        language = self._get_language(documents=documents)
        output = {"documents": documents}

        if language not in self.languages:
            raise ValueError(f"'{language}' is not in the list of supported languages ({', '.join(self.languages)}).")

        return output, f"output_{self.languages.index(language) + 1}"

    def run_batch(self, documents: Union[List[Document], List[List[Document]]]):
        """
        Sends out documents on a different output edge depending on their language, in batches.
        :param documents: list of lists of documents to classify.
        """
        if isinstance(documents[0], Document):
            return self.run(documents=documents)

        split = {f"output_{pos}": [] for pos in range(len(self.languages))}
        for documents_list in documents:
            output, edge_name = self.run(documents=documents_list)
            split[edge_name]["documents"].append(output)
        return split, "split"
