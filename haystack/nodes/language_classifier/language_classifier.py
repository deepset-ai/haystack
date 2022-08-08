from haystack.nodes.base import BaseComponent, Document
from typing import *
import logging

try:
    from langdetect import detect
except ImportError as ie:
    logging.debug(
        "Failed to import 'detect' (from 'langdetect')."
    )

DEFAULT_LANGUAGES = ['en', 'de', 'es', 'cs', 'nl']


class LanguageClassifier(BaseComponent):

    outgoing_edges = len(DEFAULT_LANGUAGES)

    def __init__(self, custom_languages: List[str] = DEFAULT_LANGUAGES):
        """
        Node that sends out files on a different output edge depending on their language.
        :param custom_langusges: languages that this node can distinguish.
            Note that you can only use languages, which you specified in custom_language variable.
            Number of edges will be the same as number of specified languages in custom_language variable.
        """

        if len(set(custom_languages)) != len(custom_languages):
            duplicates = custom_languages
            for item in set(custom_languages):
                duplicates.remove(item)
            raise ValueError(f"supported_types can't contain duplicate values ({duplicates}).")

        super().__init__()

        self.custom_languages = custom_languages

    def _get_language(self, document: Document) -> str:
        """
        Return a string value of detected language from text of document variable.
        """

        try:
            language = detect(document.content)
            return language

        except LangDetectException:
            raise ValueError(f"Not enough text in document for proper detection of language.")

    def _check_single_language(self, documents: List[Document]) -> bool:
        """
         Return boolean variable True if all languages from documents variable are the same.
        """
        language = [self._get_language(document) for document in documents]
        if len(set(language)) == 1:
            return True
        else:
            raise ValueError('You are using documents with multiple languages!')

    def run(self, documents: List[Document]):
        """
        Sends out files on a different output edge depending on their language.
        :param documents: list of documents to route on different edges.
        """

        if not isinstance(documents, list):
            documents = [documents]

        if self._check_single_language(documents):
            output = {"documents": documents}

        try:
            language = self.custom_languages.index(self._get_language(documents[0]))
        except ValueError:
            raise Valuerror("You are not using document with language from custom_languages")
        return output, f"output_{language + 1}"

    def run_batch(self, documents: List[Document]):
        self.run(documents=documents)