from abc import ABC, abstractmethod
from typing import List, Optional, Dict

from haystack import Document


class BaseSummarizer(ABC):
    """
    Abstract class for Summarizer
    """

    outgoing_edges = 1

    @abstractmethod
    def predict(self, documents: List[Document], generate_single_summary: bool = False, query: str = None) -> Dict:
        """
        Abstract method for creating a summary.

        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
        :param generate_single_summary: Whether to generate a single summary for all documents or one summary per document.
                                        If set to "True", all docs will be joined to a single string that will then
                                        be summarized.
                                        Important: The summary will depend on the order of the supplied documents!
        :param query: Query
        :return: Generated answers plus additional infos in a dict like this:

        ```python
        |     {'query': 'Where is Eiffel Tower?',
        |      'answers':
        |          [{'query': 'Where is Eiffel Tower?',
        |            'answer': 'The Eiffel Tower is a landmark in Paris, France.',
        |            'meta': {
        |                      'text': 'The tower is 324 metres ...'
        |      }}]}
        ```
        """
        pass

    def run(self, documents: List[Document], generate_single_summary: bool = False, query: str = None, **kwargs):

        if documents:
            results = self.predict(query=query, documents=documents, generate_single_summary=generate_single_summary)
        else:
            results = {"answers": []}

        results.update(**kwargs)
        return results, "output_1"
