from abc import abstractmethod
from typing import Any, List, Optional, Dict

from haystack.schema import Answer, Document
from haystack.nodes.base import BaseComponent


class BaseGenerator(BaseComponent):
    """
    Abstract class for Generators
    """

    outgoing_edges = 1

    @abstractmethod
    def predict(self, query: str, documents: List[Document], top_k: Optional[int]) -> Dict:
        """
        Abstract method to generate answers.

        :param query: Query
        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
        :param top_k: Number of returned answers
        :return: Generated answers plus additional infos in a dict
        """
        pass

    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None):  # type: ignore

        if documents:
            results = self.predict(query=query, documents=documents, top_k=top_k)
        else:
            results = {"answers": []}

        return results, "output_1"

    def _flatten_docs(self, documents: List[Document]):
        flat_docs_dict: Dict[str, Any] = {}
        for document in documents:
            for k, v in document.to_dict().items():
                if k not in flat_docs_dict:
                    flat_docs_dict[k] = []
                flat_docs_dict[k].append(v)
        return flat_docs_dict

    def _create_answers(self, generated_answers: List[str], documents: List[Document]) -> List[Answer]:
        flat_docs_dict = self._flatten_docs(documents)
        answers: List[Any] = []
        for generated_answer in generated_answers:
            answers.append(
                Answer(
                    answer=generated_answer,
                    type="generative",
                    meta={
                        "doc_ids": flat_docs_dict["id"],
                        "doc_scores": flat_docs_dict["score"],
                        "content": flat_docs_dict["content"],
                        "titles": [d.get("name", "") for d in flat_docs_dict["meta"]],
                    },
                )
            )
        return answers
