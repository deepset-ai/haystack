from abc import abstractmethod
from typing import Any, List, Optional, Dict, Union

from haystack.errors import HaystackError
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

    def run_batch(  # type: ignore
        self,
        queries: Union[str, List[str]],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        results = self.predict_batch(queries=queries, documents=documents, top_k=top_k, batch_size=batch_size)
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

    def predict_batch(
        self,
        queries: Union[str, List[str]],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        .
        """
        # TODO: This method currently just calls the predict method multiple times, so there is room for improvement.

        results: Dict = {"queries": queries, "answers": []}
        # Query case 1: single query
        if isinstance(queries, str):
            query = queries
            # Docs case 1: single list of Documents -> apply single query to all Documents
            if len(documents) > 0 and isinstance(documents[0], Document):
                for doc in documents:
                    if not isinstance(doc, Document):
                        raise HaystackError("Expected a Document.")
                    preds = self.predict(query=query, documents=[doc], top_k=top_k)
                    results["answers"].append(preds["answers"])

            # Docs case 2: list of lists of Documents -> apply single query to each list of Documents
            elif len(documents) > 0 and isinstance(documents[0], list):
                for docs in documents:
                    if not isinstance(docs, list):
                        raise HaystackError("Expected a list of Documents.")
                    preds = self.predict(query=query, documents=docs, top_k=top_k)
                    results["answers"].append(preds["answers"])

        # Query case 2: list of queries
        elif isinstance(queries, list) and len(queries) > 0 and isinstance(queries[0], str):
            # Docs case 1: single list of Documents -> apply each query to all Documents
            if len(documents) > 0 and isinstance(documents[0], Document):
                for query in queries:
                    for doc in documents:
                        if not isinstance(doc, Document):
                            raise HaystackError("Expected a Document.")
                        preds = self.predict(query=query, documents=[doc], top_k=top_k)
                        results["answers"].append(preds["answers"])

            # Docs case 2: list of lists of Documents -> apply each query to corresponding list of Documents
            elif len(documents) > 0 and isinstance(documents[0], list):
                if len(queries) != len(documents):
                    raise HaystackError("Number of queries must be equal to number of provided Document lists.")
                for query, cur_docs in zip(queries, documents):
                    if not isinstance(cur_docs, list):
                        raise HaystackError("Expected a list of Documents.")
                    preds = self.predict(query=query, documents=cur_docs, top_k=top_k)
                    results["answers"].append(preds["answers"])

        return results
