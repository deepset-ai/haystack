from abc import abstractmethod
from typing import Any, List, Optional, Dict, Union

from tqdm.auto import tqdm

from haystack.errors import HaystackError
from haystack.schema import Answer, Document, MultiLabel
from haystack.nodes.base import BaseComponent


class BaseGenerator(BaseComponent):
    """
    Abstract class for Generators
    """

    outgoing_edges = 1

    def __init__(self, progress_bar: bool = True):
        super().__init__()
        self.progress_bar = progress_bar

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

    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None, labels: Optional[MultiLabel] = None, add_isolated_node_eval: bool = False):  # type: ignore

        if documents:
            results = self.predict(query=query, documents=documents, top_k=top_k)
        else:
            results = {"answers": []}

        # run evaluation with "perfect" labels as node inputs to calculate "upper bound" metrics for just this node
        if add_isolated_node_eval and labels is not None:
            relevant_documents = list({label.document.id: label.document for label in labels.labels}.values())
            results_label_input = self.predict(query=query, documents=relevant_documents, top_k=top_k)
            results["answers_isolated"] = results_label_input["answers"]

        return results, "output_1"

    def run_batch(  # type: ignore
        self,
        queries: List[str],
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
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Generate the answer to the input queries. The generation will be conditioned on the supplied documents.
        These documents can for example be retrieved via the Retriever.

        - If you provide a list containing a single query...

            - ... and a single list of Documents, the query will be applied to each Document individually.
            - ... and a list of lists of Documents, the query will be applied to each list of Documents and the Answers
              will be aggregated per Document list.

        - If you provide a list of multiple queries...

            - ... and a single list of Documents, each query will be applied to each Document individually.
            - ... and a list of lists of Documents, each query will be applied to its corresponding list of Documents
              and the Answers will be aggregated per query-Document pair.

        :param queries: List of queries.
        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
                          Can be a single list of Documents or a list of lists of Documents.
        :param top_k: Number of returned answers per query.
        :param batch_size: Not applicable.
        :return: Generated answers plus additional infos in a dict like this:

         ```python
         {'queries': 'who got the first nobel prize in physics',
          'answers':
              [{'query': 'who got the first nobel prize in physics',
                'answer': ' albert einstein',
                'meta': { 'doc_ids': [...],
                          'doc_scores': [80.42758 ...],
                          'doc_probabilities': [40.71379089355469, ...
                          'content': ['Albert Einstein was a ...]
                          'titles': ['"Albert Einstein"', ...]
          }}]}
         ```
        """
        # TODO: This method currently just calls the predict method multiple times, so there is room for improvement.

        results: Dict = {"queries": queries, "answers": []}

        single_doc_list = False
        # Docs case 1: single list of Documents -> apply each query to all Documents
        if len(documents) > 0 and isinstance(documents[0], Document):
            single_doc_list = True
            pb = tqdm(total=len(queries) * len(documents), disable=not self.progress_bar, desc="Generating answers")
            for query in queries:
                for doc in documents:
                    if not isinstance(doc, Document):
                        raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
                    preds = self.predict(query=query, documents=[doc], top_k=top_k)
                    results["answers"].append(preds["answers"])
                    pb.update(1)
            pb.close()

        # Docs case 2: list of lists of Documents -> apply each query to corresponding list of Documents, if queries
        # contains only one query, apply it to each list of Documents
        elif len(documents) > 0 and isinstance(documents[0], list):
            if len(queries) == 1:
                queries = queries * len(documents)
            if len(queries) != len(documents):
                raise HaystackError("Number of queries must be equal to number of provided Document lists.")
            pb = tqdm(total=min(len(queries), len(documents)), disable=not self.progress_bar, desc="Generating answers")
            for query, cur_docs in zip(queries, documents):
                if not isinstance(cur_docs, list):
                    raise HaystackError(f"cur_docs was of type {type(cur_docs)}, but expected a list of Documents.")
                preds = self.predict(query=query, documents=cur_docs, top_k=top_k)
                results["answers"].append(preds["answers"])
                pb.update(1)
            pb.close()

        # Group answers by question in case of multiple queries and single doc list
        if single_doc_list and len(queries) > 1:
            answers_per_query = int(len(results["answers"]) / len(queries))
            answers = []
            for i in range(0, len(results["answers"]), answers_per_query):
                answer_group = results["answers"][i : i + answers_per_query]
                answers.append(answer_group)
            results["answers"] = answers

        return results
