from abc import ABC, abstractmethod
from typing import List
import logging

from haystack import Document
from haystack.document_store.base import BaseDocumentStore

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    document_store: BaseDocumentStore

    @abstractmethod
    def retrieve(self, query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        pass

    def eval(
        self,
        label_index: str = "label",
        doc_index: str = "eval_document",
        label_origin: str = "gold_label",
        top_k: int = 10,
        open_domain: bool = False
    ) -> dict:
        """
        Performs evaluation on the Retriever.
        Retriever is evaluated based on whether it finds the correct document given the question string and at which
        position in the ranking of documents the correct document is.

        |  Returns a dict containing the following metrics:

            - "recall": Proportion of questions for which correct document is among retrieved documents
            - "mean avg precision": Mean of average precision for each question. Rewards retrievers that give relevant
              documents a higher rank.

        :param label_index: Index/Table in DocumentStore where labeled questions are stored
        :param doc_index: Index/Table in DocumentStore where documents that are used for evaluation are stored
        :param top_k: How many documents to return per question
        :param open_domain: If ``True``, retrieval will be evaluated by checking if the answer string to a question is
                            contained in the retrieved docs (common approach in open-domain QA).
                            If ``False``, retrieval uses a stricter evaluation that checks if the retrieved document ids
                            are within ids explicitly stated in the labels.
        """

        # Extract all questions for evaluation
        filters = {"origin": [label_origin]}

        labels = self.document_store.get_all_labels_aggregated(index=label_index, filters=filters)

        correct_retrievals = 0
        summed_avg_precision = 0

        # Collect questions and corresponding answers/document_ids in a dict
        question_label_dict = {}
        for label in labels:
            if open_domain:
                question_label_dict[label.question] = label.multiple_answers
            else:
                deduplicated_doc_ids = list(set([str(x) for x in label.multiple_document_ids]))
                question_label_dict[label.question] = deduplicated_doc_ids

        # Option 1: Open-domain evaluation by checking if the answer string is in the retrieved docs
        if open_domain:
            for question, gold_answers in question_label_dict.items():
                retrieved_docs = self.retrieve(question, top_k=top_k, index=doc_index)
                # check if correct doc in retrieved docs
                for doc_idx, doc in enumerate(retrieved_docs):
                    for gold_answer in gold_answers:
                        if gold_answer in doc.text:
                            correct_retrievals += 1
                            summed_avg_precision += 1 / (doc_idx + 1)  # type: ignore
                            break
        # Option 2: Strict evaluation by document ids that are listed in the labels
        else:
            for question, gold_ids in question_label_dict.items():
                retrieved_docs = self.retrieve(question, top_k=top_k, index=doc_index)
                # check if correct doc in retrieved docs
                for doc_idx, doc in enumerate(retrieved_docs):
                    for gold_id in gold_ids:
                        if str(doc.id) == gold_id:
                            correct_retrievals += 1
                            summed_avg_precision += 1 / (doc_idx + 1)  # type: ignore
                            break
        # Metrics
        number_of_questions = len(question_label_dict)
        recall = correct_retrievals / number_of_questions
        mean_avg_precision = summed_avg_precision / number_of_questions

        logger.info((f"For {correct_retrievals} out of {number_of_questions} questions ({recall:.2%}), the answer was in"
                     f" the top-{top_k} candidate passages selected by the retriever."))

        return {"recall": recall, "map": mean_avg_precision}