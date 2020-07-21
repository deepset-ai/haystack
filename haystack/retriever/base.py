from abc import ABC, abstractmethod
from typing import List, Type
import logging

from haystack.database.base import Document
from haystack.database.base import BaseDocumentStore

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    document_store: Type[BaseDocumentStore]

    @abstractmethod
    def retrieve(self, query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]:
        pass

    def eval(
        self,
        label_index: str = "feedback",
        doc_index: str = "eval_document",
        label_origin: str = "gold_label",
        top_k: int = 10,
    ) -> dict:
        """
        Performs evaluation on the Retriever.
        Retriever is evaluated based on whether it finds the correct document given the question string and at which
        position in the ranking of documents the correct document is.

        Returns a dict containing the following metrics:
            - "recall": Proportion of questions for which correct document is among retrieved documents
            - "mean avg precision": Mean of average precision for each question. Rewards retrievers that give relevant
              documents a higher rank.

        :param label_index: Index/Table in DocumentStore where labeled questions are stored
        :param doc_index: Index/Table in DocumentStore where documents that are used for evaluation are stored
        :param top_k: How many documents to return per question
        """

        # extract all questions for evaluation
        filter = {"origin": label_origin}

        # TODO get Documents back here
        questions = self.document_store.get_all_documents_in_index(index=label_index, filters=filter)

        # calculate recall and mean-average-precision
        correct_retrievals = 0
        summed_avg_precision = 0
        for q_idx, question in enumerate(questions):
            question_string = question["_source"]["question"]
            retrieved_docs = self.retrieve(question_string, top_k=top_k, index=doc_index)
            # check if correct doc in retrieved docs
            for doc_idx, doc in enumerate(retrieved_docs):
                if doc.meta["doc_id"] == question["_source"]["doc_id"]:
                    correct_retrievals += 1
                    summed_avg_precision += 1 / (doc_idx + 1)  # type: ignore
                    break

        number_of_questions = q_idx + 1
        recall = correct_retrievals / number_of_questions
        mean_avg_precision = summed_avg_precision / number_of_questions

        logger.info((f"For {correct_retrievals} out of {number_of_questions} questions ({recall:.2%}), the answer was in"
                     f" the top-{top_k} candidate passages selected by the retriever."))

        return {"recall": recall, "map": mean_avg_precision}