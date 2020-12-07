import numpy as np
from scipy.special import expit
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Sequence

from haystack import Document


class BaseReader(ABC):
    return_no_answers: bool
    outgoing_edges = 1

    @abstractmethod
    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        pass

    @abstractmethod
    def predict_batch(self, query_doc_list: List[dict], top_k: Optional[int] = None, batch_size: Optional[int] = None):
        pass

    @staticmethod
    def _calc_no_answer(no_ans_gaps: Sequence[float], best_score_answer: float):
        # "no answer" scores and positive answers scores are difficult to compare, because
        # + a positive answer score is related to one specific document
        # - a "no answer" score is related to all input documents
        # Thus we compute the "no answer" score relative to the best possible answer and adjust it by
        # the most significant difference between scores.
        # Most significant difference: a model switching from predicting an answer to "no answer" (or vice versa).
        # No_ans_gap is a list of this most significant difference per document
        no_ans_gaps = np.array(no_ans_gaps)
        max_no_ans_gap = np.max(no_ans_gaps)
        # all passages "no answer" as top score
        if (np.sum(no_ans_gaps < 0) == len(no_ans_gaps)):  # type: ignore
            no_ans_score = best_score_answer - max_no_ans_gap  # max_no_ans_gap is negative, so it increases best pos score
        else:  # case: at least one passage predicts an answer (positive no_ans_gap)
            no_ans_score = best_score_answer - max_no_ans_gap

        no_ans_prediction = {"answer": None,
               "score": no_ans_score,
               "probability": float(expit(np.asarray(no_ans_score) / 8)),  # just a pseudo prob for now
               "context": None,
               "offset_start": 0,
               "offset_end": 0,
               "document_id": None,
               "meta": None,}
        return no_ans_prediction, max_no_ans_gap

    def run(self, query: str, documents: List[Document], top_k_reader: Optional[int] = None, **kwargs):
        if documents:
            results = self.predict(query=query, documents=documents, top_k=top_k_reader)
        else:
            results = {"answers": [], "query": query}

        # Add corresponding document_name and more meta data, if an answer contains the document_id
        for ans in results["answers"]:
            ans["meta"] = {}
            for doc in documents:
                if doc.id == ans["document_id"]:
                    ans["meta"] = deepcopy(doc.meta)

        results.update(**kwargs)
        return results, "output_1"
