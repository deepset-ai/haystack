from typing import List, Optional, Sequence, Dict, Tuple

import numpy as np
from scipy.special import expit
from abc import abstractmethod
from copy import deepcopy
from functools import wraps
from time import perf_counter

from haystack.schema import Document, Answer, Span
from haystack.nodes.base import BaseComponent


class BaseReader(BaseComponent):
    return_no_answers: bool
    outgoing_edges = 1
    query_count = 0
    query_time = 0

    @abstractmethod
    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        pass

    @abstractmethod
    def predict_batch(self, query_doc_list: List[dict], top_k: Optional[int] = None, batch_size: Optional[int] = None):
        pass

    @staticmethod
    def _calc_no_answer(no_ans_gaps: Sequence[float],
                        best_score_answer: float,
                        use_confidence_scores: bool = True) -> Tuple[Answer, float]:
        # "no answer" scores and positive answers scores are difficult to compare, because
        # + a positive answer score is related to one specific document
        # - a "no answer" score is related to all input documents
        # Thus we compute the "no answer" score relative to the best possible answer and adjust it by
        # the most significant difference between scores.
        # Most significant difference: a model switching from predicting an answer to "no answer" (or vice versa).
        # No_ans_gap is a list of this most significant difference per document
        no_ans_gap_array = np.array(no_ans_gaps)
        max_no_ans_gap = np.max(no_ans_gap_array)
        # all passages "no answer" as top score
        if np.sum(no_ans_gap_array < 0) == len(no_ans_gap_array):
            no_ans_score = best_score_answer - max_no_ans_gap  # max_no_ans_gap is negative, so it increases best pos score
        else:  # case: at least one passage predicts an answer (positive no_ans_gap)
            no_ans_score = best_score_answer - max_no_ans_gap

        no_ans_prediction = Answer(answer="",
                                   type="extractive",
                                   score=float(expit(np.asarray(no_ans_score) / 8)) if use_confidence_scores else no_ans_score,  # just a pseudo prob for now or old score,
                                   context=None,
                                   offsets_in_context=[Span(start=0, end=0)],
                                   offsets_in_document=[Span(start=0, end=0)],
                                   document_id=None,
                                   meta=None)

        return no_ans_prediction, max_no_ans_gap

    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None):  # type: ignore
        self.query_count += 1
        if documents:
            predict = self.timing(self.predict, "query_time")
            results = predict(query=query, documents=documents, top_k=top_k)
        else:
            results = {"answers": []}

        # Add corresponding document_name and more meta data, if an answer contains the document_id
        for ans in results["answers"]:
            if ans.meta is None:
                ans.meta = {}
            # get meta from doc
            meta_from_doc = {}
            for doc in documents:
                if doc.id == ans.document_id:
                    meta_from_doc = deepcopy(doc.meta)
                    break
            # append to "own" meta
            ans.meta.update(meta_from_doc)

        return results, "output_1"

    def run_batch(self, query_doc_list: List[Dict], top_k: Optional[int] = None):
        """ A unoptimized implementation of running Reader queries in batch """
        self.query_count += len(query_doc_list)
        results = []
        if query_doc_list:
            for qd in query_doc_list:
                q = qd["queries"]
                docs = qd["docs"]
                predict = self.timing(self.predict, "query_time")
                result = predict(query=q, documents=docs, top_k=top_k)
                results.append(result)
        else:
            results = [{"answers": [], "query": ""}]
        return {"results": results}, "output_1"

    def timing(self, fn, attr_name):
        """Wrapper method used to time functions. """
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if attr_name not in self.__dict__:
                self.__dict__[attr_name] = 0
            tic = perf_counter()
            ret = fn(*args, **kwargs)
            toc = perf_counter()
            self.__dict__[attr_name] += toc - tic
            return ret
        return wrapper

    def print_time(self):
        print("Reader (Speed)")
        print("---------------")
        if not self.query_count:
            print("No querying performed via Retriever.run()")
        else:
            print(f"Queries Performed: {self.query_count}")
            print(f"Query time: {self.query_time}s")
            print(f"{self.query_time / self.query_count} seconds per query")


