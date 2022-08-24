import itertools
from typing import List, Optional, Sequence, Tuple, Union

from abc import abstractmethod
from copy import deepcopy
from functools import wraps
from time import perf_counter

import numpy as np
from scipy.special import expit

from haystack.schema import Document, Answer, Span, MultiLabel
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
    def predict_batch(
        self, queries: List[str], documents: Union[List[Document], List[List[Document]]], top_k: Optional[int] = None
    ):
        pass

    @staticmethod
    def _calc_no_answer(
        no_ans_gaps: Sequence[float], best_score_answer: float, use_confidence_scores: bool = True
    ) -> Tuple[Answer, float]:
        # "no answer" scores and positive answers scores are difficult to compare, because
        # + a positive answer score is related to one specific document
        # - a "no answer" score is related to all input documents
        # Thus we compute the "no answer" score relative to the best possible answer and adjust it by
        # the most significant difference between scores.
        # Most significant difference: a model switching from predicting an answer to "no answer" (or vice versa).
        # No_ans_gap is a list of this most significant difference per document
        no_ans_gap_array = np.array(no_ans_gaps)
        max_no_ans_gap = np.max(no_ans_gap_array)
        # case 1: all passages "no answer" as top score
        # max_no_ans_gap is negative, so it increases best pos score
        # case 2: at least one passage predicts an answer (positive no_ans_gap)
        no_ans_score = best_score_answer - max_no_ans_gap

        no_ans_prediction = Answer(
            answer="",
            type="extractive",
            score=float(expit(np.asarray(no_ans_score) / 8))
            if use_confidence_scores
            else no_ans_score,  # just a pseudo prob for now or old score,
            context=None,
            offsets_in_context=[Span(start=0, end=0)],
            offsets_in_document=[Span(start=0, end=0)],
            document_id=None,
            meta=None,
        )

        return no_ans_prediction, max_no_ans_gap

    @staticmethod
    def add_doc_meta_data_to_answer(documents: List[Document], answer):
        # Add corresponding document_name and more meta data, if the answer contains the document_id
        if answer.meta is None:
            answer.meta = {}
        # get meta from doc
        meta_from_doc = {}
        for doc in documents:
            if doc.id == answer.document_id:
                meta_from_doc = deepcopy(doc.meta)
                break
        # append to "own" meta
        answer.meta.update(meta_from_doc)
        return answer

    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None, labels: Optional[MultiLabel] = None, add_isolated_node_eval: bool = False):  # type: ignore
        self.query_count += 1
        predict = self.timing(self.predict, "query_time")
        if documents:
            results = predict(query=query, documents=documents, top_k=top_k)
        else:
            results = {"answers": []}

        # Add corresponding document_name and more meta data, if an answer contains the document_id
        results["answers"] = [
            BaseReader.add_doc_meta_data_to_answer(documents=documents, answer=answer) for answer in results["answers"]
        ]

        # run evaluation with labels as node inputs
        if add_isolated_node_eval and labels is not None:
            relevant_documents = {label.document.id: label.document for label in labels.labels}.values()
            results_label_input = predict(query=query, documents=relevant_documents, top_k=top_k)

            # Add corresponding document_name and more meta data, if an answer contains the document_id
            results["answers_isolated"] = [
                BaseReader.add_doc_meta_data_to_answer(documents=documents, answer=answer)
                for answer in results_label_input["answers"]
            ]

        return results, "output_1"

    def run_batch(  # type: ignore
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
        labels: Optional[List[MultiLabel]] = None,
        add_isolated_node_eval: bool = False,
    ):
        self.query_count += len(queries)
        if not documents:
            return {"answers": []}, "output_1"

        predict_batch = self.timing(self.predict_batch, "query_time")

        results = predict_batch(queries=queries, documents=documents, top_k=top_k, batch_size=batch_size)

        # Add corresponding document_name and more meta data, if an answer contains the document_id
        answer_iterator = itertools.chain.from_iterable(results["answers"])
        if isinstance(documents[0], Document):
            answer_iterator = itertools.chain.from_iterable(itertools.chain.from_iterable(results["answers"]))
        flattened_documents = []
        for doc_list in documents:
            if isinstance(doc_list, list):
                flattened_documents.extend(doc_list)
            else:
                flattened_documents.append(doc_list)

        for answer in answer_iterator:
            BaseReader.add_doc_meta_data_to_answer(documents=flattened_documents, answer=answer)

        # run evaluation with labels as node inputs
        if add_isolated_node_eval and labels is not None:
            relevant_documents = []
            for labelx in labels:
                relevant_documents.append([label.document for label in labelx.labels])
            results_label_input = predict_batch(queries=queries, documents=relevant_documents, top_k=top_k)

            # Add corresponding document_name and more meta data, if an answer contains the document_id
            answer_iterator = itertools.chain.from_iterable(results_label_input["answers"])
            if isinstance(documents[0], Document):
                if isinstance(queries, list):
                    answer_iterator = itertools.chain.from_iterable(
                        itertools.chain.from_iterable(results_label_input["answers"])
                    )
            flattened_documents = []
            for doc_list in documents:
                if isinstance(doc_list, list):
                    flattened_documents.extend(doc_list)
                else:
                    flattened_documents.append(doc_list)

            for answer in answer_iterator:
                BaseReader.add_doc_meta_data_to_answer(documents=flattened_documents, answer=answer)

            results["answers_isolated"] = results_label_input["answers"]

        return results, "output_1"

    def timing(self, fn, attr_name):
        """Wrapper method used to time functions."""

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
