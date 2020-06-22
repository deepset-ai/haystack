import logging
from collections import Counter
import time
from typing import Optional, Dict, Any, List

import numpy as np
from scipy.special import expit

from haystack.reader.base import BaseReader
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.retriever.base import BaseRetriever
from haystack.eval import calculate_average_precision, eval_counts_reader_batch, eval_counts_reader, calculate_reader_metrics

logger = logging.getLogger(__name__)


class Finder:
    """
    Finder ties together instances of the Reader and Retriever class.

    It provides an interface to predict top n answers for a given question.
    """

    def __init__(self, reader: Optional[BaseReader], retriever: Optional[BaseRetriever]):
        self.retriever = retriever
        self.reader = reader
        if self.reader is None and self.retriever is None:
            raise AttributeError("Finder: self.reader and self.retriever can not be both None")

    def get_answers(self, question: str, top_k_reader: int = 1, top_k_retriever: int = 10, filters: Optional[dict] = None):
        """
        Get top k answers for a given question.

        :param question: the question string
        :param top_k_reader: number of answers returned by the reader
        :param top_k_retriever: number of text units to be retrieved
        :param filters: limit scope to documents having the given tags and their corresponding values.
            The format for the dict is {"tag-1": ["value-1","value-2"], "tag-2": ["value-3]" ...}
        :return:
        """

        if self.retriever is None or self.reader is None:
            raise AttributeError("Finder.get_answers requires self.retriever AND self.reader")

        # 1) Apply retriever(with optional filters) to get fast candidate documents
        documents = self.retriever.retrieve(question, filters=filters, top_k=top_k_retriever)

        if len(documents) == 0:
            logger.info("Retriever did not return any documents. Skipping reader ...")
            empty_result = {"question": question, "answers": []}
            return empty_result

        # 2) Apply reader to get granular answer(s)
        len_chars = sum([len(d.text) for d in documents])
        logger.info(f"Reader is looking for detailed answer in {len_chars} chars ...")

        results = self.reader.predict(question=question,
                                      documents=documents,
                                      top_k=top_k_reader)  # type: Dict[str, Any]

        # Add corresponding document_name and more meta data, if an answer contains the document_id
        for ans in results["answers"]:
            ans["meta"] = {}
            for doc in documents:
                if doc.id == ans["document_id"]:
                    ans["meta"] = doc.meta

        return results

    def get_answers_via_similar_questions(self, question: str, top_k_retriever: int = 10, filters: Optional[dict] = None):
        """
        Get top k answers for a given question using only a retriever.

        :param question: the question string
        :param top_k_retriever: number of text units to be retrieved
        :param filters: limit scope to documents having the given tags and their corresponding values.
            The format for the dict is {"tag-1": "value-1", "tag-2": "value-2" ...}
        :return:
        """

        if self.retriever is None:
            raise AttributeError("Finder.get_answers_via_similar_questions requires self.retriever")

        results = {"question": question, "answers": []}  # type: Dict[str, Any]

        # 1) Optional: reduce the search space via document tags
        if filters:
            logging.info(f"Apply filters: {filters}")
            candidate_doc_ids = self.retriever.document_store.get_document_ids_by_tags(filters)  # type: ignore
            logger.info(f"Got candidate IDs due to filters:  {candidate_doc_ids}")

            if len(candidate_doc_ids) == 0:
                # We didn't find any doc matching the filters
                return results

        else:
            candidate_doc_ids = None  # type: ignore

        # 2) Apply retriever to match similar questions via cosine similarity of embeddings
        documents = self.retriever.retrieve(question, top_k=top_k_retriever, candidate_doc_ids=candidate_doc_ids)  # type: ignore

        # 3) Format response
        for doc in documents:
            #TODO proper calibratation of pseudo probabilities
            cur_answer = {"question": doc.question, "answer": doc.text, "context": doc.text,  # type: ignore
                          "score": doc.query_score, "offset_start": 0, "offset_end": len(doc.text), "meta": doc.meta
                          }
            if self.retriever.embedding_model:  # type: ignore
                probability = (doc.query_score + 1) / 2  # type: ignore
            else:
                probability = float(expit(np.asarray(doc.query_score / 8)))  # type: ignore

            cur_answer["probability"] = probability
            results["answers"].append(cur_answer)

        return results

    def eval(
        self,
        label_index: str = "feedback",
        doc_index: str = "eval_document",
        label_origin: str = "gold_label",
        top_k_retriever: int = 10,
        top_k_reader: int = 10,
    ):
        """
        Evaluation of the whole pipeline by first evaluating the Retriever and then evaluating the Reader on the result
        of the Retriever.

        Returns a dict containing the following metrics:
            - "retriever_recall": Proportion of questions for which correct document is among retrieved documents
            - "retriever_map": Mean of average precision for each question. Rewards retrievers that give relevant
              documents a higher rank.
            - "reader_top1_accuracy": Proportion of highest ranked predicted answers that overlap with corresponding correct answer
            - "reader_top1_accuracy_has_answer": Proportion of highest ranked predicted answers that overlap
                                                 with corresponding correct answer for answerable questions
            - "reader_top_k_accuracy": Proportion of predicted answers that overlap with corresponding correct answer
            - "reader_topk_accuracy_has_answer": Proportion of predicted answers that overlap with corresponding correct answer
                                                 for answerable questions
            - "reader_top1_em": Proportion of exact matches of highest ranked predicted answers with their corresponding
                                correct answers
            - "reader_top1_em_has_answer": Proportion of exact matches of highest ranked predicted answers with their corresponding
                                           correct answers for answerable questions
            - "reader_topk_em": Proportion of exact matches of predicted answers with their corresponding correct answers
            - "reader_topk_em_has_answer": Proportion of exact matches of predicted answers with their corresponding
                                           correct answers for answerable questions
            - "reader_top1_f1": Average overlap between highest ranked predicted answers and their corresponding correct answers
            - "reader_top1_f1_has_answer": Average overlap between highest ranked predicted answers and their corresponding
                                           correct answers for answerable questions
            - "reader_topk_f1": Average overlap between predicted answers and their corresponding correct answers
            - "reader_topk_f1_has_answer": Average overlap between predicted answers and their corresponding correct answers
                                           for answerable questions
            - "reader_top1_no_answer_accuracy": Proportion of correct predicting unanswerable question at highest ranked prediction
            - "reader_topk_no_answer_accuracy": Proportion of correct predicting unanswerable question among all predictions
            - "total_retrieve_time": Time retriever needed to retrieve documents for all questions
            - "avg_retrieve_time": Average time needed to retrieve documents for one question
            - "total_reader_time": Time reader needed to extract answer out of retrieved documents for all questions
                                   where the correct document is among the retrieved ones
            - "avg_reader_time": Average time needed to extract answer out of retrieved documents for one question
            - "total_finder_time": Total time for whole pipeline

        :param label_index: Elasticsearch index where labeled questions are stored
        :type label_index: str
        :param doc_index: Elasticsearch index where documents that are used for evaluation are stored
        :type doc_index: str
        :param top_k_retriever: How many documents per question to return and pass to reader
        :type top_k_retriever: int
        :param top_k_reader: How many answers to return per question
        :type top_k_reader: int
        """

        if not self.reader or not self.retriever:
            raise Exception("Finder needs to have a reader and retriever for the evaluation.")
        if isinstance(self.reader, FARMReader):
            reader_type = "farm"
        elif isinstance(self.reader, TransformersReader):
            reader_type = "transformers"

        metric_counts = Counter()  # type: Counter
        finder_start_time = time.time()
        # extract all questions for evaluation
        filter = {"origin": label_origin}
        questions = list(self.retriever.document_store.get_all_documents_in_index(index=label_index, filters=filter))  # type: ignore
        number_of_questions = len(questions)

        # retrieve documents
        retriever_start_time = time.time()
        questions_with_docs = self._retrieve_docs(questions, top_k=top_k_retriever, doc_index=doc_index)
        retriever_total_time = time.time() - retriever_start_time

        questions_with_correct_doc, summed_avg_precision_retriever = calculate_average_precision(questions_with_docs)
        correct_retrievals = len(questions_with_correct_doc)

        # extract answers and count metrics
        previous_return_no_answers = self.reader.return_no_answers
        self.reader.return_no_answers = True
        reader_start_time = time.time()

        for q_idx, question in enumerate(questions_with_correct_doc):
            if (q_idx + 1) % 100 == 0:
                logger.info(f"Processed {q_idx + 1} questions in Reader.")
            question_string = question["question"]["_source"]["question"]
            docs = question["docs"]
            predicted_answers = self.reader.predict(question_string, docs, top_k_reader)
            metric_counts = eval_counts_reader(question, predicted_answers, metric_counts, reader_type)

        reader_total_time = time.time() - reader_start_time
        finder_total_time = time.time() - finder_start_time

        results = calculate_reader_metrics(metric_counts, correct_retrievals)
        results["retriever_recall"] = correct_retrievals / number_of_questions
        results["retriever_map"] = summed_avg_precision_retriever / number_of_questions
        results["total_retrieve_time"] = retriever_total_time
        results["avg_retrieve_time"] =  retriever_total_time / number_of_questions
        results["total_reader_time"] =  reader_total_time
        results["avg_reader_time"] =  reader_total_time / correct_retrievals
        results["total_finder_time"] =  finder_total_time

        self.reader.return_no_answers = previous_return_no_answers

        logger.info((f"{metric_counts['correct_readings_topk']} out of {number_of_questions} questions were correctly"
                     f" answered ({(metric_counts['correct_readings_topk']/number_of_questions):.2%})."))
        logger.info(f"{number_of_questions-correct_retrievals} questions could not be answered due to the retriever.")
        logger.info(f"{correct_retrievals-metric_counts['correct_readings_topk']} questions could not be answered due to the reader.")

        return results

    def eval_batch(
        self,
        label_index: str = "feedback",
        doc_index: str = "eval_document",
        label_origin: str = "gold_label",
        top_k_retriever: int = 10,
        top_k_reader: int = 10,
        batch_size: int = 50
        ):
        """
        Evaluation of the whole pipeline by first evaluating the Retriever and then evaluating the Reader on the result
        of the Retriever. Passes all retrieved question-document pairs to the Reader at once.

        Returns a dict containing the following metrics:
            - "retriever_recall": Proportion of questions for which correct document is among retrieved documents
            - "retriever_map": Mean of average precision for each question. Rewards retrievers that give relevant
              documents a higher rank.
            - "reader_top1_accuracy": Proportion of highest ranked predicted answers that overlap with corresponding correct answer
            - "reader_top1_accuracy_has_answer": Proportion of highest ranked predicted answers that overlap
                                                 with corresponding correct answer for answerable questions
            - "reader_top_k_accuracy": Proportion of predicted answers that overlap with corresponding correct answer
            - "reader_topk_accuracy_has_answer": Proportion of predicted answers that overlap with corresponding correct answer
                                                 for answerable questions
            - "reader_top1_em": Proportion of exact matches of highest ranked predicted answers with their corresponding
                                correct answers
            - "reader_top1_em_has_answer": Proportion of exact matches of highest ranked predicted answers with their corresponding
                                           correct answers for answerable questions
            - "reader_topk_em": Proportion of exact matches of predicted answers with their corresponding correct answers
            - "reader_topk_em_has_answer": Proportion of exact matches of predicted answers with their corresponding
                                           correct answers for answerable questions
            - "reader_top1_f1": Average overlap between highest ranked predicted answers and their corresponding correct answers
            - "reader_top1_f1_has_answer": Average overlap between highest ranked predicted answers and their corresponding
                                           correct answers for answerable questions
            - "reader_topk_f1": Average overlap between predicted answers and their corresponding correct answers
            - "reader_topk_f1_has_answer": Average overlap between predicted answers and their corresponding correct answers
                                           for answerable questions
            - "reader_top1_no_answer_accuracy": Proportion of correct predicting unanswerable question at highest ranked prediction
            - "reader_topk_no_answer_accuracy": Proportion of correct predicting unanswerable question among all predictions
            - "total_retrieve_time": Time retriever needed to retrieve documents for all questions
            - "avg_retrieve_time": Average time needed to retrieve documents for one question
            - "total_reader_time": Time reader needed to extract answer out of retrieved documents for all questions
                                   where the correct document is among the retrieved ones
            - "avg_reader_time": Average time needed to extract answer out of retrieved documents for one question
            - "total_finder_time": Total time for whole pipeline

        :param label_index: Elasticsearch index where labeled questions are stored
        :type label_index: str
        :param doc_index: Elasticsearch index where documents that are used for evaluation are stored
        :type doc_index: str
        :param top_k_retriever: How many documents per question to return and pass to reader
        :type top_k_retriever: int
        :param top_k_reader: How many answers to return per question
        :type top_k_reader: int
        :param batch_size: Number of samples per batch computed at once
        :type batch_size: int
        """

        if not self.reader or not self.retriever:
            raise Exception("Finder needs to have a reader and retriever for the evaluation.")

        metric_counts = Counter()  # type: Counter
        finder_start_time = time.time()
        # extract_all_questions for evaluation
        filter = {"origin": label_origin}
        questions = list(self.retriever.document_store.get_all_documents_in_index(index=label_index, filters=filter))  # type: ignore
        number_of_questions = len(questions)

        # retrieve documents
        retriever_start_time = time.time()
        questions_with_docs = self._retrieve_docs(questions, top_k=top_k_retriever, doc_index=doc_index)
        retriever_total_time = time.time() - retriever_start_time

        questions_with_correct_doc, summed_avg_precision_retriever = calculate_average_precision(questions_with_docs)
        correct_retrievals = len(questions_with_correct_doc)

        # extract answers
        previous_return_no_answers = self.reader.return_no_answers
        self.reader.return_no_answers = True
        reader_start_time = time.time()
        predictions = self.reader.predict_batch(questions_with_correct_doc, top_k_per_question=top_k_reader, batch_size=batch_size)
        reader_total_time = time.time() - reader_start_time

        for pred in predictions:
            metric_counts = eval_counts_reader_batch(pred, metric_counts)

        finder_total_time = time.time() - finder_start_time

        results = calculate_reader_metrics(metric_counts, correct_retrievals)
        results["retriever_recall"] = correct_retrievals / number_of_questions
        results["retriever_map"] = summed_avg_precision_retriever / number_of_questions
        results["total_retrieve_time"] = retriever_total_time
        results["avg_retrieve_time"] = retriever_total_time / number_of_questions
        results["total_reader_time"] = reader_total_time
        results["avg_reader_time"] = reader_total_time / correct_retrievals
        results["total_finder_time"] = finder_total_time

        self.reader.return_no_answers = previous_return_no_answers  # type: ignore

        logger.info((f"{metric_counts['correct_readings_topk']} out of {number_of_questions} questions were correctly"
                     f" answered ({(metric_counts['correct_readings_topk'] / number_of_questions):.2%})."))
        logger.info(f"{number_of_questions - correct_retrievals} questions could not be answered due to the retriever.")
        logger.info(
            f"{correct_retrievals - metric_counts['correct_readings_topk']} questions could not be answered due to the reader.")

        return results

    def _retrieve_docs(self, questions: List[dict], top_k: int, doc_index: str):
        # Retrieves documents for a list of questions.

        questions_with_docs = []

        for question in questions:
            question_string = question["_source"]["question"]
            retrieved_docs = self.retriever.retrieve(question_string, top_k=top_k, index=doc_index)  # type: ignore
            questions_with_docs.append({
                "question": question,
                "docs": retrieved_docs,
            })

        return questions_with_docs
