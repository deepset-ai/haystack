import logging
import time
from statistics import mean
from typing import Optional, Dict, Any, List
from collections import defaultdict

from haystack.reader.base import BaseReader
from haystack.retriever.base import BaseRetriever
from haystack import MultiLabel
from haystack.eval import calculate_average_precision, eval_counts_reader_batch, calculate_reader_metrics, \
    eval_counts_reader

logger = logging.getLogger(__name__)


class Finder:
    """
    Finder ties together instances of the Reader and Retriever class.

    It provides an interface to predict top n answers for a given question.
    """

    def __init__(self, reader: Optional[BaseReader], retriever: Optional[BaseRetriever]):
        """
        Initialize a Finder instance.

        :param reader: Reader instance
        :param retriever: Retriever instance
        """
        self.retriever = retriever
        self.reader = reader
        if self.reader is None and self.retriever is None:
            raise AttributeError("Finder: self.reader and self.retriever can not be both None")

    def get_answers(self, question: str, top_k_reader: int = 1, top_k_retriever: int = 10, filters: Optional[dict] = None, index: str = None):
        """
        Get top k answers for a given question.

        :param question: The question string
        :param top_k_reader: Number of answers returned by the reader
        :param top_k_retriever: Number of text units to be retrieved
        :param filters: Limit scope to documents having the given meta data values.
            The format for the dict is `{"key-1": ["value-1", "value-2"], "key-2": ["value-3]" ...}``
        :param index: Index to retrieve documents from
        :return:
        """

        if self.retriever is None or self.reader is None:
            raise AttributeError("Finder.get_answers requires self.retriever AND self.reader")

        # 1) Apply retriever(with optional filters) to get fast candidate documents
        documents = self.retriever.retrieve(question, filters=filters, top_k=top_k_retriever, index=index)

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

    def get_answers_via_similar_questions(self, question: str, top_k_retriever: int = 10, filters: Optional[dict] = None, index: str = None):
        """
        Get top k answers for a given question using only a retriever.

        :param question: The question string
        :param top_k_retriever: Number of text units to be retrieved
        :param filters: Limit scope to documents having the given meta data values.
            The format for the dict is ``{"key-1": ["value-1", "value-2"], "key-2": ["value-3]" ...}``
        :param index: Index to retrieve documents from
        :return:
        """

        if self.retriever is None:
            raise AttributeError("Finder.get_answers_via_similar_questions requires self.retriever")

        results = {"question": question, "answers": []}  # type: Dict[str, Any]


        # 1) Apply retriever to match similar questions via cosine similarity of embeddings
        documents = self.retriever.retrieve(question, top_k=top_k_retriever, filters=filters, index=index)

        # 2) Format response
        for doc in documents:
            #TODO proper calibratation of pseudo probabilities
            cur_answer = {
                "question": doc.question,
                "answer": doc.text,
                "document_id": doc.id,
                "context": doc.text,
                "score": doc.score,
                "probability": doc.probability,
                "offset_start": 0,
                "offset_end": len(doc.text),
                "meta": doc.meta
             }

            results["answers"].append(cur_answer)

        return results

    def eval(
        self,
        label_index: str,
        doc_index: str,
        label_origin: str = "gold_label",
        top_k_retriever: int = 10,
        top_k_reader: int = 10,
    ):
        """
        Evaluation of the whole pipeline by first evaluating the Retriever and then evaluating the Reader on the result
        of the Retriever.
        Returns a dict containing the following metrics:
            - ``"retriever_recall"``: Proportion of questions for which correct document is among retrieved documents
            - ``"retriever_map"``: Mean of average precision for each question. Rewards retrievers that give relevant
              documents a higher rank.
            - ``"reader_top1_accuracy"``: Proportion of highest ranked predicted answers that overlap with corresponding correct answer
            - ``"reader_top1_accuracy_has_answer"``: Proportion of highest ranked predicted answers that overlap
              with corresponding correct answer for answerable questions
            - ``"reader_top_k_accuracy"``: Proportion of predicted answers that overlap with corresponding correct answer
            - ``"reader_topk_accuracy_has_answer"``: Proportion of predicted answers that overlap with corresponding correct answer
              for answerable questions
            - ``"reader_top1_em"``: Proportion of exact matches of highest ranked predicted answers with their corresponding
              correct answers
            - ``"reader_top1_em_has_answer"``: Proportion of exact matches of highest ranked predicted answers with their corresponding
              correct answers for answerable questions
            - ``"reader_topk_em"``: Proportion of exact matches of predicted answers with their corresponding correct answers
            - ``"reader_topk_em_has_answer"``: Proportion of exact matches of predicted answers with their corresponding
              correct answers for answerable questions
            - ``"reader_top1_f1"``: Average overlap between highest ranked predicted answers and their corresponding correct answers
            - ``"reader_top1_f1_has_answer"``: Average overlap between highest ranked predicted answers and their corresponding
              correct answers for answerable questions
            - ``"reader_topk_f1"``: Average overlap between predicted answers and their corresponding correct answers
            - ``"reader_topk_f1_has_answer"``: Average overlap between predicted answers and their corresponding correct answers
              for answerable questions
            - ``"reader_top1_no_answer_accuracy"``: Proportion of correct predicting unanswerable question at highest ranked prediction
            - ``"reader_topk_no_answer_accuracy"``: Proportion of correct predicting unanswerable question among all predictions
            - ``"total_retrieve_time"``: Time retriever needed to retrieve documents for all questions
            - ``"avg_retrieve_time"``: Average time needed to retrieve documents for one question
            - ``"total_reader_time"``: Time reader needed to extract answer out of retrieved documents for all questions
              where the correct document is among the retrieved ones
            - ``"avg_reader_time"``: Average time needed to extract answer out of retrieved documents for one question
            - ``"total_finder_time"``: Total time for whole pipeline

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

        finder_start_time = time.time()
        # extract all questions for evaluation
        filters = {"origin": [label_origin]}
        questions = self.retriever.document_store.get_all_labels_aggregated(index=label_index, filters=filters)

        counts = defaultdict(float)  # type: Dict[str, float]
        retrieve_times = []
        read_times = []

        # retrieve documents
        questions_with_docs = []
        retriever_start_time = time.time()
        for q_idx, question in enumerate(questions):
            question_string = question.question
            single_retrieve_start = time.time()
            retrieved_docs = self.retriever.retrieve(question_string, top_k=top_k_retriever, index=doc_index)
            retrieve_times.append(time.time() - single_retrieve_start)

            # check if correct doc among retrieved docs
            for doc_idx, doc in enumerate(retrieved_docs):
                if doc.id in question.multiple_document_ids:
                    counts["correct_retrievals"] += 1
                    counts["summed_avg_precision_retriever"] += 1 / (doc_idx + 1)
                    questions_with_docs.append({
                        "question": question,
                        "docs": retrieved_docs
                    })
                    break

        retriever_total_time = time.time() - retriever_start_time
        counts["number_of_questions"] = q_idx + 1

        previous_return_no_answers = self.reader.return_no_answers
        self.reader.return_no_answers = True

        # extract answers
        reader_start_time = time.time()
        for q_idx, question_docs in enumerate(questions_with_docs):
            if (q_idx + 1) % 100 == 0:
                print(f"Processed {q_idx+1} questions.")

            question = question_docs["question"]  # type: ignore
            question_string = question.question
            docs = question_docs["docs"]  # type: ignore
            single_reader_start = time.time()
            predicted_answers = self.reader.predict(question_string, docs, top_k=top_k_reader)  # type: ignore
            read_times.append(time.time() - single_reader_start)
            counts = eval_counts_reader(question, predicted_answers, counts)

        counts["number_of_has_answer"] = counts["correct_retrievals"] - counts["number_of_no_answer"]

        reader_total_time = time.time() - reader_start_time
        finder_total_time = time.time() - finder_start_time

        self.reader.return_no_answers = previous_return_no_answers  # type: ignore

        logger.info((f"{counts['correct_readings_topk']} out of {counts['number_of_questions']} questions were correctly"
                     f" answered {(counts['correct_readings_topk']/counts['number_of_questions']):.2%})."))
        logger.info((f"{counts['number_of_questions']-counts['correct_retrievals']} questions could not be answered due "
                    f"to the retriever."))
        logger.info((f"{counts['correct_retrievals']-counts['correct_readings_topk']} questions could not be answered "
                    f"due to the reader."))

        eval_results = self.calc_eval_results(counts)
        eval_results["total_retrieve_time"] = retriever_total_time
        eval_results["avg_retrieve_time"] = mean(retrieve_times)
        eval_results["total_reader_time"] = reader_total_time
        eval_results["avg_reader_time"] = mean(read_times)
        eval_results["total_finder_time"] = finder_total_time

        return eval_results

    def eval_batch(
        self,
        label_index: str,
        doc_index : str,
        label_origin: str = "gold_label",
        top_k_retriever: int = 10,
        top_k_reader: int = 10,
        batch_size: int = 50
    ):
        """
        Evaluation of the whole pipeline by first evaluating the Retriever and then evaluating the Reader on the result
        of the Retriever. Passes all retrieved question-document pairs to the Reader at once.
        Returns a dict containing the following metrics:
            - ``"retriever_recall"``: Proportion of questions for which correct document is among retrieved documents
            - ``"retriever_map"``: Mean of average precision for each question. Rewards retrievers that give relevant
              documents a higher rank.
            - ``"reader_top1_accuracy"``: Proportion of highest ranked predicted answers that overlap with corresponding correct answer
            - ``"reader_top1_accuracy_has_answer"``: Proportion of highest ranked predicted answers that overlap
              with corresponding correct answer for answerable questions
            - ``"reader_top_k_accuracy"``: Proportion of predicted answers that overlap with corresponding correct answer
            - ``"reader_topk_accuracy_has_answer"``: Proportion of predicted answers that overlap with corresponding correct answer
              for answerable questions
            - ``"reader_top1_em"``: Proportion of exact matches of highest ranked predicted answers with their corresponding
              correct answers
            - ``"reader_top1_em_has_answer"``: Proportion of exact matches of highest ranked predicted answers with their corresponding
              correct answers for answerable questions
            - ``"reader_topk_em"``: Proportion of exact matches of predicted answers with their corresponding correct answers
            - ``"reader_topk_em_has_answer"``: Proportion of exact matches of predicted answers with their corresponding
              correct answers for answerable questions
            - ``"reader_top1_f1"``: Average overlap between highest ranked predicted answers and their corresponding correct answers
            - ``"reader_top1_f1_has_answer"``: Average overlap between highest ranked predicted answers and their corresponding
              correct answers for answerable questions
            - ``"reader_topk_f1"``: Average overlap between predicted answers and their corresponding correct answers
            - ``"reader_topk_f1_has_answer"``: Average overlap between predicted answers and their corresponding correct answers
              for answerable questions
            - ``"reader_top1_no_answer_accuracy"``: Proportion of correct predicting unanswerable question at highest ranked prediction
            - ``"reader_topk_no_answer_accuracy"``: Proportion of correct predicting unanswerable question among all predictions
            - ``"total_retrieve_time"``: Time retriever needed to retrieve documents for all questions
            - ``"avg_retrieve_time"``: Average time needed to retrieve documents for one question
            - ``"total_reader_time"``: Time reader needed to extract answer out of retrieved documents for all questions
              where the correct document is among the retrieved ones
            - ``"avg_reader_time"``: Average time needed to extract answer out of retrieved documents for one question
            - ``"total_finder_time"``: Total time for whole pipeline

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
            raise Exception("Finder needs to have a reader and retriever for the evalutaion.")

        counts = defaultdict(float)  # type: Dict[str, float]
        finder_start_time = time.time()

        # extract all questions for evaluation
        filters = {"origin": [label_origin]}
        questions = self.retriever.document_store.get_all_labels_aggregated(index=label_index, filters=filters)
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
        predictions = self.reader.predict_batch(questions_with_correct_doc,
                                                top_k_per_question=top_k_reader, batch_size=batch_size)
        reader_total_time = time.time() - reader_start_time

        for pred in predictions:
            counts = eval_counts_reader_batch(pred, counts)

        finder_total_time = time.time() - finder_start_time

        results = calculate_reader_metrics(counts, correct_retrievals)
        results["retriever_recall"] = correct_retrievals / number_of_questions
        results["retriever_map"] = summed_avg_precision_retriever / number_of_questions
        results["total_retrieve_time"] = retriever_total_time
        results["avg_retrieve_time"] = retriever_total_time / number_of_questions
        results["total_reader_time"] = reader_total_time
        results["avg_reader_time"] = reader_total_time / correct_retrievals
        results["total_finder_time"] = finder_total_time

        logger.info((f"{counts['correct_readings_topk']} out of {number_of_questions} questions were correctly "
                     f"answered ({(counts['correct_readings_topk'] / number_of_questions):.2%})."))
        logger.info(f"{number_of_questions - correct_retrievals} questions could not be answered due to the retriever.")
        logger.info(f"{correct_retrievals - counts['correct_readings_topk']} questions could not be answered due to the reader.")

        return results


    def _retrieve_docs(self, questions: List[MultiLabel], top_k: int, doc_index: str):
        # Retrieves documents for a list of Labels (= questions)
        questions_with_docs = []

        for question in questions:
            question_string = question.question
            retrieved_docs = self.retriever.retrieve(question_string, top_k=top_k, index=doc_index)  # type: ignore
            questions_with_docs.append({
                "question": question,
                "docs": retrieved_docs
            })

        return questions_with_docs


    @staticmethod
    def print_eval_results(finder_eval_results: Dict):
        print("\n___Retriever Metrics in Finder___")
        print(f"Retriever Recall            : {finder_eval_results['retriever_recall']:.3f}")
        print(f"Retriever Mean Avg Precision: {finder_eval_results['retriever_map']:.3f}")

        # Reader is only evaluated with those questions, where the correct document is among the retrieved ones
        print("\n___Reader Metrics in Finder___")
        print("Top-k accuracy")
        print(f"Reader Top-1 accuracy             : {finder_eval_results['reader_top1_accuracy']:.3f}")
        print(f"Reader Top-1 accuracy (has answer): {finder_eval_results['reader_top1_accuracy_has_answer']:.3f}")
        print(f"Reader Top-k accuracy             : {finder_eval_results['reader_topk_accuracy']:.3f}")
        print(f"Reader Top-k accuracy (has answer): {finder_eval_results['reader_topk_accuracy_has_answer']:.3f}")
        print("Exact Match")
        print(f"Reader Top-1 EM                   : {finder_eval_results['reader_top1_em']:.3f}")
        print(f"Reader Top-1 EM (has answer)      : {finder_eval_results['reader_top1_em_has_answer']:.3f}")
        print(f"Reader Top-k EM                   : {finder_eval_results['reader_topk_em']:.3f}")
        print(f"Reader Top-k EM (has answer)      : {finder_eval_results['reader_topk_em_has_answer']:.3f}")
        print("F1 score")
        print(f"Reader Top-1 F1                   : {finder_eval_results['reader_top1_f1']:.3f}")
        print(f"Reader Top-1 F1 (has answer)      : {finder_eval_results['reader_top1_f1_has_answer']:.3f}")
        print(f"Reader Top-k F1                   : {finder_eval_results['reader_topk_f1']:.3f}")
        print(f"Reader Top-k F1 (has answer)      : {finder_eval_results['reader_topk_f1_has_answer']:.3f}")
        if finder_eval_results['reader_top1_no_answer_accuracy']:
            print("No Answer")
            print(f"Reader Top-1 no-answer accuracy   : {finder_eval_results['reader_top1_no_answer_accuracy']:.3f}")
            print(f"Reader Top-k no-answer accuracy   : {finder_eval_results['reader_topk_no_answer_accuracy']:.3f}")

        # Time measurements
        print("\n___Time Measurements___")
        print(f"Total retrieve time           : {finder_eval_results['total_retrieve_time']:.3f}")
        print(f"Avg retrieve time per question: {finder_eval_results['avg_retrieve_time']:.3f}")
        print(f"Total reader timer            : {finder_eval_results['total_reader_time']:.3f}")
        print(f"Avg read time per question    : {finder_eval_results['avg_reader_time']:.3f}")
        print(f"Total Finder time             : {finder_eval_results['total_finder_time']:.3f}")

    @staticmethod
    def calc_eval_results(eval_counts: Dict):
        eval_results = {}
        number_of_questions = eval_counts["number_of_questions"]
        correct_retrievals = eval_counts["correct_retrievals"]
        number_of_has_answer = eval_counts["number_of_has_answer"]
        number_of_no_answer = eval_counts["number_of_no_answer"]

        eval_results["retriever_recall"] = eval_counts["correct_retrievals"] / number_of_questions
        eval_results["retriever_map"] = eval_counts["summed_avg_precision_retriever"] / number_of_questions

        eval_results["reader_top1_accuracy"] = eval_counts["correct_readings_top1"] / correct_retrievals
        eval_results["reader_top1_accuracy_has_answer"] = eval_counts["correct_readings_top1_has_answer"] / number_of_has_answer
        eval_results["reader_topk_accuracy"] = eval_counts["correct_readings_topk"] / correct_retrievals
        eval_results["reader_topk_accuracy_has_answer"] = eval_counts["correct_readings_topk_has_answer"] / number_of_has_answer
        eval_results["reader_top1_em"] = eval_counts["exact_matches_top1"] / correct_retrievals
        eval_results["reader_top1_em_has_answer"] = eval_counts["exact_matches_top1_has_answer"] / number_of_has_answer
        eval_results["reader_topk_em"] = eval_counts["exact_matches_topk"] / correct_retrievals
        eval_results["reader_topk_em_has_answer"] = eval_counts["exact_matches_topk_has_answer"] / number_of_has_answer
        eval_results["reader_top1_f1"] = eval_counts["summed_f1_top1"] / correct_retrievals
        eval_results["reader_top1_f1_has_answer"] = eval_counts["summed_f1_top1_has_answer"] / number_of_has_answer
        eval_results["reader_topk_f1"] = eval_counts["summed_f1_topk"] / correct_retrievals
        eval_results["reader_topk_f1_has_answer"] = eval_counts["summed_f1_topk_has_answer"] / number_of_has_answer
        if number_of_no_answer:
            eval_results["reader_top1_no_answer_accuracy"] = eval_counts["correct_no_answers_top1"] / number_of_no_answer
            eval_results["reader_topk_no_answer_accuracy"] = eval_counts["correct_no_answers_topk"] / number_of_no_answer
        else:
            eval_results["reader_top1_no_answer_accuracy"] = None
            eval_results["reader_topk_no_answer_accuracy"] = None

        return eval_results

