import logging

import numpy as np
from scipy.special import expit
import time
from statistics import mean

logger = logging.getLogger(__name__)


class Finder:
    """
    Finder ties together instances of the Reader and Retriever class.

    It provides an interface to predict top n answers for a given question.
    """

    def __init__(self, reader, retriever):
        self.retriever = retriever
        self.reader = reader

    def get_answers(self, question: str, top_k_reader: int = 1, top_k_retriever: int = 10, filters: dict = None):
        """
        Get top k answers for a given question.

        :param question: the question string
        :param top_k_reader: number of answers returned by the reader
        :param top_k_retriever: number of text units to be retrieved
        :param filters: limit scope to documents having the given tags and their corresponding values.
            The format for the dict is {"tag-1": ["value-1","value-2"], "tag-2": ["value-3]" ...}
        :return:
        """

        # 1) Apply retriever(with optional filters) to get fast candidate documents
        documents = self.retriever.retrieve(question, filters=filters, top_k=top_k_retriever)

        if len(documents) == 0:
            logger.info("Retriever did not return any documents. Skipping reader ...")
            results = {"question": question, "answers": []}
            return results

        # 2) Apply reader to get granular answer(s)
        len_chars = sum([len(d.text) for d in documents])
        logger.info(f"Reader is looking for detailed answer in {len_chars} chars ...")
        results = self.reader.predict(question=question,
                                      documents=documents,
                                      top_k=top_k_reader)

        # Add corresponding document_name and more meta data, if an answer contains the document_id
        for ans in results["answers"]:
            ans["meta"] = {}
            for doc in documents:
                if doc.id == ans["document_id"]:
                    ans["meta"] = doc.meta

        return results

    def get_answers_via_similar_questions(self, question: str, top_k_retriever: int = 10, filters: dict = None):
        """
        Get top k answers for a given question using only a retriever.

        :param question: the question string
        :param top_k_retriever: number of text units to be retrieved
        :param filters: limit scope to documents having the given tags and their corresponding values.
            The format for the dict is {"tag-1": "value-1", "tag-2": "value-2" ...}
        :return:
        """

        results = {"question": question, "answers": []}

        # 1) Optional: reduce the search space via document tags
        if filters:
            logging.info(f"Apply filters: {filters}")
            candidate_doc_ids = self.retriever.document_store.get_document_ids_by_tags(filters)
            logger.info(f"Got candidate IDs due to filters:  {candidate_doc_ids}")

            if len(candidate_doc_ids) == 0:
                # We didn't find any doc matching the filters
                return results

        else:
            candidate_doc_ids = None

        # 2) Apply retriever to match similar questions via cosine similarity of embeddings
        documents = self.retriever.retrieve(question, top_k=top_k_retriever, candidate_doc_ids=candidate_doc_ids)

        # 3) Format response
        for doc in documents:
            #TODO proper calibratation of pseudo probabilities
            cur_answer = {"question": doc.question, "answer": doc.text, "context": doc.text,
                          "score": doc.query_score, "offset_start": 0, "offset_end": len(doc.text),
                          }
            if self.retriever.embedding_model:
                probability = (doc.query_score + 1) / 2
            else:
                probability = float(expit(np.asarray(doc.query_score / 8)))
            cur_answer["probability"] = probability
            results["answers"].append(cur_answer)

        return results

    def eval(self, label_index: str = "feedback", doc_index: str = "eval_document", label_origin: str = "gold_label",
             top_k_retriever: int = 10, top_k_reader: int = 10):
        """
        Evaluation of the whole pipeline by first evaluating the Retriever and then evaluating the Reader on the result
        of the Retriever.

        Returns a dict containing the following metrics:
            - "retriever_recall": Proportion of questions for which correct document is among retrieved documents
            - "retriever_map": Mean of average precision for each question. Rewards retrievers that give relevant
              documents a higher rank.
            - "reader_recall": Proportion of predicted answers that overlap with correct answer
            - "reader_map": Reader re-ranks documents retrieved by Retriever. Mean average precision of the re-ranking.
            - "reader_em": Proportion of exact matches of predicted answers with their corresponding correct answers
            - "reader_f1": Average overlap between predicted answers and their corresponding correct answers
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
        finder_start_time = time.time()
        # extract all questions for evaluation
        filter = {"origin": label_origin}
        questions = self.retriever.document_store.get_all_docs_in_index(index=label_index, filters=filter)

        correct_retrievals = 0
        correct_readings = 0
        summed_avg_precision_retriever = 0
        summed_avg_precision_reader = 0
        exact_matches = 0
        summed_f1 = 0
        retrieve_times = []
        read_times = []

        # retrieve documents
        questions_with_docs = []
        retriever_start_time = time.time()
        for q_idx, question in enumerate(questions):
            question_string = question["_source"]["question"]
            single_retrieve_start = time.time()
            retrieved_docs = self.retriever.retrieve(question_string, top_k=top_k_retriever, index=doc_index)
            retrieve_times.append(time.time() - single_retrieve_start)
            for doc_idx, doc in enumerate(retrieved_docs):
                # check if correct doc among retrieved docs
                if doc.meta["doc_id"] == question["_source"]["doc_id"]:
                    correct_retrievals += 1
                    summed_avg_precision_retriever += 1 / (doc_idx + 1)
                    questions_with_docs.append({
                        "question": question,
                        "docs": retrieved_docs,
                        "correct_es_doc_id": doc.id})
                    break
        retriever_total_time = time.time() - retriever_start_time
        number_of_questions = q_idx + 1

        previous_top_k_per_candidate = self.reader.top_k_per_candidate
        previous_return_no_answers = self.reader.return_no_answers
        self.reader.top_k_per_candidate = 1
        self.reader.return_no_answers = True
        # extract answers
        reader_start_time = time.time()
        for q_idx, question in enumerate(questions_with_docs):
            question_string = question["question"]["_source"]["question"]
            docs = question["docs"]
            single_reader_start = time.time()
            predicted_answers = self.reader.predict(question_string, docs, top_k_reader)
            read_times.append(time.time() - single_reader_start)
            # check if question is answerable
            if question["question"]["_source"]["answers"]:
                for answer_idx, answer in enumerate(predicted_answers["answers"]):
                    found_answer = False
                    found_em = False
                    best_f1 = 0
                    # check if correct document
                    if answer["document_id"] == question["correct_es_doc_id"]:
                        summed_avg_precision_reader += 1 / (answer_idx + 1)
                        gold_spans = [(gold_answer["answer_start"], gold_answer["answer_start"] + len(gold_answer["text"]) + 1)
                                      for gold_answer in question["question"]["_source"]["answers"]]
                        predicted_span = (answer["offset_start_in_doc"], answer["offset_end_in_doc"])

                        for gold_span in gold_spans:
                            # check if overlap between gold answer and predicted answer
                            if not found_answer:
                                if (gold_span[0] <= predicted_span[1]) and (predicted_span[0] <= gold_span[1]):
                                    correct_readings += 1
                                    found_answer = True
                            # check for exact match
                            if not found_em:
                                if (gold_span[0] == predicted_span[0]) and (gold_span[1] == predicted_span[1]):
                                    exact_matches += 1
                                    found_em = True
                            # calculate f1
                            pred_indices = list(range(predicted_span[0], predicted_span[1] + 1))
                            gold_indices = list(range(gold_span[0], gold_span[1] + 1))
                            n_overlap = len([x for x in pred_indices if x in gold_indices])
                            if pred_indices and gold_indices and n_overlap:
                                precision = n_overlap / len(pred_indices)
                                recall = n_overlap / len(gold_indices)
                                current_f1 = (2 * precision * recall) / (precision + recall)
                                if current_f1 > best_f1:
                                    best_f1 = current_f1
                        summed_f1 += best_f1

                    if found_answer and found_em:
                        break
            # question not answerable
            else:
                # As question is not answerable, it is not clear how to compute average precision for this question.
                # For now, we decided to calculate average precision based on the rank of 'no answer'.
                for answer_idx, answer in enumerate(predicted_answers["answers"]):
                    # check if 'no answer'
                    if answer["answer"] is None:
                        correct_readings += 1
                        summed_avg_precision_reader += 1 / (answer_idx + 1)
                        exact_matches += 1
                        summed_f1 += 1
                        break
        reader_total_time = time.time() - reader_start_time
        finder_total_time = time.time() - finder_start_time

        retriever_recall = correct_retrievals / number_of_questions
        retriever_map = summed_avg_precision_retriever / number_of_questions
        reader_recall = correct_readings / correct_retrievals
        reader_map = summed_avg_precision_reader / correct_retrievals
        reader_em = exact_matches / correct_retrievals
        reader_f1 = summed_f1 / correct_retrievals

        self.reader.top_k_per_candidate = previous_top_k_per_candidate
        self.reader.return_no_answers = previous_return_no_answers

        logger.info((f"{correct_readings} out of {number_of_questions} questions were correctly answered "
                     f"({(correct_readings/number_of_questions):.2%})."))
        logger.info(f"{number_of_questions-correct_retrievals} questions could not be answered due to the retriever.")
        logger.info(f"{correct_retrievals-correct_readings} questions could not be answered due to the reader.")

        results = {
            "retriever_recall": retriever_recall,
            "retriever_map": retriever_map,
            "reader_recall": reader_recall,
            "reader_map": reader_map,
            "reader_em": reader_em,
            "reader_f1" : reader_f1,
            "total_retrieve_time": retriever_total_time,
            "avg_retrieve_time": mean(retrieve_times),
            "total_reader_time": reader_total_time,
            "avg_reader_time": mean(read_times),
            "total_finder_time": finder_total_time
        }

        return results