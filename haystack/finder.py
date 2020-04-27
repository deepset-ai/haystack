import logging

import numpy as np
from scipy.special import expit

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
