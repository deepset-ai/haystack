import logging
logger = logging.getLogger(__name__)


class Finder:
    """
    Finder ties together instances of the Reader and Retriever class.

    It provides an interface to predict top n answers for a given question.
    """

    def __init__(self, reader, retriever):
        self.retriever = retriever
        self.reader = reader

    def get_answers(self, question, top_k_reader=1, top_k_retriever=10, filters=None):
        """
        Get top k answers for a given question.

        :param question: the question string
        :param top_k_reader: number of answers returned by the reader
        :param top_k_retriever: number of text units to be retrieved
        :param filters: limit scope to documents having the given tags and their corresponding values.
            The format for the dict is {"tag-1": "value-1", "tag-2": "value-2" ...}
        :return:
        """

        # 1) Optional: reduce the search space via document tags
        if filters:
            logging.info(f"Apply filters: {filters}")
            candidate_doc_ids = self.retriever.document_store.get_document_ids_by_tags(filters)
            logger.info(f"Got candidate IDs due to filters:  {candidate_doc_ids}")

            if len(candidate_doc_ids) == 0:
                # We didn't find any doc matching the filters
                results = {"question": question, "answers": []}
                return results

        else:
            candidate_doc_ids = None

        # 2) Apply retriever to get fast candidate paragraphs
        paragraphs, meta_data = self.retriever.retrieve(question, top_k=top_k_retriever, candidate_doc_ids=candidate_doc_ids)

        if len(paragraphs) == 0:
            logger.info("Retriever did not return any documents. Skipping reader ...")
            results = {"question": question, "answers": []}
            return results

        # 3) Apply reader to get granular answer(s)
        len_chars = sum([len (p) for p in paragraphs])
        logger.info(f"Reader is looking for detailed answer in {len_chars} chars ...")
        results = self.reader.predict(question=question,
                                      paragraphs=paragraphs,
                                      meta_data_paragraphs=meta_data,
                                      top_k=top_k_reader)
        # Add corresponding document_name if an answer contains the document_id (only supported in FARMReader)
        for ans in results["answers"]:
            ans["document_name"] = None
            for meta in meta_data:
                if meta["document_id"] == ans["document_id"]:
                    ans["document_name"] = meta.get("document_name", None)

        return results

    def get_answers_via_similar_questions(self, question, top_k_retriever=10, filters=None):
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
        paragraphs, meta_data = self.retriever.retrieve(question, top_k=top_k_retriever,
                                                        candidate_doc_ids=candidate_doc_ids)

        # 3) Format response
        for answer, meta in zip(paragraphs, meta_data):
            #TODO proper calibratation of pseudo probabilities
            if self.retriever.embedding_model:
                cur_answer = {"question": meta["question"], "answer": answer, "context": answer, "score": meta["score"],
                              "probability": (meta["score"]+1)/2, "offset_start": 0, "offset_end": len(answer),
                              "meta": meta}
            else:
                cur_answer = {"question": meta["question"], "answer": answer, "context": answer, "score": meta["score"],
                              "probability": meta["score"]/ 10, "offset_start": 0, "offset_end": len(answer), "meta": meta}
            results["answers"].append(cur_answer)

        return results
