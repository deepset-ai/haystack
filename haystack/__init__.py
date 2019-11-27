from haystack.retriever.tfidf import TfidfRetriever
from haystack.reader.farm import FARMReader
from haystack.database import db
import logging
import farm

import pandas as pd
pd.options.display.max_colwidth = 80

logger = logging.getLogger(__name__)

logging.getLogger('farm').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)


class Finder:
    """
    Finder ties together instances of the Reader and Retriever class.

    It provides an interface to predict top n answers for a given question.
    """

    def __init__(self, reader, retriever):
        self.retriever = retriever
        self.retriever.fit()

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

        if filters:
            query = """
                SELECT id FROM document WHERE id in (
                    SELECT dt.document_id
                    FROM document_tag dt JOIN
                        tag t
                        ON t.id = dt.tag_id
                    GROUP BY dt.document_id
            """
            tag_filters = []
            if filters:
                for tag, value in filters.items():
                    if value:
                        tag_filters.append(
                            f"SUM(CASE WHEN t.value='{value}' THEN 1 ELSE 0 END) > 0"
                        )

            final_query = f"{query} HAVING {' AND '.join(tag_filters)});"
            query_results = db.session.execute(final_query)
            candidate_doc_ids = [row[0] for row in query_results]
        else:
            candidate_doc_ids = None

        retrieved_scores = self.retriever.retrieve(question, top_k=top_k_retriever)

        inference_dicts = self._convert_retrieved_text_to_reader_format(
            retrieved_scores, question, candidate_doc_ids=candidate_doc_ids
        )
        results = self.reader.predict(inference_dicts, top_k=top_k_reader)
        return results["results"]

    def _convert_retrieved_text_to_reader_format(
        self, retrieved_scores, question, candidate_doc_ids=None, verbose=True
    ):
        """
        The reader expect the input as:
        {
            "text": "FARM is a home for all species of pretrained language models (e.g. BERT) that can be adapted to 
            different domain languages or down-stream tasks. With FARM you can easily create SOTA NLP models for tasks 
            like document classification, NER or question answering.",
            "document_id": 127,
            "questions" : ["What can you do with FARM?"]
        }

        :param retrieved_scores: tfidf scores as returned by the retriever
        :param question: question string
        :param verbose: enable verbose logging
        """
        df_sliced = self.retriever.df.loc[retrieved_scores.keys()]
        if verbose:
            logger.info(
                f"Identified {df_sliced.shape[0]} candidates via retriever:\n {df_sliced.to_string(col_space=10, index=False)}"
            )
            logger.info(
                f"Applying the reader now to look for the answer in detail ..."
                )
        inference_dicts = []
        for idx, row in df_sliced.iterrows():
            if candidate_doc_ids and row["document_id"] not in candidate_doc_ids:
                continue
            inference_dicts.append(
                {
                    "text": row["text"],
                    "document_id": row["document_id"],
                    "questions": [question],
                }
            )

        return inference_dicts
