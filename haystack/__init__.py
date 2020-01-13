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

        # 1) Optional: reduce the search space via document tags
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

        # 2) Apply retriever to get fast candidate paragraphs
        paragraphs, meta_data = self.retriever.retrieve(question, top_k=top_k_retriever, candidate_doc_ids=candidate_doc_ids)

        # 3) Apply reader to get granular answer(s)
        logger.info(f"Applying the reader now to look for the answer in detail ...")
        results = self.reader.predict(question=question,
                                      paragrahps=paragraphs,
                                      meta_data_paragraphs=meta_data,
                                      top_k=top_k_reader)

        return results

