import logging
from collections import OrderedDict
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from haystack.document_store.base import BaseDocumentStore
from haystack import Document
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.base import BaseRetriever
from collections import namedtuple

logger = logging.getLogger(__name__)


class ElasticsearchRetriever(BaseRetriever):
    def __init__(self, document_store: ElasticsearchDocumentStore, custom_query: str = None):
        """
        :param document_store: an instance of a DocumentStore to retrieve documents from.
        :param custom_query: query string as per Elasticsearch DSL with a mandatory query placeholder(query).

                             Optionally, ES `filter` clause can be added where the values of `terms` are placeholders
                             that get substituted during runtime. The placeholder(${filter_name_1}, ${filter_name_2}..)
                             names must match with the filters dict supplied in self.retrieve().
                             ::

                                 **An example custom_query:**
                                 ```python
                                |    {
                                |        "size": 10,
                                |        "query": {
                                |            "bool": {
                                |                "should": [{"multi_match": {
                                |                    "query": "${query}",                 // mandatory query placeholder
                                |                    "type": "most_fields",
                                |                    "fields": ["text", "title"]}}],
                                |                "filter": [                                 // optional custom filters
                                |                    {"terms": {"year": "${years}"}},
                                |                    {"terms": {"quarter": "${quarters}"}},
                                |                    {"range": {"date": {"gte": "${date}"}}}
                                |                    ],
                                |            }
                                |        },
                                |    }
                                 ```

                             **For this custom_query, a sample retrieve() could be:**
                             ```python
                            |    self.retrieve(query="Why did the revenue increase?",
                            |                  filters={"years": ["2019"], "quarters": ["Q1", "Q2"]})
                            ```
        """
        self.document_store: ElasticsearchDocumentStore = document_store
        self.custom_query = custom_query

    def retrieve(self, query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        if index is None:
            index = self.document_store.index

        documents = self.document_store.query(query, filters, top_k, self.custom_query, index)
        return documents


class ElasticsearchFilterOnlyRetriever(ElasticsearchRetriever):
    """
    Naive "Retriever" that returns all documents that match the given filters. No impact of query at all.
    Helpful for benchmarking, testing and if you want to do QA on small documents without an "active" retriever.
    """

    def retrieve(self, query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        if index is None:
            index = self.document_store.index
        documents = self.document_store.query(query=None, filters=filters, top_k=top_k,
                                              custom_query=self.custom_query, index=index)
        return documents

# TODO make Paragraph generic for configurable units of text eg, pages, paragraphs, or split by a char_limit
Paragraph = namedtuple("Paragraph", ["paragraph_id", "document_id", "text", "meta"])


class TfidfRetriever(BaseRetriever):
    """
    Read all documents from a SQL backend.

    Split documents into smaller units (eg, paragraphs or pages) to reduce the
    computations when text is passed on to a Reader for QA.

    It uses sklearn's TfidfVectorizer to compute a tf-idf matrix.
    """

    def __init__(self, document_store: BaseDocumentStore):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 1),
        )

        self.document_store = document_store
        self.paragraphs = self._get_all_paragraphs()
        self.df = None
        self.fit()

    def _get_all_paragraphs(self) -> List[Paragraph]:
        """
        Split the list of documents in paragraphs
        """
        documents = self.document_store.get_all_documents()

        paragraphs = []
        p_id = 0
        for doc in documents:
            for p in doc.text.split("\n\n"):  # TODO: this assumes paragraphs are separated by "\n\n". Can be switched to paragraph tokenizer.
                if not p.strip():  # skip empty paragraphs
                    continue
                paragraphs.append(
                    Paragraph(document_id=doc.id, paragraph_id=p_id, text=(p,), meta=doc.meta)
                )
                p_id += 1
        logger.info(f"Found {len(paragraphs)} candidate paragraphs from {len(documents)} docs in DB")
        return paragraphs

    def _calc_scores(self, query: str) -> dict:
        question_vector = self.vectorizer.transform([query])

        scores = self.tfidf_matrix.dot(question_vector.T).toarray()
        idx_scores = [(idx, score) for idx, score in enumerate(scores)]
        indices_and_scores = OrderedDict(
            sorted(idx_scores, key=(lambda tup: tup[1]), reverse=True)
        )
        return indices_and_scores

    def retrieve(self, query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        if self.df is None:
            raise Exception("fit() needs to called before retrieve()")

        if filters:
            raise NotImplementedError("Filters are not implemented in TfidfRetriever.")
        if index:
            raise NotImplementedError("Switching index is not supported in TfidfRetriever.")

        # get scores
        indices_and_scores = self._calc_scores(query)

        # rank paragraphs
        df_sliced = self.df.loc[indices_and_scores.keys()]
        df_sliced = df_sliced[:top_k]

        logger.debug(
            f"Identified {df_sliced.shape[0]} candidates via retriever:\n {df_sliced.to_string(col_space=10, index=False)}"
        )

        # get actual content for the top candidates
        paragraphs = list(df_sliced.text.values)
        meta_data = [{"document_id": row["document_id"], "paragraph_id": row["paragraph_id"],  "meta": row.get("meta", {})}
                     for idx, row in df_sliced.iterrows()]

        documents = []
        for para, meta in zip(paragraphs, meta_data):
            documents.append(
                Document(
                    id=meta["document_id"],
                    text=para,
                    meta=meta.get("meta", {})
                ))

        return documents

    def fit(self):
        """
        Performing training on this class according to the TF-IDF algorithm.
        """
        if not self.paragraphs or len(self.paragraphs) == 0:
            self.paragraphs = self._get_all_paragraphs()
            if not self.paragraphs or len(self.paragraphs) == 0:
                logger.warning("Fit method called with empty document store")
                return

        self.df = pd.DataFrame.from_dict(self.paragraphs)
        self.df["text"] = self.df["text"].apply(lambda x: " ".join(x))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["text"])