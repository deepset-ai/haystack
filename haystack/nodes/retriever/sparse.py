from typing import Dict, List, Optional

import logging
import pandas as pd
from collections import OrderedDict, namedtuple
from sklearn.feature_extraction.text import TfidfVectorizer

from haystack.schema import Document
from haystack.document_stores import BaseDocumentStore, KeywordDocumentStore
from haystack.nodes.retriever import BaseRetriever

from haystack.document_stores import BaseDocumentStore


logger = logging.getLogger(__name__)


class ElasticsearchRetriever(BaseRetriever):
    def __init__(self, document_store: KeywordDocumentStore, top_k: int = 10, custom_query: str = None):
        """
        :param document_store: an instance of an ElasticsearchDocumentStore to retrieve documents from.
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
                                |                    "query": ${query},                 // mandatory query placeholder
                                |                    "type": "most_fields",
                                |                    "fields": ["content", "title"]}}],
                                |                "filter": [                                 // optional custom filters
                                |                    {"terms": {"year": ${years}}},
                                |                    {"terms": {"quarter": ${quarters}}},
                                |                    {"range": {"date": {"gte": ${date}}}}
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

                             Optionally, highlighting can be defined by specifying Elasticsearch's highlight settings.
                             See https://www.elastic.co/guide/en/elasticsearch/reference/current/highlighting.html.
                             You will find the highlighted output in the returned Document's meta field by key "highlighted".
                             ::

                                 **Example custom_query with highlighting:**
                                 ```python
                                |    {
                                |        "size": 10,
                                |        "query": {
                                |            "bool": {
                                |                "should": [{"multi_match": {
                                |                    "query": ${query},                 // mandatory query placeholder
                                |                    "type": "most_fields",
                                |                    "fields": ["content", "title"]}}],
                                |            }
                                |        },
                                |        "highlight": {             // enable highlighting
                                |            "fields": {            // for fields content and title
                                |                "content": {},
                                |                "title": {}
                                |            }
                                |        },
                                |    }
                                 ```

                                 **For this custom_query, highlighting info can be accessed by:**
                                ```python
                                |    docs = self.retrieve(query="Why did the revenue increase?")
                                |    highlighted_content = docs[0].meta["highlighted"]["content"]
                                |    highlighted_title = docs[0].meta["highlighted"]["title"]
                                ```

        :param top_k: How many documents to return per query.
        """
        # save init parameters to enable export of component config as YAML
        self.set_config(document_store=document_store, top_k=top_k, custom_query=custom_query)
        self.document_store: KeywordDocumentStore = document_store
        self.top_k = top_k
        self.custom_query = custom_query

    def retrieve(
        self,
        query: str,
        filters: dict = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = self.document_store.index

        documents = self.document_store.query(query, filters, top_k, self.custom_query, index, headers=headers)
        return documents


class ElasticsearchFilterOnlyRetriever(ElasticsearchRetriever):
    """
    Naive "Retriever" that returns all documents that match the given filters. No impact of query at all.
    Helpful for benchmarking, testing and if you want to do QA on small documents without an "active" retriever.
    """

    def retrieve(
        self,
        query: str,
        filters: dict = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = self.document_store.index
        documents = self.document_store.query(
            query=None, filters=filters, top_k=top_k, custom_query=self.custom_query, index=index, headers=headers
        )
        return documents


# TODO make Paragraph generic for configurable units of text eg, pages, paragraphs, or split by a char_limit
Paragraph = namedtuple("Paragraph", ["paragraph_id", "document_id", "content", "meta"])


class TfidfRetriever(BaseRetriever):
    """
    Read all documents from a SQL backend.

    Split documents into smaller units (eg, paragraphs or pages) to reduce the
    computations when text is passed on to a Reader for QA.

    It uses sklearn's TfidfVectorizer to compute a tf-idf matrix.
    """

    def __init__(self, document_store: BaseDocumentStore, top_k: int = 10, auto_fit=True):
        """
        :param document_store: an instance of a DocumentStore to retrieve documents from.
        :param top_k: How many documents to return per query.
        :param auto_fit: Whether to automatically update tf-idf matrix by calling fit() after new documents have been added
        """
        # save init parameters to enable export of component config as YAML
        self.set_config(document_store=document_store, top_k=top_k, auto_fit=auto_fit)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 1),
        )

        self.document_store = document_store
        self.paragraphs = self._get_all_paragraphs()
        self.df = None
        self.top_k = top_k
        self.auto_fit = auto_fit
        self.document_count = 0
        self.fit()

    def _get_all_paragraphs(self) -> List[Paragraph]:
        """
        Split the list of documents in paragraphs
        """
        documents = self.document_store.get_all_documents()

        paragraphs = []
        p_id = 0
        for doc in documents:
            for p in doc.content.split(
                "\n\n"
            ):  # TODO: this assumes paragraphs are separated by "\n\n". Can be switched to paragraph tokenizer.
                if not p.strip():  # skip empty paragraphs
                    continue
                paragraphs.append(Paragraph(document_id=doc.id, paragraph_id=p_id, content=(p,), meta=doc.meta))
                p_id += 1
        logger.info(f"Found {len(paragraphs)} candidate paragraphs from {len(documents)} docs in DB")
        return paragraphs

    def _calc_scores(self, query: str) -> dict:
        question_vector = self.vectorizer.transform([query])

        scores = self.tfidf_matrix.dot(question_vector.T).toarray()
        idx_scores = [(idx, score) for idx, score in enumerate(scores)]
        indices_and_scores = OrderedDict(sorted(idx_scores, key=(lambda tup: tup[1]), reverse=True))
        return indices_and_scores

    def retrieve(
        self,
        query: str,
        filters: dict = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        if self.auto_fit:
            if self.document_store.get_document_count(headers=headers) != self.document_count:
                # run fit() to update self.df, self.tfidf_matrix and self.document_count
                logger.warning(
                    "Indexed documents have been updated and fit() method needs to be run before retrieval. Running it now."
                )
                self.fit()
        if self.df is None:
            raise Exception(
                "Retrieval requires dataframe df and tf-idf matrix but fit() did not calculate them probably due to an empty document store."
            )

        if filters:
            raise NotImplementedError("Filters are not implemented in TfidfRetriever.")
        if index:
            raise NotImplementedError("Switching index is not supported in TfidfRetriever.")

        if top_k is None:
            top_k = self.top_k
        # get scores
        indices_and_scores = self._calc_scores(query)

        # rank paragraphs
        df_sliced = self.df.loc[indices_and_scores.keys()]
        df_sliced = df_sliced[:top_k]

        logger.debug(
            f"Identified {df_sliced.shape[0]} candidates via retriever:\n {df_sliced.to_string(col_space=10, index=False)}"
        )

        # get actual content for the top candidates
        paragraphs = list(df_sliced.content.values)
        meta_data = [
            {"document_id": row["document_id"], "paragraph_id": row["paragraph_id"], "meta": row.get("meta", {})}
            for idx, row in df_sliced.iterrows()
        ]

        documents = []
        for para, meta in zip(paragraphs, meta_data):
            documents.append(Document(id=meta["document_id"], content=para, meta=meta.get("meta", {})))

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
        self.df["content"] = self.df["content"].apply(lambda x: " ".join(x))  # pylint: disable=unnecessary-lambda
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["content"])
        self.document_count = self.document_store.get_document_count()
