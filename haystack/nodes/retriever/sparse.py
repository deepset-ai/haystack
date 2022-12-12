# mypy: disable-error-code=override
from typing import Dict, List, Optional, Union

import logging
from collections import OrderedDict, namedtuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from haystack.schema import Document
from haystack.document_stores.base import BaseDocumentStore, FilterType
from haystack.document_stores import KeywordDocumentStore
from haystack.nodes.retriever import BaseRetriever
from haystack.errors import DocumentStoreError


logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        document_store: Optional[KeywordDocumentStore] = None,
        top_k: int = 10,
        all_terms_must_match: bool = False,
        custom_query: Optional[str] = None,
        scale_score: bool = True,
    ):
        """
        :param document_store: an instance of one of the following DocumentStores to retrieve from: ElasticsearchDocumentStore, OpenSearchDocumentStore and OpenDistroElasticsearchDocumentStore.
            If None, a document store must be passed to the retrieve method for this Retriever to work.
        :param all_terms_must_match: Whether all terms of the query must match the document.
                                     If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
                                     Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
                                     Defaults to False.
        :param custom_query: query string as per Elasticsearch DSL with a mandatory query placeholder(query).

                             Optionally, ES `filter` clause can be added where the values of `terms` are placeholders
                             that get substituted during runtime. The placeholder(${filter_name_1}, ${filter_name_2}..)
                             names must match with the filters dict supplied in self.retrieve().

                                **An example custom_query:**

                                ```python
                                {
                                    "size": 10,
                                    "query": {
                                        "bool": {
                                            "should": [{"multi_match": {
                                                "query": ${query},                 // mandatory query placeholder
                                                "type": "most_fields",
                                                "fields": ["content", "title"]}}],
                                            "filter": [                                 // optional custom filters
                                                {"terms": {"year": ${years}}},
                                                {"terms": {"quarter": ${quarters}}},
                                                {"range": {"date": {"gte": ${date}}}}
                                                ],
                                        }
                                    },
                                }
                                ```

                            **For this custom_query, a sample retrieve() could be:**

                            ```python
                            self.retrieve(query="Why did the revenue increase?",
                                          filters={"years": ["2019"], "quarters": ["Q1", "Q2"]})
                            ```

                             Optionally, highlighting can be defined by specifying Elasticsearch's highlight settings.
                             See https://www.elastic.co/guide/en/elasticsearch/reference/current/highlighting.html.
                             You will find the highlighted output in the returned Document's meta field by key "highlighted".


                                 **Example custom_query with highlighting:**

                                 ```python
                                 {
                                     "size": 10,
                                     "query": {
                                         "bool": {
                                             "should": [{"multi_match": {
                                                 "query": ${query},                 // mandatory query placeholder
                                                 "type": "most_fields",
                                                 "fields": ["content", "title"]}}],
                                         }
                                     },
                                     "highlight": {             // enable highlighting
                                         "fields": {            // for fields content and title
                                             "content": {},
                                             "title": {}
                                         }
                                     },
                                 }
                                 ```

                                 **For this custom_query, highlighting info can be accessed by:**
                                ```python
                                docs = self.retrieve(query="Why did the revenue increase?")
                                highlighted_content = docs[0].meta["highlighted"]["content"]
                                highlighted_title = docs[0].meta["highlighted"]["title"]
                                ```

        :param top_k: How many documents to return per query.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

        """
        super().__init__()
        self.document_store: Optional[KeywordDocumentStore] = document_store
        self.top_k = top_k
        self.custom_query = custom_query
        self.all_terms_must_match = all_terms_must_match
        self.scale_score = scale_score

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterType] = None,
        top_k: Optional[int] = None,
        all_terms_must_match: Optional[bool] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:

                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param all_terms_must_match: Whether all terms of the query must match the document.
                                     When set to `True`, the Retriever returns only documents that contain all query terms (that means the AND operator is being used implicitly between query terms. For example, the query "cozy fish restaurant" is read as "cozy AND fish AND restaurant").
                                     When set to `False`, the Retriever returns documents containing at least one query term (this means the OR operator is being used implicitly between query terms. For example, the query "cozy fish restaurant" is read as "cozy OR fish OR restaurant").
                                     Defaults to `None`. If you set a value for this parameter, it overwrites self.all_terms_must_match at runtime.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                                           If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                                           Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
            )
        if not isinstance(document_store, KeywordDocumentStore):
            raise ValueError("document_store must be a subclass of KeywordDocumentStore.")

        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        if all_terms_must_match is None:
            all_terms_must_match = self.all_terms_must_match

        documents = document_store.query(
            query=query,
            filters=filters,
            top_k=top_k,
            all_terms_must_match=all_terms_must_match,
            custom_query=self.custom_query,
            index=index,
            headers=headers,
            scale_score=scale_score,
        )
        return documents

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: Optional[int] = None,
        all_terms_must_match: Optional[bool] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the supplied queries.

        Returns a list of lists of Documents (one per query).

        :param queries: List of query strings.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:

                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param all_terms_must_match: Whether all terms of the query must match the document.
                                     When set to `True`, the Retriever returns only documents that contain all query terms (that means the AND operator is being used implicitly between query terms. For example, the query "cozy fish restaurant" is read as "cozy AND fish AND restaurant").
                                     When set to `False`, the Retriever returns documents containing at least one query term (this means the OR operator is being used implicitly between query terms. For example, the query "cozy fish restaurant" is read as "cozy OR fish OR restaurant").).
                                     Defaults to `None`. If you set a value for this parameter, it overwrites self.all_terms_must_match at runtime.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :param batch_size: Not applicable.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true similarity scores (e.g. cosine or dot_product) which naturally have a different
                            value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve_batch() method."
            )
        if not isinstance(document_store, KeywordDocumentStore):
            raise ValueError("document_store must be a subclass of KeywordDocumentStore.")

        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        if all_terms_must_match is None:
            all_terms_must_match = self.all_terms_must_match

        documents = document_store.query_batch(
            queries=queries,
            filters=filters,
            top_k=top_k,
            all_terms_must_match=all_terms_must_match,
            custom_query=self.custom_query,
            index=index,
            headers=headers,
            scale_score=scale_score,
        )
        return documents


class ElasticsearchRetriever(BM25Retriever):
    def __init__(
        self,
        document_store: Optional[KeywordDocumentStore] = None,
        top_k: int = 10,
        all_terms_must_match: bool = False,
        custom_query: Optional[str] = None,
    ):
        logger.warn("This class is now deprecated. Please use the BM25Retriever instead")
        super().__init__(document_store, top_k, all_terms_must_match, custom_query)


class FilterRetriever(BM25Retriever):
    """
    Naive "Retriever" that returns all documents that match the given filters. No impact of query at all.
    Helpful for benchmarking, testing and if you want to do QA on small documents without an "active" retriever.
    """

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterType] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: Has no effect, can pass in empty string
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: Has no effect, pass in any int or None
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                                           If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                                           Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
            )
        if index is None:
            index = document_store.index
        documents = document_store.get_all_documents(filters=filters, index=index, headers=headers)
        return documents


class ElasticsearchFilterOnlyRetriever(FilterRetriever):
    def __init__(
        self,
        document_store: Optional[KeywordDocumentStore] = None,
        top_k: int = 10,
        all_terms_must_match: bool = False,
        custom_query: Optional[str] = None,
    ):
        logger.warn("This class is now deprecated. Please use the FilterRetriever instead")
        super().__init__(document_store, top_k, all_terms_must_match, custom_query)


# TODO make Paragraph generic for configurable units of text eg, pages, paragraphs, or split by a char_limit
Paragraph = namedtuple("Paragraph", ["paragraph_id", "document_id", "content", "meta"])


class TfidfRetriever(BaseRetriever):
    """
    Read all documents from a SQL backend.

    Split documents into smaller units (eg, paragraphs or pages) to reduce the
    computations when text is passed on to a Reader for QA.

    It uses sklearn's TfidfVectorizer to compute a tf-idf matrix.
    """

    def __init__(self, document_store: Optional[BaseDocumentStore] = None, top_k: int = 10, auto_fit=True):
        """
        :param document_store: an instance of a DocumentStore to retrieve documents from.
        :param top_k: How many documents to return per query.
        :param auto_fit: Whether to automatically update tf-idf matrix by calling fit() after new documents have been added
        """
        super().__init__()

        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words=None, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1)
        )
        self.document_store = document_store
        self.top_k = top_k
        self.auto_fit = auto_fit
        self.document_count = 0
        if document_store and document_store.get_document_count():
            self.fit(document_store=document_store)

    def _get_all_paragraphs(self, document_store: BaseDocumentStore) -> List[Paragraph]:
        """
        Split the list of documents in paragraphs
        """
        documents = document_store.get_all_documents()

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
        logger.info("Found %s candidate paragraphs from %s docs in DB", len(paragraphs), len(documents))
        return paragraphs

    def _calc_scores(self, queries: Union[str, List[str]]) -> List[Dict[int, float]]:
        if isinstance(queries, str):
            queries = [queries]
        question_vector = self.vectorizer.transform(queries)
        doc_scores_per_query = self.tfidf_matrix.dot(question_vector.T).T.toarray()
        doc_scores_per_query = [
            [(doc_idx, doc_score) for doc_idx, doc_score in enumerate(doc_scores)]
            for doc_scores in doc_scores_per_query
        ]
        indices_and_scores: List[Dict] = [
            OrderedDict(sorted(query_idx_scores, key=lambda tup: tup[1], reverse=True))
            for query_idx_scores in doc_scores_per_query
        ]
        return indices_and_scores

    def retrieve(
        self,
        query: str,
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                                           If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                                           Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.

        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        if document_store is None:
            document_store = self.document_store
            if document_store is None:
                raise ValueError(
                    "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
                )
        else:
            self.fit(document_store=document_store)

        if self.auto_fit:
            if document_store.get_document_count(headers=headers) != self.document_count:
                # run fit() to update self.df, self.tfidf_matrix and self.document_count
                logger.warning(
                    "Indexed documents have been updated and fit() method needs to be run before retrieval. Running it now."
                )
                self.fit(document_store=document_store)
        if self.df is None:
            raise DocumentStoreError(
                "Retrieval requires dataframe df and tf-idf matrix but fit() did not calculate them probably due to an empty document store."
            )

        if filters:
            raise NotImplementedError("Filters are not implemented in TfidfRetriever.")
        if index:
            raise NotImplementedError("Switching index is not supported in TfidfRetriever.")
        if scale_score:
            raise NotImplementedError("Scaling score to the unit interval is not supported in TfidfRetriever.")

        if top_k is None:
            top_k = self.top_k
        # get scores
        indices_and_scores = self._calc_scores(query)

        # rank paragraphs
        df_sliced = self.df.loc[indices_and_scores[0].keys()]
        df_sliced = df_sliced[:top_k]

        logger.debug(
            "Identified %s candidates via retriever:\n%s",
            df_sliced.shape[0],
            df_sliced.to_string(col_space=10, index=False),
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

    def retrieve_batch(
        self,
        queries: Union[str, List[str]],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the supplied queries.

        Returns a list of lists of Documents (one per query).

        :param queries: Single query string or list of queries.
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param batch_size: Not applicable.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true similarity scores (e.g. cosine or dot_product) which naturally have a different
                            value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        if document_store is None:
            document_store = self.document_store
            if document_store is None:
                raise ValueError(
                    "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
                )
        else:
            self.fit(document_store=document_store)

        if self.auto_fit:
            if document_store.get_document_count(headers=headers) != self.document_count:
                # run fit() to update self.df, self.tfidf_matrix and self.document_count
                logger.warning(
                    "Indexed documents have been updated and fit() method needs to be run before retrieval. Running it now."
                )
                self.fit(document_store=document_store)
        if self.df is None:
            raise DocumentStoreError(
                "Retrieval requires dataframe df and tf-idf matrix but fit() did not calculate them probably due to an empty document store."
            )

        if filters:
            raise NotImplementedError("Filters are not implemented in TfidfRetriever.")
        if index:
            raise NotImplementedError("Switching index is not supported in TfidfRetriever.")
        if scale_score:
            raise NotImplementedError("Scaling score to the unit interval is not supported in TfidfRetriever.")

        if top_k is None:
            top_k = self.top_k

        indices_and_scores = self._calc_scores(queries)
        all_documents = []
        for query_result in indices_and_scores:
            df_sliced = self.df.loc[query_result.keys()]
            df_sliced = df_sliced[:top_k]
            logger.debug(
                "Identified %s candidates via retriever:\n%s",
                df_sliced.shape[0],
                df_sliced.to_string(col_space=10, index=False),
            )

            # get actual content for the top candidates
            paragraphs = list(df_sliced.content.values)
            meta_data = [
                {"document_id": row["document_id"], "paragraph_id": row["paragraph_id"], "meta": row.get("meta", {})}
                for idx, row in df_sliced.iterrows()
            ]
            cur_documents = []
            for para, meta in zip(paragraphs, meta_data):
                cur_documents.append(Document(id=meta["document_id"], content=para, meta=meta.get("meta", {})))
            all_documents.append(cur_documents)

        return all_documents

    def fit(self, document_store: BaseDocumentStore):
        """
        Performing training on this class according to the TF-IDF algorithm.
        """
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the fit() method."
            )
        paragraphs = self._get_all_paragraphs(document_store=document_store)
        if not paragraphs or len(paragraphs) == 0:
            raise DocumentStoreError("Fit method called with empty document store")

        self.df = pd.DataFrame.from_dict(paragraphs)
        self.df["content"] = self.df["content"].apply(lambda x: " ".join(x))  # pylint: disable=unnecessary-lambda
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["content"])
        self.document_count = document_store.get_document_count()
