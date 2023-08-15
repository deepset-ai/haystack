import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from haystack.document_stores.filter_utils import LogicalFilterClause
from haystack.document_stores.search_engine import SearchEngineDocumentStore
from haystack.errors import DocumentStoreError
from haystack.schema import Document, FilterType

logger = logging.getLogger(__name__)


class _ElasticsearchDocumentStore(SearchEngineDocumentStore):
    def __init__(
        self,
        client: Any,
        index: str = "document",
        label_index: str = "label",
        search_fields: Union[str, list] = "content",
        content_field: str = "content",
        name_field: str = "name",
        embedding_field: str = "embedding",
        embedding_dim: int = 768,
        custom_mapping: Optional[dict] = None,
        excluded_meta_data: Optional[list] = None,
        analyzer: str = "standard",
        recreate_index: bool = False,
        create_index: bool = True,
        refresh_type: str = "wait_for",
        similarity: str = "dot_product",
        return_embedding: bool = False,
        duplicate_documents: str = "overwrite",
        scroll: str = "1d",
        skip_missing_embeddings: bool = True,
        synonyms: Optional[List] = None,
        synonym_type: str = "synonym",
        batch_size: int = 10_000,
        index_type: str = "exact",
        hnsw_num_candidates: Optional[int] = None,
    ):
        self.index_type = index_type
        self.hnsw_num_candidates = hnsw_num_candidates
        super().__init__(
            client=client,
            index=index,
            label_index=label_index,
            search_fields=search_fields,
            content_field=content_field,
            name_field=name_field,
            embedding_field=embedding_field,
            embedding_dim=embedding_dim,
            custom_mapping=custom_mapping,
            excluded_meta_data=excluded_meta_data,
            analyzer=analyzer,
            recreate_index=recreate_index,
            create_index=create_index,
            refresh_type=refresh_type,
            similarity=similarity,
            return_embedding=return_embedding,
            duplicate_documents=duplicate_documents,
            scroll=scroll,
            skip_missing_embeddings=skip_missing_embeddings,
            synonyms=synonyms,
            synonym_type=synonym_type,
            batch_size=batch_size,
        )

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
    ) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
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
        :param top_k: How many documents to return
        :param index: Index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :return:
        """
        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        if not self.embedding_field:
            raise RuntimeError("Please specify arg `embedding_field` in ElasticsearchDocumentStore()")

        body = self._construct_dense_query_body(
            query_emb=query_emb, filters=filters, top_k=top_k, return_embedding=return_embedding
        )

        try:
            result = self._search(index=index, **body, headers=headers)["hits"]["hits"]
            if len(result) == 0:
                count_documents = self.get_document_count(index=index, headers=headers)
                if count_documents == 0:
                    logger.warning("Index is empty. First add some documents to search them.")
                count_embeddings = self.get_embedding_count(index=index, headers=headers)
                if count_embeddings == 0:
                    logger.warning("No documents with embeddings. Run the document store's update_embeddings() method.")
        # TODO: hnsw does not throw an error since it directly ignores all documents without embedding
        # What should we do? Artifically throw an error? -> Affects test_query_with_filters_and_missing_embeddings
        except self._RequestError as e:
            if e.error == "search_phase_execution_exception":
                error_message: str = (
                    "search_phase_execution_exception: Likely some of your stored documents don't have embeddings. "
                    "Run the document store's update_embeddings() method."
                )
                raise self._RequestError(e.status_code, error_message, e.info)
            raise e

        documents = [
            self._convert_es_hit_to_document(hit, adapt_score_for_embedding=True, scale_score=scale_score)
            for hit in result
        ]
        return documents

    def _construct_dense_query_body(
        self, query_emb: np.ndarray, return_embedding: bool, filters: Optional[FilterType] = None, top_k: int = 10
    ):
        if self.index_type == "hnsw":
            body = self._construct_dense_query_body_ann(query_emb=query_emb, top_k=top_k, filters=filters)
        elif self.index_type == "exact":
            body = self._construct_dense_query_body_knn(query_emb=query_emb, top_k=top_k, filters=filters)
        else:
            raise DocumentStoreError(f"index_type {self.index_type} not supported")
        excluded_fields = self._get_excluded_fields(return_embedding=return_embedding)
        if excluded_fields:
            body["_source"] = {"excludes": excluded_fields}

        return body

    def _construct_dense_query_body_knn(self, query_emb: np.ndarray, top_k: int, filters: Optional[FilterType] = None):
        filter_ = self._construct_filter(filters)
        body = {"size": top_k, "query": {"script_score": self._get_vector_similarity_query(query_emb, top_k=top_k)}}
        body["query"]["script_score"]["query"] = {"bool": {"filter": filter_}}  # type: ignore
        return body

    def _construct_dense_query_body_ann(self, query_emb: np.ndarray, top_k: int, filters: Optional[FilterType] = None):
        filter_ = self._construct_filter(filters)
        body = {
            "knn": {
                "field": self.embedding_field,
                "query_vector": query_emb,
                "num_candidates": self._get_ann_num_candidates(top_k),
                "k": top_k,
                "filter": filter_,
            }
        }
        return body

    def _get_ann_num_candidates(self, top_k: int) -> int:
        if self.hnsw_num_candidates is None:
            return 10 * top_k
        return self.hnsw_num_candidates

    def _construct_filter(self, filters: Optional[FilterType] = None) -> Dict:
        filter_ = []
        if filters:
            filter_.append(LogicalFilterClause.parse(filters).convert_to_elasticsearch())
        if self.skip_missing_embeddings:
            skip_missing_embedding_filter = {"exists": {"field": self.embedding_field}}
            filter_.append(skip_missing_embedding_filter)
        if len(filter_) == 0:
            return {"match_all": {}}
        return {"bool": {"must": filter_}}

    def _create_document_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        """
        Create a new index for storing documents.
        """
        if self.custom_mapping:
            mapping = self.custom_mapping
        else:
            mapping = {
                "mappings": {
                    "properties": {self.name_field: {"type": "keyword"}, self.content_field: {"type": "text"}},
                    "dynamic_templates": [
                        {"strings": {"path_match": "*", "match_mapping_type": "string", "mapping": {"type": "keyword"}}}
                    ],
                },
                "settings": {"analysis": {"analyzer": {"default": {"type": self.analyzer}}}},
            }

            if self.synonyms:
                for field in self.search_fields:
                    mapping["mappings"]["properties"].update({field: {"type": "text", "analyzer": "synonym"}})
                mapping["mappings"]["properties"][self.content_field] = {"type": "text", "analyzer": "synonym"}

                mapping["settings"]["analysis"]["analyzer"]["synonym"] = {
                    "tokenizer": "whitespace",
                    "filter": ["lowercase", "synonym"],
                }
                mapping["settings"]["analysis"]["filter"] = {
                    "synonym": {"type": self.synonym_type, "synonyms": self.synonyms}
                }

            else:
                for field in self.search_fields:
                    mapping["mappings"]["properties"].update({field: {"type": "text"}})

            if self.embedding_field:
                mapping["mappings"]["properties"][self.embedding_field] = self._create_embedding_field_mapping()

        try:
            self._index_create(index=index_name, **mapping, headers=headers)
        except self._RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self._index_exists(index_name, headers=headers):
                raise e

    def _create_embedding_field_mapping(self):
        mapping = {"type": "dense_vector", "dims": self.embedding_dim}
        if self.index_type == "exact":
            return mapping
        mapping["index"] = True
        mapping["similarity"] = self._get_similarity_string()
        return mapping

    def _get_similarity_string(self):
        if self.similarity == "dot_product":
            return "dot_product"
        elif self.similarity == "cosine":
            return "cosine"
        elif self.similarity == "l2":
            return "l2_norm"
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity}")

    def _create_label_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        mapping = {
            "mappings": {
                "properties": {
                    "query": {"type": "text"},
                    "answer": {"type": "nested"},
                    "document": {"type": "nested"},
                    "is_correct_answer": {"type": "boolean"},
                    "is_correct_document": {"type": "boolean"},
                    "origin": {"type": "keyword"},  # e.g. user-feedback or gold-label
                    "document_id": {"type": "keyword"},
                    "no_answer": {"type": "boolean"},
                    "pipeline_id": {"type": "keyword"},
                    "created_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"},
                    "updated_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"}
                    # TODO add pipeline_hash and pipeline_name once we migrated the REST API to pipelines
                }
            }
        }
        try:
            self._index_create(index=index_name, **mapping, headers=headers)
        except self._RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self._index_exists(index_name, headers=headers):
                raise e

    def _validate_and_adjust_document_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        """
        Validates an existing document index. If there's no embedding field, we'll add it.
        """
        indices = self._index_get(index=index_name, headers=headers)

        if not any(indices):
            logger.warning(
                "To use an index, you must create it first. The index called '%s' doesn't exist. "
                "You can create it by setting `create_index=True` on init or by calling `write_documents()` if you prefer to create it on demand. "
                "Note that this instance doesn't validate the index after you create it.",
                index_name,
            )

        # If the index name is an alias that groups multiple existing indices, each of them must have an embedding_field.
        for index_id, index_info in indices.items():
            mapping = index_info["mappings"]
            if self.search_fields:
                for search_field in self.search_fields:
                    if search_field in mapping["properties"]:
                        if mapping["properties"][search_field]["type"] != "text":
                            raise DocumentStoreError(
                                f"Remove '{search_field}' from `search_fields` or use another index if you want to query it using full text search.  "
                                f"'{search_field}' of index '{index_id}' has type '{mapping['properties'][search_field]['type']}' but needs 'text' for full text search."
                                f"This error might occur if you are trying to use Haystack 1.0 and above with an existing Elasticsearch index created with a previous version of Haystack. "
                                f"Recreating the index with `recreate_index=True` will fix your environment. "
                                f"Note that you'll lose all data stored in the index."
                            )
                    else:
                        mapping["properties"][search_field] = (
                            {"type": "text", "analyzer": "synonym"} if self.synonyms else {"type": "text"}
                        )

            if self.embedding_field:
                if (
                    self.embedding_field in mapping["properties"]
                    and mapping["properties"][self.embedding_field]["type"] != "dense_vector"
                ):
                    raise DocumentStoreError(
                        f"Update the document store to use a different name for the `embedding_field` parameter. "
                        f"The index '{index_id}' in Elasticsearch already has a field called '{self.embedding_field}' "
                        f"of type '{mapping['properties'][self.embedding_field]['type']}'."
                    )
                request_mapping = self._create_embedding_field_mapping()
                embedding_mapping_exists = self.embedding_field in mapping["properties"]
                if embedding_mapping_exists and (
                    mapping["properties"][self.embedding_field].keys() != request_mapping.keys()
                ):
                    raise DocumentStoreError(
                        "The mapping of the existing embedding field is not compatible with the requested mapping."
                    )
                mapping["properties"][self.embedding_field] = request_mapping
                self._index_put_mapping(index=index_id, body=mapping, headers=headers)

    def _validate_server_version(self, expected_version: int):
        """
        Validate that the Elasticsearch server version is compatible with the used ElasticsearchDocumentStore.
        """
        if self.server_version[0] != expected_version:
            logger.warning(
                "This ElasticsearchDocumentStore has been built for Elasticsearch %s, but the detected version of the "
                "Elasticsearch server is %s. Unexpected behaviors or errors may occur due to version incompatibility.",
                expected_version,
                ".".join(map(str, self.server_version)),
            )

    def _get_vector_similarity_query(self, query_emb: np.ndarray, top_k: int) -> Dict[str, Any]:
        """
        Generate Elasticsearch query for vector similarity.
        """
        if self.similarity == "cosine":
            similarity_fn_name = "cosineSimilarity"
        elif self.similarity == "dot_product":
            similarity_fn_name = "dotProduct"
        elif self.similarity == "l2":
            similarity_fn_name = "l2norm"
        else:
            raise DocumentStoreError(
                "Invalid value for similarity in ElasticSearchDocumentStore constructor. Choose between 'cosine', 'dot_product' and 'l2'"
            )

        # Elasticsearch 7.6 introduced a breaking change regarding the vector function signatures:
        # https://www.elastic.co/guide/en/elasticsearch/reference/7.6/breaking-changes-7.6.html#_update_to_vector_function_signatures
        if self.server_version[0] == 7 and self.server_version[1] < 6:
            similarity_script_score = f"{similarity_fn_name}(params.query_vector,doc['{self.embedding_field}']) + 1000"
        else:
            similarity_script_score = f"{similarity_fn_name}(params.query_vector,'{self.embedding_field}') + 1000"

        query = {
            "script": {
                # offset score to ensure a positive range as required by Elasticsearch
                "source": similarity_script_score,
                "params": {"query_vector": query_emb.tolist()},
            }
        }
        return query

    def _get_raw_similarity_score(self, score):
        return score - 1000
