# pylint: disable=too-many-public-methods


from copy import deepcopy
from typing import List, Optional, Union, Dict, Any, Generator
from abc import abstractmethod
import json
import logging
import time
from string import Template

import numpy as np
from scipy.special import expit
from tqdm.auto import tqdm
from pydantic.error_wrappers import ValidationError

from haystack.document_stores import KeywordDocumentStore
from haystack.schema import Document, FilterType, Label
from haystack.document_stores.base import get_batches_from_generator
from haystack.document_stores.filter_utils import LogicalFilterClause
from haystack.errors import DocumentStoreError, HaystackError
from haystack.nodes.retriever import DenseRetriever


logger = logging.getLogger(__name__)


def prepare_hosts(host, port):
    """
    Create a list of host(s) + port(s) to allow direct client connections to multiple nodes,
    in the format expected by the client.
    """
    if isinstance(host, list):
        if isinstance(port, list):
            if not len(port) == len(host):
                raise ValueError("Length of list `host` must match length of list `port`")
            hosts = [{"host": h, "port": p} for h, p in zip(host, port)]
        else:
            hosts = [{"host": h, "port": port} for h in host]
    else:
        hosts = [{"host": host, "port": port}]
    return hosts


class SearchEngineDocumentStore(KeywordDocumentStore):
    """
    Base class implementing the common logic for Elasticsearch and Opensearch
    """

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
    ):
        super().__init__()

        self.client = client
        self._RequestError: Any = Exception

        # configure mappings to ES fields that will be used for querying / displaying results
        if type(search_fields) == str:
            search_fields = [search_fields]

        self.search_fields = search_fields
        self.content_field = content_field
        self.name_field = name_field
        self.embedding_field = embedding_field
        self.embedding_dim = embedding_dim
        self.excluded_meta_data = excluded_meta_data
        self.analyzer = analyzer
        self.return_embedding = return_embedding

        self.custom_mapping = custom_mapping
        self.synonyms = synonyms
        self.synonym_type = synonym_type
        self.index: str = index
        self.label_index: str = label_index
        self.scroll = scroll
        self.skip_missing_embeddings: bool = skip_missing_embeddings
        self.duplicate_documents = duplicate_documents
        self.refresh_type = refresh_type
        if similarity in ["cosine", "dot_product", "l2"]:
            self.similarity: str = similarity
        else:
            raise DocumentStoreError(
                f"Invalid value {similarity} for similarity, choose between 'cosine', 'l2' and 'dot_product'"
            )

        self._init_indices(
            index=index, label_index=label_index, create_index=create_index, recreate_index=recreate_index
        )

    def _init_indices(self, index: str, label_index: str, create_index: bool, recreate_index: bool) -> None:
        if recreate_index:
            self._delete_index(index)
            self._delete_index(label_index)

        if not self._index_exists(index) and (create_index or recreate_index):
            self._create_document_index(index)

        if self.custom_mapping:
            logger.warning("Cannot validate index for custom mappings. Skipping index validation.")
        else:
            self._validate_and_adjust_document_index(index)

        if not self._index_exists(label_index) and (create_index or recreate_index):
            self._create_label_index(label_index)

    def _split_document_list(
        self, documents: Union[List[dict], List[Document]], number_of_lists: int
    ) -> Generator[Union[List[dict], List[Document]], None, None]:
        chunk_size = max((len(documents) + 1) // number_of_lists, 1)
        for i in range(0, len(documents), chunk_size):
            yield documents[i : i + chunk_size]

    @abstractmethod
    def _do_bulk(self, *args, **kwargs):
        pass

    @abstractmethod
    def _do_scan(self, *args, **kwargs):
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def _create_document_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        pass

    @abstractmethod
    def _create_label_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        pass

    def _index_exists(self, index_name: str, headers: Optional[Dict[str, str]] = None) -> bool:
        if logger.isEnabledFor(logging.DEBUG):
            if self.client.indices.exists_alias(name=index_name):
                logger.debug("Index name %s is an alias.", index_name)

        return self.client.indices.exists(index_name, headers=headers)

    @abstractmethod
    def _validate_and_adjust_document_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        pass

    @abstractmethod
    def _get_vector_similarity_query(self, query_emb: np.ndarray, top_k: int):
        pass

    @abstractmethod
    def _get_raw_similarity_score(self, score):
        pass

    def _bulk(
        self,
        documents: Union[List[dict], List[Document]],
        headers: Optional[Dict[str, str]] = None,
        request_timeout: int = 300,
        refresh: str = "wait_for",
        _timeout: int = 1,
        _remaining_tries: int = 10,
    ) -> None:
        """
        Bulk index documents using a custom retry logic with
        exponential backoff and exponential batch size reduction to avoid overloading the cluster.

        The ingest node returns '429 Too Many Requests' when the write requests can't be
        processed because there are too many requests in the queue or the single request is too large and exceeds the
        memory of the nodes. Since the error code is the same for both of these cases we need to wait
        and reduce the batch size simultaneously.

        :param documents: List of documents to index
        :param headers: Optional headers to pass to the bulk request
        :param request_timeout: Timeout for the bulk request
        :param refresh: Refresh policy for the bulk request
        :param _timeout: Timeout for the exponential backoff
        :param _remaining_tries: Number of remaining retries
        """

        try:
            self._do_bulk(self.client, documents, request_timeout=300, refresh=self.refresh_type, headers=headers)
        except Exception as e:
            if hasattr(e, "status_code") and e.status_code == 429:  # type: ignore
                logger.warning(
                    "Failed to insert a batch of '%s' documents because of a 'Too Many Requeset' response. "
                    "Splitting the number of documents into two chunks with the same size and retrying in %s seconds.",
                    len(documents),
                    _timeout,
                )
                if len(documents) == 1:
                    logger.warning(
                        "Failed to index a single document. Your indexing queue on the cluster is probably full. Try resizing your cluster or reducing the number of parallel processes that are writing to the cluster."
                    )

                time.sleep(_timeout)

                _remaining_tries -= 1
                if _remaining_tries == 0:
                    raise DocumentStoreError("Last try of bulk indexing documents failed.")

                for split_docs in self._split_document_list(documents, 2):
                    self._bulk(
                        documents=split_docs,
                        headers=headers,
                        request_timeout=request_timeout,
                        refresh=refresh,
                        _timeout=_timeout * 2,
                        _remaining_tries=_remaining_tries,
                    )
                return
            raise e

    # TODO: Add flexibility to define other non-meta and meta fields expected by the Document class
    def _create_document_field_map(self) -> Dict:
        return {self.content_field: "content", self.embedding_field: "embedding"}

    def get_document_by_id(
        self, id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> Optional[Document]:
        """Fetch a document by specifying its text id string"""
        index = index or self.index
        documents = self.get_documents_by_id([id], index=index, headers=headers)
        if documents:
            return documents[0]
        else:
            return None

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Fetch documents by specifying a list of text id strings.

        :param ids: List of document IDs. Be aware that passing a large number of ids might lead to performance issues.
        :param index: search index where the documents are stored. If not supplied,
                      self.index will be used.
        :param batch_size: Maximum number of results for each query.
                           Limited to 10,000 documents by default.
                           To reduce the pressure on the cluster, you can lower this limit, at the expense
                           of longer retrieval times.
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                        Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """
        index = index or self.index
        documents = []
        for i in range(0, len(ids), batch_size):
            ids_for_batch = ids[i : i + batch_size]
            query = {"size": len(ids_for_batch), "query": {"ids": {"values": ids_for_batch}}}
            if not self.return_embedding and self.embedding_field:
                query["_source"] = {"excludes": [self.embedding_field]}
            result = self.client.search(index=index, body=query, headers=headers)["hits"]["hits"]
            documents.extend([self._convert_es_hit_to_document(hit) for hit in result])
        return documents

    def get_metadata_values_by_key(
        self,
        key: str,
        query: Optional[str] = None,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[dict]:
        """
        Get values associated with a metadata key. The output is in the format:
            [{"value": "my-value-1", "count": 23}, {"value": "my-value-2", "count": 12}, ... ]

        :param key: the meta key name to get the values for.
        :param query: narrow down the scope to documents matching the query string.
        :param filters: Narrow down the scope to documents that match the given filters.
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
                            ```
        :param index: search index where the meta values should be searched. If not supplied,
                      self.index will be used.
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """
        body: dict = {
            "size": 0,
            "aggs": {"metadata_agg": {"composite": {"sources": [{key: {"terms": {"field": key}}}]}}},
        }
        if query:
            body["query"] = {
                "bool": {
                    "should": [{"multi_match": {"query": query, "type": "most_fields", "fields": self.search_fields}}]
                }
            }
        if filters:
            if not body.get("query"):
                body["query"] = {"bool": {}}
            body["query"]["bool"].update({"filter": LogicalFilterClause.parse(filters).convert_to_elasticsearch()})
        result = self.client.search(body=body, index=index, headers=headers)

        values = []
        current_buckets = result["aggregations"]["metadata_agg"]["buckets"]
        after_key = result["aggregations"]["metadata_agg"].get("after_key", False)
        for bucket in current_buckets:
            values.append({"value": bucket["key"][key], "count": bucket["doc_count"]})

        # Only 10 results get returned at a time, so apply pagination
        while after_key:
            body["aggs"]["metadata_agg"]["composite"]["after"] = after_key
            result = self.client.search(body=body, index=index, headers=headers)
            current_buckets = result["aggregations"]["metadata_agg"]["buckets"]
            after_key = result["aggregations"]["metadata_agg"].get("after_key", False)
            for bucket in current_buckets:
                values.append({"value": bucket["key"][key], "count": bucket["doc_count"]})

        return values

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Indexes documents for later queries.

        If a document with the same ID already exists:
        a) (Default) Manage duplication according to the `duplicate_documents` parameter.
        b) If `self.update_existing_documents=True` for DocumentStore: Overwrite existing documents.
        (This is only relevant if you pass your own ID when initializing a `Document`.
        If you don't set custom IDs for your Documents or just pass a list of dictionaries here,
        they automatically get UUIDs assigned. See the `Document` class for details.)

        :param documents: A list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"content": "<the-actual-text>"}.
                          Optionally: Include meta data via {"content": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          You can use it for filtering and you can access it in the responses of the Finder.
                          Advanced: If you are using your own field mapping, change the key names in the dictionary
                          to what you have set for self.content_field and self.name_field.
        :param index: search index where the documents should be indexed. If you don't specify it, self.index is used.
        :param batch_size: Number of documents that are passed to the bulk function at each round.
        :param duplicate_documents: Handle duplicate documents based on parameter options.
                                    Parameter options: ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicate documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: Raises an error if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to the client (for example {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                For more information, see [HTTP/REST clients and security](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html).
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return: None
        """

        if index and not self._index_exists(index, headers=headers):
            self._create_document_index(index, headers=headers)

        if index is None:
            index = self.index
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert (
            duplicate_documents in self.duplicate_documents_options
        ), f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(
            documents=document_objects, index=index, duplicate_documents=duplicate_documents, headers=headers
        )
        documents_to_index = []
        for doc in document_objects:
            _doc = {
                "_op_type": "index" if duplicate_documents == "overwrite" else "create",
                "_index": index,
                **doc.to_dict(field_map=self._create_document_field_map()),
            }  # type: Dict[str, Any]

            # cast embedding type as ES cannot deal with np.array
            if _doc[self.embedding_field] is not None:
                if type(_doc[self.embedding_field]) == np.ndarray:
                    _doc[self.embedding_field] = _doc[self.embedding_field].tolist()

            # rename id for elastic
            _doc["_id"] = str(_doc.pop("id"))

            # don't index query score and empty fields
            _ = _doc.pop("score", None)
            _doc = {k: v for k, v in _doc.items() if v is not None}

            # In order to have a flat structure in elastic + similar behaviour to the other DocumentStores,
            # we "unnest" all value within "meta"
            if "meta" in _doc.keys():
                for k, v in _doc["meta"].items():
                    _doc[k] = v
                _doc.pop("meta")
            documents_to_index.append(_doc)

            # Pass batch_size number of documents to bulk
            if len(documents_to_index) % batch_size == 0:
                self._bulk(documents_to_index, request_timeout=300, refresh=self.refresh_type, headers=headers)
                documents_to_index = []

        if documents_to_index:
            self._bulk(documents_to_index, request_timeout=300, refresh=self.refresh_type, headers=headers)

    def write_labels(
        self,
        labels: Union[List[Label], List[dict]],
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: int = 10_000,
    ):
        """Write annotation labels into document store.

        :param labels: A list of Python dictionaries or a list of Haystack Label objects.
        :param index: search index where the labels should be stored. If not supplied, self.label_index will be used.
        :param batch_size: Number of labels that are passed to the bulk function at each round.
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """
        index = index or self.label_index
        if index and not self._index_exists(index, headers=headers):
            self._create_label_index(index, headers=headers)

        label_list: List[Label] = [Label.from_dict(label) if isinstance(label, dict) else label for label in labels]
        duplicate_ids: list = [label.id for label in self._get_duplicate_labels(label_list, index=index)]
        if len(duplicate_ids) > 0:
            logger.warning(
                "Duplicate Label IDs: Inserting a Label whose id already exists in this document store."
                " This will overwrite the old Label. Please make sure Label.id is a unique identifier of"
                " the answer annotation and not the question."
                " Problematic ids: %s",
                ",".join(duplicate_ids),
            )
        labels_to_index = []
        for label in label_list:
            # create timestamps if not available yet
            if not label.created_at:  # type: ignore
                label.created_at = time.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore
            if not label.updated_at:  # type: ignore
                label.updated_at = label.created_at  # type: ignore

            _label = {
                "_op_type": "index"
                if self.duplicate_documents == "overwrite" or label.id in duplicate_ids
                else "create",  # type: ignore
                "_index": index,
                **label.to_dict(),  # type: ignore
            }  # type: Dict[str, Any]

            # rename id for elastic
            if label.id is not None:  # type: ignore
                _label["_id"] = str(_label.pop("id"))  # type: ignore

            labels_to_index.append(_label)

            # Pass batch_size number of labels to bulk
            if len(labels_to_index) % batch_size == 0:
                self._bulk(labels_to_index, request_timeout=300, refresh=self.refresh_type, headers=headers)
                labels_to_index = []

        if labels_to_index:
            self._bulk(labels_to_index, request_timeout=300, refresh=self.refresh_type, headers=headers)

    def update_document_meta(
        self, id: str, meta: Dict[str, str], index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ):
        """
        Update the metadata dictionary of a document by specifying its string id
        """
        if not index:
            index = self.index
        body = {"doc": meta}
        self.client.update(index=index, id=id, body=body, refresh=self.refresh_type, headers=headers)

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Return the number of documents in the document store.
        """
        index = index or self.index

        body: dict = {"query": {"bool": {}}}
        if only_documents_without_embedding:
            body["query"]["bool"]["must_not"] = [{"exists": {"field": self.embedding_field}}]

        if filters:
            body["query"]["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        result = self.client.count(index=index, body=body, headers=headers)
        count = result["count"]
        return count

    def get_label_count(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int:
        """
        Return the number of labels in the document store
        """
        index = index or self.label_index
        return self.get_document_count(index=index, headers=headers)

    def get_embedding_count(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Return the count of embeddings in the document store.
        """

        index = index or self.index

        body: dict = {"query": {"bool": {"must": [{"exists": {"field": self.embedding_field}}]}}}
        if filters:
            body["query"]["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        result = self.client.count(index=index, body=body, headers=headers)
        count = result["count"]
        return count

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
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
                            ```
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """
        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size, headers=headers
        )
        documents = list(result)
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
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
                            ```
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """

        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        excludes = None
        if not return_embedding and self.embedding_field:
            excludes = [self.embedding_field]

        result = self._get_all_documents_in_index(
            index=index, filters=filters, batch_size=batch_size, headers=headers, excludes=excludes
        )
        for hit in result:
            document = self._convert_es_hit_to_document(hit)
            yield document

    def get_all_labels(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: int = 10_000,
    ) -> List[Label]:
        """
        Return all labels in the document store
        """
        index = index or self.label_index
        result = list(
            self._get_all_documents_in_index(index=index, filters=filters, batch_size=batch_size, headers=headers)
        )
        try:
            labels = [Label.from_dict({**hit["_source"], "id": hit["_id"]}) for hit in result]
        except ValidationError as e:
            raise DocumentStoreError(
                f"Failed to create labels from the content of index '{index}'. Are you sure this index contains labels?"
            ) from e
        return labels

    def _get_all_documents_in_index(
        self,
        index: str,
        filters: Optional[FilterType] = None,
        batch_size: int = 10_000,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
        excludes: Optional[List[str]] = None,
    ) -> Generator[dict, None, None]:
        """
        Return all documents in a specific index in the document store
        """
        body: dict = {"query": {"bool": {}}}

        if filters:
            body["query"]["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        if only_documents_without_embedding:
            body["query"]["bool"]["must_not"] = [{"exists": {"field": self.embedding_field}}]

        if excludes:
            body["_source"] = {"excludes": excludes}

        result = self._do_scan(
            self.client, query=body, index=index, size=batch_size, scroll=self.scroll, headers=headers
        )
        yield from result

    def query(
        self,
        query: Optional[str],
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        all_terms_must_match: bool = False,
        scale_score: bool = True,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query as defined by the BM25 algorithm.

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
        :param custom_query: query string containing a mandatory `${query}` placeholder.

                             Optionally, ES `filter` clause can be added where the values of `terms` are placeholders
                             that get substituted during runtime. The placeholder(${filter_name_1}, ${filter_name_2}..)
                             names must match with the filters dict supplied in self.retrieve().
                             ::

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

                             Optionally, highlighting can be defined by specifying the highlight settings.
                             See https://www.elastic.co/guide/en/elasticsearch/reference/current/highlighting.html.
                             You will find the highlighted output in the returned Document's meta field by key "highlighted".
                             ::

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

        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :param all_terms_must_match: Whether all terms of the query must match the document.
                                     If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
                                     Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
                                     Defaults to false.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """

        if index is None:
            index = self.index

        body = self._construct_query_body(
            query=query,
            filters=filters,
            top_k=top_k,
            custom_query=custom_query,
            all_terms_must_match=all_terms_must_match,
        )

        result = self.client.search(index=index, body=body, headers=headers)["hits"]["hits"]

        documents = [self._convert_es_hit_to_document(hit, scale_score=scale_score) for hit in result]
        return documents

    def query_batch(
        self,
        queries: List[str],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        all_terms_must_match: bool = False,
        scale_score: bool = True,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the provided queries as defined by keyword matching algorithms like BM25.

        This method lets you find relevant documents for list of query strings (output: List of Lists of Documents).

        :param queries: List of query strings.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions. Can be a single filter that will be applied to each query or a list of filters
                        (one filter per query).

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
        :param custom_query: Custom query to be executed.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        :param all_terms_must_match: Whether all terms of the query must match the document.
                                     If true all query terms must be present in a document in order to be retrieved (i.e the AND operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy AND fish AND restaurant").
                                     Otherwise at least one query term must be present in a document in order to be retrieved (i.e the OR operator is being used implicitly between query terms: "cozy fish restaurant" -> "cozy OR fish OR restaurant").
                                     Defaults to False.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """

        if index is None:
            index = self.index
        if headers is None:
            headers = {}

        if isinstance(filters, list):
            if len(filters) != len(queries):
                raise HaystackError(
                    "Number of filters does not match number of queries. Please provide as many filters"
                    " as queries or a single filter that will be applied to each query."
                )
        else:
            filters = [filters] * len(queries)

        body = []
        for query, cur_filters in zip(queries, filters):
            cur_query_body = self._construct_query_body(
                query=query,
                filters=cur_filters,
                top_k=top_k,
                custom_query=custom_query,
                all_terms_must_match=all_terms_must_match,
            )
            body.append(headers)
            body.append(cur_query_body)

        responses = self.client.msearch(index=index, body=body)

        all_documents = []
        cur_documents = []
        for response in responses["responses"]:
            cur_result = response["hits"]["hits"]
            cur_documents = [self._convert_es_hit_to_document(hit, scale_score=scale_score) for hit in cur_result]
            all_documents.append(cur_documents)

        return all_documents

    def _construct_query_body(
        self,
        query: Optional[str],
        filters: Optional[FilterType],
        top_k: int,
        custom_query: Optional[str],
        all_terms_must_match: bool,
    ) -> Dict[str, Any]:
        # Naive retrieval without BM25, only filtering
        if query is None:
            body = {"query": {"bool": {"must": {"match_all": {}}}}}  # type: Dict[str, Any]
            body["size"] = "10000"  # Set to the ES default max_result_window
            if filters:
                body["query"]["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        # Retrieval via custom query
        elif custom_query:  # substitute placeholder for query and filters for the custom_query template string
            template = Template(custom_query)
            # replace all "${query}" placeholder(s) with query
            substitutions = {"query": json.dumps(query)}
            # For each filter we got passed, we'll try to find & replace the corresponding placeholder in the template
            # Example: filters={"years":[2018]} => replaces {$years} in custom_query with '[2018]'
            if filters:
                for key, values in filters.items():
                    values_str = json.dumps(values)
                    substitutions[key] = values_str
            custom_query_json = template.substitute(**substitutions)
            body = json.loads(custom_query_json)
            # add top_k
            body["size"] = str(top_k)

        # Default Retrieval via BM25 using the user query on `self.search_fields`
        else:
            if not isinstance(query, str):
                logger.warning(
                    "The query provided seems to be not a string, but an object "
                    "of type %s. This can cause the query to fail.",
                    type(query),
                )
            operator = "AND" if all_terms_must_match else "OR"
            body = {
                "size": str(top_k),
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "type": "most_fields",
                                    "fields": self.search_fields,
                                    "operator": operator,
                                }
                            }
                        ]
                    }
                },
            }

            if filters:
                body["query"]["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        excluded_fields = self._get_excluded_fields(return_embedding=self.return_embedding)
        if excluded_fields:
            body["_source"] = {"excludes": excluded_fields}

        return body

    def _get_excluded_fields(self, return_embedding: bool) -> Optional[List[str]]:
        excluded_meta_data: Optional[list] = None

        if self.excluded_meta_data:
            excluded_meta_data = deepcopy(self.excluded_meta_data)

            if return_embedding is True and self.embedding_field in excluded_meta_data:
                excluded_meta_data.remove(self.embedding_field)
            elif return_embedding is False and self.embedding_field not in excluded_meta_data:
                excluded_meta_data.append(self.embedding_field)
        elif return_embedding is False:
            excluded_meta_data = [self.embedding_field]
        return excluded_meta_data

    def _convert_es_hit_to_document(
        self, hit: dict, adapt_score_for_embedding: bool = False, scale_score: bool = True
    ) -> Document:
        # We put all additional data of the doc into meta_data and return it in the API
        try:
            meta_data = {
                k: v
                for k, v in hit["_source"].items()
                if k not in (self.content_field, "content_type", "id_hash_keys", self.embedding_field)
            }
            name = meta_data.pop(self.name_field, None)
            if name:
                meta_data["name"] = name

            if "highlight" in hit:
                meta_data["highlighted"] = hit["highlight"]

            score = hit["_score"]
            if score:
                if adapt_score_for_embedding:
                    score = self._get_raw_similarity_score(score)

                if scale_score:
                    if adapt_score_for_embedding:
                        score = self.scale_to_unit_interval(score, self.similarity)
                    else:
                        score = float(expit(np.asarray(score / 8)))  # scaling probability from TFIDF/BM25

            embedding = None
            embedding_list = hit["_source"].get(self.embedding_field)
            if embedding_list:
                embedding = np.asarray(embedding_list, dtype=np.float32)

            doc_dict = {
                "id": hit["_id"],
                "content": hit["_source"].get(self.content_field),
                "content_type": hit["_source"].get("content_type", None),
                "id_hash_keys": hit["_source"].get("id_hash_keys", None),
                "meta": meta_data,
                "score": score,
                "embedding": embedding,
            }
            document = Document.from_dict(doc_dict)
        except (KeyError, ValidationError) as e:
            raise DocumentStoreError(
                "Failed to create documents from the content of the document store. Make sure the index you specified contains documents."
            ) from e
        return document

    def query_by_embedding_batch(
        self,
        query_embs: Union[List[np.ndarray], np.ndarray],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
    ) -> List[List[Document]]:
        """
        Find the documents that are most similar to the provided `query_embs` by using a vector similarity metric.

        :param query_embs: Embeddings of the queries (e.g. gathered from DPR).
                        Can be a list of one-dimensional numpy arrays or a two-dimensional numpy array.
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

        if headers is None:
            headers = {}

        if not self.embedding_field:
            raise DocumentStoreError("Please set a valid `embedding_field` for OpenSearchDocumentStore")

        if isinstance(filters, list):
            if len(filters) != len(query_embs):
                raise HaystackError(
                    "Number of filters does not match number of query_embs. Please provide as many filters"
                    " as query_embs or a single filter that will be applied to each query_emb."
                )
        else:
            filters = [filters] * len(query_embs) if filters is not None else [{}] * len(query_embs)

        body = []
        for query_emb, cur_filters in zip(query_embs, filters):
            cur_query_body = self._construct_dense_query_body(
                query_emb=query_emb, filters=cur_filters, top_k=top_k, return_embedding=return_embedding
            )
            body.append(headers)
            body.append(cur_query_body)

        logger.debug("Retriever query: %s", body)
        responses = self.client.msearch(index=index, body=body)

        all_documents = []
        cur_documents = []
        for response in responses["responses"]:
            cur_result = response["hits"]["hits"]
            cur_documents = [
                self._convert_es_hit_to_document(hit, adapt_score_for_embedding=True, scale_score=scale_score)
                for hit in cur_result
            ]
            all_documents.append(cur_documents)

        return all_documents

    @abstractmethod
    def _construct_dense_query_body(
        self, query_emb: np.ndarray, return_embedding: bool, filters: Optional[FilterType] = None, top_k: int = 10
    ):
        pass

    def update_embeddings(
        self,
        retriever: DenseRetriever,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        update_existing_embeddings: bool = True,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to update the embeddings.
        :param index: Index name to update
        :param update_existing_embeddings: Whether to update existing embeddings of the documents. If set to False,
                                           only documents without embeddings are processed. This mode can be used for
                                           incremental updating of embeddings, wherein, only newly indexed documents
                                           get processed.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
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
                            ```
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :return: None
        """
        if index is None:
            index = self.index

        if self.refresh_type == "false":
            self.client.indices.refresh(index=index, headers=headers)

        if not self.embedding_field:
            raise RuntimeError("Please specify the arg `embedding_field` when initializing the Document Store")

        if update_existing_embeddings:
            document_count = self.get_document_count(index=index, headers=headers)
        else:
            document_count = self.get_document_count(
                index=index, filters=filters, only_documents_without_embedding=True, headers=headers
            )

        logger.info(
            "Updating embeddings for all %s docs %s...",
            document_count,
            "without embeddings" if not update_existing_embeddings else "",
        )

        result = self._get_all_documents_in_index(
            index=index,
            filters=filters,
            batch_size=batch_size,
            only_documents_without_embedding=not update_existing_embeddings,
            headers=headers,
            excludes=[self.embedding_field],
        )

        logging.getLogger(__name__).setLevel(logging.CRITICAL)

        with tqdm(total=document_count, position=0, unit=" Docs", desc="Updating embeddings") as progress_bar:
            for result_batch in get_batches_from_generator(result, batch_size):
                document_batch = [self._convert_es_hit_to_document(hit) for hit in result_batch]
                embeddings = self._embed_documents(document_batch, retriever)

                doc_updates = []
                for doc, emb in zip(document_batch, embeddings):
                    update = {
                        "_op_type": "update",
                        "_index": index,
                        "_id": doc.id,
                        "doc": {self.embedding_field: emb.tolist()},
                    }
                    doc_updates.append(update)

                self._bulk(documents=doc_updates, request_timeout=300, refresh=self.refresh_type, headers=headers)
                progress_bar.update(batch_size)

    def _embed_documents(self, documents: List[Document], retriever: DenseRetriever) -> np.ndarray:
        """
        Embed a list of documents using a Retriever.
        :param documents: List of documents to embed.
        :param retriever: Retriever to use for embedding.
        :return: embeddings of documents.
        """
        embeddings = retriever.embed_documents(documents)
        self._validate_embeddings_shape(
            embeddings=embeddings, num_documents=len(documents), embedding_dim=self.embedding_dim
        )

        return embeddings

    def delete_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the document from.
        :param filters: Optional filters to narrow down the documents to be deleted.
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
                            ```
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :return: None
        """
        logger.warning(
            """DEPRECATION WARNINGS:
                1. delete_all_documents() method is deprecated, please use delete_documents method
                For more details, please refer to the issue: https://github.com/deepset-ai/haystack/issues/1045
                """
        )
        self.delete_documents(index, None, filters, headers=headers)

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the documents from. If None, the
                      DocumentStore's default index (self.index) will be used
        :param ids: Optional list of IDs to narrow down the documents to be deleted.
        :param filters: Optional filters to narrow down the documents to be deleted.
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
                            ```

                            If filters are provided along with a list of IDs, this method deletes the
                            intersection of the two query results (documents that match the filters and
                            have their ID in the list).
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :return: None
        """
        index = index or self.index
        query: Dict[str, Any] = {"query": {}}
        if filters:
            query["query"]["bool"] = {"filter": LogicalFilterClause.parse(filters).convert_to_elasticsearch()}

            if ids:
                query["query"]["bool"]["must"] = {"ids": {"values": ids}}

        elif ids:
            query["query"]["ids"] = {"values": ids}
        else:
            query["query"] = {"match_all": {}}
        self.client.delete_by_query(index=index, body=query, ignore=[404], headers=headers)
        # We want to be sure that all docs are deleted before continuing (delete_by_query doesn't support wait_for)
        if self.refresh_type == "wait_for":
            time.sleep(2)

    def delete_labels(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete labels in an index. All labels are deleted if no filters are passed.

        :param index: Index name to delete the labels from. If None, the
                      DocumentStore's default label index (self.label_index) will be used
        :param ids: Optional list of IDs to narrow down the labels to be deleted.
        :param filters: Optional filters to narrow down the labels to be deleted.
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
                            ```
        :param headers: Custom HTTP headers to pass to the client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :return: None
        """
        index = index or self.label_index
        self.delete_documents(index=index, ids=ids, filters=filters, headers=headers)

    def delete_index(self, index: str):
        """
        Delete an existing search index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        if index == self.index:
            logger.warning(
                "Deletion of default index '%s' detected. "
                "If you plan to use this index again, please reinstantiate '%s' in order to avoid side-effects.",
                index,
                self.__class__.__name__,
            )
        self._delete_index(index)

    def _delete_index(self, index: str):
        if self._index_exists(index):
            self.client.indices.delete(index=index, ignore=[400, 404])
            logger.info("Index '%s' deleted.", index)
