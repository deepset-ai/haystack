import json
import logging
from copy import deepcopy
from string import Template
from typing import Dict, List, Optional, Type, Union, Any, Generator
import time

import numpy as np
from pydantic import ValidationError
from tqdm.auto import tqdm

from haystack.document_stores.base import get_batches_from_generator
from haystack.nodes.retriever import DenseRetriever
from haystack.utils.scipy_utils import expit

try:
    from elasticsearch import Elasticsearch, RequestError
    from elastic_transport import BaseNode, RequestsHttpNode, Urllib3HttpNode
    from elasticsearch.helpers import bulk, scan
except (ImportError, ModuleNotFoundError) as ie:
    from haystack.utils.import_utils import _optional_component_not_installed

    _optional_component_not_installed(__name__, "elasticsearch8", ie)

from haystack.errors import DocumentStoreError, HaystackError
from haystack.schema import Document, FilterType, Label
from haystack.document_stores.filter_utils import LogicalFilterClause
from haystack.document_stores import KeywordDocumentStore


logger = logging.getLogger(__name__)


class ElasticsearchDocumentStore(KeywordDocumentStore):
    def __init__(
        self,
        host: Union[str, List[str]] = "localhost",
        port: Union[int, List[int]] = 9200,
        username: str = "",
        password: str = "",
        api_key_id: Optional[str] = None,
        api_key: Optional[str] = None,
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
        scheme: str = "http",
        ca_certs: Optional[str] = None,
        verify_certs: bool = False,
        recreate_index: bool = False,
        refresh_type: str = "wait_for",
        similarity: str = "dot_product",
        timeout: int = 300,
        return_embedding: bool = False,
        duplicate_documents: str = "overwrite",
        scroll: str = "1d",
        skip_missing_embeddings: bool = True,
        synonyms: Optional[List] = None,
        synonym_type: str = "synonym",
        use_system_proxy: bool = False,
        batch_size: int = 10_000,
    ):
        """
        A DocumentStore using Elasticsearch 8 for storing and retrieving documents.

            * Keeps all the logic to store and query documents from Elastic, incl. mapping of fields, adding filters or
              boosts to your queries, and storing embeddings.
            * You can either use an existing Elasticsearch index or create a new one via Haystack.
            * Retrievers operate on top of this DocumentStore to find the relevant documents for a query.

        :param host: URL(s) of Elasticsearch nodes.
        :param port: Port(s) of Elasticsearch nodes.
        :param username: Username (standard authentication via http_auth).
        :param password: Password (standard authentication via http_auth).
        :param api_key_id: ID of the API key (alternative authentication mode to the above http_auth).
        :param api_key: Secret value of the API key (alternative authentication mode to the above http_auth).
        :param index: Name of index in Elasticsearch to use for storing the documents that we want to search.
                      If non-existent, a new one will be created.
        :param label_index: Name of index in Elasticsearch to use for storing labels. If non-existent, a new one will
                            be created.
        :param search_fields: Name of fields used by BM25Retriever to find matches in the documents to the incoming query
                              (using Elasticsearch's multi_match query), for example `["title", "full_text"]`.
        :param content_field: Name of field that might contain the answer and will therefore be passed to the Reader Model
                              (for example `"full_text"`). If no Reader is used (e.g. in FAQ-Style QA),
                              the plain content of this field will just be returned.
        :param name_field: Name of field that contains the title of the documents.
        :param embedding_field: Name of field containing an embedding vector.
                                (Only needed when using a dense retriever, for example `EmbeddingRetriever`, on top.)
        :param embedding_dim: Dimensionality of embedding vector.
                              (Only needed when using a dense retriever, for example `EmbeddingRetriever`, on top.)
        :param custom_mapping: If you want to use your own custom mapping for creating a new index in Elasticsearch,
                               you can supply it here as a dictionary.
        :param analyzer: Specify the default analyzer from one of the built-ins when creating a new Elasticsearch Index.
                         Elasticsearch also has built-in analyzers for different languages
                         (impacting tokenization, for example).
                         You can find more details in the [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-analyzers.html).
        :param excluded_meta_data: Name of fields in Elasticsearch that should not be returned
                                   (for example `["field_one", "field_two"]`).
                                   Helpful if you have fields with long, irrelevant content that you don't want to
                                   display in results (for example embedding vectors).
        :param scheme: `'https'` or `'http'`, protocol used to connect to your Elasticsearch instance.
        :param ca_certs: Root certificates for SSL: it is a path to certificate authority (CA) certs on disk.
                         You can use the `certifi` package with `certifi.where()` to find where the CA certs file is
                         located on your machine.
        :param verify_certs: Whether to be strict about ca certificates.
        :param recreate_index: If set to `True`, an existing Elasticsearch index will be deleted and a new one will be
                               created using the configuration you are using for initialization. Be aware that all data
                               in the old index will be lost if you choose to recreate the index. Be aware that both the
                               document_index and the label_index will be recreated.
        :param refresh_type: Type of Elasticsearch refresh used to control when changes made by a request (e.g. bulk)
                             are made visible to search.
                             If set to `'wait_for'`, continue only after changes are visible (slow, but safe).
                             If set to `'false'`, continue directly (fast, but sometimes unintuitive behaviour when
                             documents are not immediately available after ingestion).
                             You can find more details in the [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-refresh.html).
        :param similarity: The similarity function used to compare document vectors.
                           Available options: `'dot_product'` (default), `'cosine'`, and `'l2'`.
        :param timeout: Timeout for an Elasticsearch request in seconds.
        :param return_embedding: Whether to return the document embedding.
        :param duplicate_documents: How to handle duplicate documents.
                                    Parameter options: `'skip'`, `'overwrite'`, or `'fail'
                                    * skip: Ignore duplicate documents.
                                    * overwrite: Update any existing document with the same ID when adding documents.
                                    * fail: Raise an error if the document ID of the document being added already exists.
        :param scroll: Duration to fix the current index, for example during updating all documents with embeddings.
                       Defaults to `"1d"` and should not be larger than this. Can also be in minutes `"5m"` or hours
                       `"15h"`
                       You can find more details in the [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/scroll-api.html).
        :param skip_missing_embeddings: Whether to ignore documents without embeddings during vector similarity queries.
                                        Parameter options: `True` or `False`
                                        * False: Raises an exception if one or more documents do not have embeddings at query time.
                                        * True: Query will ignore all documents without embeddings (recommended if you concurrently index and query).
        :param synonyms: List of synonyms used by Elasticsearch.
                         For example: `["foo, bar => baz", "foozball , foosball"]`
                         You can fing more details in the [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-tokenfilter.html).
        :param synonym_type: Synonym filter type for handling synonyms during the analysis process.
                             You can fing more details in the [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-tokenfilter.html).
        :param use_system_proxy: Whether to use system proxy.
        :param batch_size: Number of Documents to index at once / Number of queries to execute at once. If you face
                           memory issues, decrease the batch_size.

        """
        super().__init__()

        self.client = self._init_elastic_client(
            host=host,
            port=port,
            username=username,
            password=password,
            api_key=api_key,
            api_key_id=api_key_id,
            scheme=scheme,
            ca_certs=ca_certs,
            verify_certs=verify_certs,
            timeout=timeout,
            use_system_proxy=use_system_proxy,
        )

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
        self.batch_size = batch_size
        if similarity in ["cosine", "dot_product", "l2"]:
            self.similarity: str = similarity
        else:
            raise DocumentStoreError(
                f"Invalid value {similarity} for similarity, choose between 'cosine', 'l2' and 'dot_product'"
            )

        self._init_indices(index=index, label_index=label_index, recreate_index=recreate_index)

    @classmethod
    def _init_elastic_client(
        cls,
        host: Union[str, List[str]],
        port: Union[int, List[int]],
        username: str,
        password: str,
        api_key_id: Optional[str],
        api_key: Optional[str],
        scheme: str,
        ca_certs: Optional[str],
        verify_certs: bool,
        timeout: int,
        use_system_proxy: bool,
    ) -> Elasticsearch:
        hosts = cls._prepare_hosts(host, port, scheme)

        if (api_key or api_key_id) and not (api_key and api_key_id):
            raise ValueError("You must provide either both or none of `api_key_id` and `api_key`.")

        node_class: Type[BaseNode] = Urllib3HttpNode
        if use_system_proxy:
            node_class = RequestsHttpNode

        if api_key:
            # api key authentication
            client = Elasticsearch(
                hosts=hosts,
                api_key=(api_key_id, api_key),
                ca_certs=ca_certs,
                verify_certs=verify_certs,
                request_timeout=timeout,
                node_class=node_class,
            )
        elif username:
            # standard http_auth
            client = Elasticsearch(
                hosts=hosts,
                basic_auth=(username, password),
                ca_certs=ca_certs,
                verify_certs=verify_certs,
                request_timeout=timeout,
                node_class=node_class,
            )
        else:
            # there is no authentication for this elasticsearch instance
            client = Elasticsearch(
                hosts=hosts,
                ca_certs=ca_certs,
                verify_certs=verify_certs,
                request_timeout=timeout,
                node_class=node_class,
            )

        # Test connection
        try:
            # ping uses a HEAD request on the root URI. In some cases, the user might not have permissions for that,
            # resulting in a HTTP Forbidden 403 response.
            if username in ["", "elastic"]:
                status = client.ping()
                if not status:
                    raise ConnectionError(
                        f"Initial connection to Elasticsearch failed. Make sure you run an Elasticsearch instance "
                        f"at `{hosts}` and that it has finished the initial ramp up (can take > 30s). Also, make sure "
                        f"you are using the correct credentials if you are using a secured Elasticsearch instance."
                    )
        except Exception:
            raise ConnectionError(
                f"Initial connection to Elasticsearch failed. Make sure you run an Elasticsearch instance at `{hosts}` "
                f"and that it has finished the initial ramp up (can take > 30s). Also, make sure you are using the "
                f"correct credentials if you are using a secured Elasticsearch instance."
            )
        return client

    def _init_indices(self, index: str, label_index: str, recreate_index: bool) -> None:
        if recreate_index:
            self._delete_index(index)
            self._delete_index(label_index)

        if not self._index_exists(index):
            self._create_document_index(index)

        if self.custom_mapping:
            logger.warning("Cannot validate index for custom mappings. Skipping index validation.")
        else:
            self._validate_and_adjust_document_index(index)

        if not self._index_exists(label_index) and recreate_index:
            self._create_label_index(label_index)

    @staticmethod
    def _prepare_hosts(host: Union[str, List[str]], port: Union[int, List[int]], scheme: str):
        """
        Create a list of host(s), port(s) and scheme to allow direct client connections to multiple nodes,
        in the format expected by the client.
        """
        if isinstance(host, list):
            if isinstance(port, list):
                if not len(port) == len(host):
                    raise ValueError("Length of list `host` must match length of list `port`")
                hosts = [{"host": h, "port": p, "scheme": scheme} for h, p in zip(host, port)]
            else:
                hosts = [{"host": h, "port": port, "scheme": scheme} for h in host]
        else:
            hosts = [{"host": host, "port": port, "scheme": scheme}]
        return hosts

    def _index_exists(self, index_name: str, headers: Optional[Dict[str, str]] = None) -> bool:
        if logger.isEnabledFor(logging.DEBUG):
            if self.client.options(headers=headers).indices.exists_alias(name=index_name):
                logger.debug("Index name %s is an alias.", index_name)

        return self.client.options(headers=headers).indices.exists(index=index_name)

    def _create_document_field_map(self) -> Dict:
        return {self.content_field: "content", self.embedding_field: "embedding"}

    def _bulk(
        self,
        documents: List[dict],
        headers: Optional[Dict[str, str]] = None,
        refresh: str = "wait_for",
        _timeout: int = 1,
        _remaining_tries: int = 10,
    ) -> None:
        """
        Bulk index documents using a custom retry logic with exponential backoff and exponential batch size reduction
        to avoid overloading the cluster.

        The ingest node returns '429 Too Many Requests' when the write requests can't be
        processed because there are too many requests in the queue or the single request is too large and exceeds the
        memory of the nodes. Since the error code is the same for both of these cases we need to wait
        and reduce the batch size simultaneously.

        :param documents: List of dictionaries containing document / label and indexing action.
        :param headers: Optional headers to pass to the bulk request.
        :param refresh: Refresh policy for the bulk request.
        :param _timeout: Timeout for the exponential backoff.
        :param _remaining_tries: Number of remaining retries.
        """

        try:
            bulk(client=self.client.options(headers=headers), actions=documents, refresh=self.refresh_type)
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
                        "Failed to index a single document. Your indexing queue on the cluster is probably full. "
                        "Try resizing your cluster or reducing the number of parallel processes that are writing to "
                        "the cluster."
                    )

                time.sleep(_timeout)

                _remaining_tries -= 1
                if _remaining_tries == 0:
                    raise DocumentStoreError("Last try of bulk indexing documents failed.")

                for split_docs in self._split_list(documents, 2):
                    self._bulk(
                        documents=split_docs,
                        headers=headers,
                        refresh=refresh,
                        _timeout=_timeout * 2,
                        _remaining_tries=_remaining_tries,
                    )
                return
            raise e

    def get_document_by_id(
        self, id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> Optional[Document]:
        """Fetch a document by specifying its ID"""
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
        Fetch documents by specifying a list of IDs.

        :param ids: List of Document IDs. Be aware that passing a large number of Ids might lead to performance issues.
        :param index: Index where the documents are stored. If not supplied, self.index will be used.
        :param batch_size: Maximum number of results for each query.
                           Limited to 10,000 documents by default.
                           To reduce the pressure on the cluster, you can lower this limit.
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
        """
        index = index or self.index
        documents = []
        for i in range(0, len(ids), batch_size):
            ids_for_batch = ids[i : i + batch_size]
            es_query = {"ids": {"values": ids_for_batch}}
            size = len(ids_for_batch)
            source_excludes = None
            if not self.return_embedding and self.embedding_field:
                source_excludes = self.embedding_field
            result = self.client.options(headers=headers).search(
                index=index, query=es_query, size=size, source_excludes=source_excludes
            )["hits"]["hits"]
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
        `[{"value": "my-value-1", "count": 23}, {"value": "my-value-2", "count": 12}, ... ]`

        :param key: The meta key name to get the values for.
        :param query: Narrow down the scope to documents matching the query string.
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
        :param index: Index where the meta values are retrieved from. If not supplied, self.index will be used.
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
        """
        aggs = {"metadata_agg": {"composite": {"sources": [{key: {"terms": {"field": key}}}]}}}
        es_query = None
        if query:
            es_query = {
                "bool": {
                    "should": [{"multi_match": {"query": query, "type": "most_fields", "fields": self.search_fields}}]
                }
            }
        if filters:
            if es_query is None:
                es_query = {"bool": {}}
            es_query["bool"].update({"filter": LogicalFilterClause.parse(filters).convert_to_elasticsearch()})
        result = self.client.options(headers=headers).search(aggs=aggs, query=es_query, size=0, index=index)

        values = []
        current_buckets = result["aggregations"]["metadata_agg"]["buckets"]
        after_key = result["aggregations"]["metadata_agg"].get("after_key", False)
        for bucket in current_buckets:
            values.append({"value": bucket["key"][key], "count": bucket["doc_count"]})

        # Only 10 results get returned at a time, so apply pagination
        while after_key:
            aggs["metadata_agg"]["composite"]["after"] = after_key
            result = self.client.options(headers=headers).search(aggs=aggs, query=es_query, size=0, index=index)
            current_buckets = result["aggregations"]["metadata_agg"]["buckets"]
            after_key = result["aggregations"]["metadata_agg"].get("after_key", False)
            for bucket in current_buckets:
                values.append({"value": bucket["key"][key], "count": bucket["doc_count"]})

        return values

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: Optional[int] = None,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Indexes documents for later queries.

        If a document with the same ID already exists, the situation is managed according to the `duplicate_documents`
        parameter.

        :param documents: A list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is `{"content": "<the-actual-text>"}`.
                          Optionally: Include meta data via
                          `{"content": "<the-actual-text>", "meta": {"name": "<some-document-name>, "author": "somebody", ...}}`
                          You can use the metadata for filtering, and you can access it in the responses of the Retriever.
                          Advanced: If you are using your own field mapping, change the key names in the dictionary
                          to what you have set for `self.content_field` and `self.name_field`.
        :param index: Name of the index where the documents should be indexed. If you don't specify it, `self.index` is used.
        :param batch_size: Number of documents that are passed to the bulk function at a time.
                           If not specified, `self.batch_size` is used.
        :param duplicate_documents: How to handle duplicate documents. If not set, `self.duplicate_documents` is used.
                                    Parameter options: `'skip'`, `'overwrite'`, or `'fail'
                                    * skip: Ignore duplicate documents.
                                    * overwrite: Update any existing document with the same ID when adding documents.
                                    * fail: Raise an error if the document ID of the document being added already exists.
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        For more information, checkout [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html).
        """

        if index and not self._index_exists(index, headers=headers):
            self._create_document_index(index, headers=headers)

        if index is None:
            index = self.index

        batch_size = batch_size or self.batch_size

        duplicate_documents = duplicate_documents or self.duplicate_documents
        if duplicate_documents not in self.duplicate_documents_options:
            raise DocumentStoreError(
                f"duplicate_documents parameter must be one " f"of: {', '.join(self.duplicate_documents_options)}. "
            )

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
                self._bulk(documents_to_index, refresh=self.refresh_type, headers=headers)
                documents_to_index = []

        if documents_to_index:
            self._bulk(documents_to_index, refresh=self.refresh_type, headers=headers)

    def write_labels(
        self,
        labels: Union[List[Label], List[dict]],
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: int = 10_000,
    ):
        """
        Write annotation labels into document store.

        :param labels: A list of Python dictionaries or a list of Haystack Label objects.
        :param index: Name of index where the labels should be stored. If not supplied, `self.label_index` will be used.
        :param batch_size: Number of labels that are passed to the bulk function at a time.
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
        """
        index = index or self.label_index
        if index and not self._index_exists(index, headers=headers):
            self._create_label_index(index, headers=headers)

        label_list: List[Label] = [Label.from_dict(label) if isinstance(label, dict) else label for label in labels]
        duplicate_ids: list = [label.id for label in self._get_duplicate_labels(label_list, index=index)]
        if len(duplicate_ids) > 0:
            logger.warning(
                "Duplicate Label IDs: Inserting a Label whose ID already exists in this document store."
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
                self._bulk(labels_to_index, refresh=self.refresh_type, headers=headers)
                labels_to_index = []

        if labels_to_index:
            self._bulk(labels_to_index, refresh=self.refresh_type, headers=headers)

    def update_document_meta(
        self, id: str, meta: Dict[str, str], index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ):
        """
        Update the metadata dictionary of a document by specifying its ID.
        :param id: ID of the document.
        :param meta: New metadata dictionary.
        :param index: Index name where the document is stored. If not supplied, `self.index` will be used.
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
        """
        if not index:
            index = self.index
        self.client.options(headers=headers).update(index=index, id=id, doc=meta, refresh=self.refresh_type)

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Return the number of documents in the document store.
        :param filters: Optional filters to narrow down the count.
        :param index: Index name where the documents are stored. If not supplied, `self.index` will be used.
        :param only_documents_without_embedding: If `True`, only documents without an embedding will be counted.
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
        """
        index = index or self.index

        es_query: dict = {"bool": {}}
        if only_documents_without_embedding:
            es_query["bool"]["must_not"] = [{"exists": {"field": self.embedding_field}}]
        if filters:
            es_query["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        result = self.client.options(headers=headers).count(index=index, query=es_query)
        count = result["count"]
        return count

    def get_label_count(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int:
        """
        Return the number of labels in the document store.
        :param index: Index name where the labels are stored. If not supplied, `self.label_index` will be used.
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
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
        :param index: Index name where the documents are stored. If not supplied, `self.index` will be used.
        :param filters: Optional filters to narrow down the count.
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
        """

        index = index or self.index

        es_query = {"bool": {"must": [{"exists": {"field": self.embedding_field}}]}}
        if filters:
            es_query["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        result = self.client.options(headers=headers).count(index=index, query=es_query)
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

        :param index: Name of the index to get the documents from. If not specified, `self.index` will be used.
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
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
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

        :param index: Name of the index to get the documents from. If not specified, `self.index` will be used.
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
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
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
        Return all labels in the document store.
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
        es_query: dict = {"query": {"bool": {}}}
        if filters:
            es_query["query"]["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        if only_documents_without_embedding:
            es_query["query"]["bool"]["must_not"] = [{"exists": {"field": self.embedding_field}}]

        if excludes:
            es_query["_source"] = {"excludes": excludes}

        result = scan(
            client=self.client, query=es_query, index=index, size=batch_size, scroll=self.scroll, headers=headers
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
        Scan through documents in DocumentStore and return the top k documents
        that are most relevant to the query as defined by the BM25 algorithm.

        :param query: The query.
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
        :param custom_query: Query string containing a mandatory `${query}` placeholder.

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

                             Optionally, highlighting can be defined by specifying the highlight settings.
                             See [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/highlighting.html).
                             You will find the highlighted output in the returned Document's meta field by key
                             `"highlighted"`.

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

        :param index: The name of the index in the DocumentStore from which to retrieve documents.
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
        :param all_terms_must_match: Whether all terms of the query must match the document.
                                     If `True`, all query terms must be present in a document in order to be retrieved
                                     (this means the AND operator is being used implicitly between query terms:
                                     "cozy fish restaurant" -> "cozy AND fish AND restaurant").
                                     Otherwise, at least one query term must be present in a document in order to be
                                     retrieved (i.e the OR operator is being used implicitly between query terms:
                                     "cozy fish restaurant" -> "cozy OR fish OR restaurant").
                                     Defaults to `False`.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If `True` (default), similarity scores (for example cosine or dot_product) which naturally
                            have a different value range will be scaled to a range of [0,1], where 1 means extremely
                            relevant. Otherwise, raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        if index is None:
            index = self.index

        es_query = self._construct_query_body(
            query=query, filters=filters, custom_query=custom_query, all_terms_must_match=all_terms_must_match
        )
        # Set to the ES default max_result_window if no query is specified
        size = 10000 if query is None else top_k
        excluded_fields = self._get_excluded_fields(return_embedding=self.return_embedding)

        result = self.client.options(headers=headers).search(
            index=index, query=es_query, size=size, source_excludes=excluded_fields
        )["hits"]["hits"]

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
        batch_size: Optional[int] = None,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return the top-k most relevant documents to the provided queries as
        defined by keyword matching algorithms like BM25.

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
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
        :param all_terms_must_match: Whether all terms of the query must match the document.
                                     If `True`, all query terms must be present in a document in order to be retrieved
                                     (this means the AND operator is being used implicitly between query terms:
                                     "cozy fish restaurant" -> "cozy AND fish AND restaurant").
                                     Otherwise, at least one query term must be present in a document in order to be
                                     retrieved (i.e. the OR operator is being used implicitly between query terms:
                                     "cozy fish restaurant" -> "cozy OR fish OR restaurant").
                                     Defaults to `False`.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If `True` (default), similarity scores (for example cosine or dot_product) which naturally
                            have a different value range will be scaled to a range of [0,1], where 1 means extremely
                            relevant. Otherwise, raw similarity scores (e.g. cosine or dot_product) will be used.
        :param batch_size: Number of queries that are processed at once. If not specified, self.batch_size is used.
        """

        if index is None:
            index = self.index
        if headers is None:
            headers = {}
        batch_size = batch_size or self.batch_size

        if isinstance(filters, list):
            if len(filters) != len(queries):
                raise HaystackError(
                    "Number of filters does not match number of queries. Please provide as many filters"
                    " as queries or a single filter that will be applied to each query."
                )
        else:
            filters = [filters] * len(queries)

        body: List[dict] = []
        all_documents = []
        for query, cur_filters in zip(queries, filters):
            cur_query_body = {
                "query": self._construct_query_body(
                    query=query,
                    filters=cur_filters,
                    custom_query=custom_query,
                    all_terms_must_match=all_terms_must_match,
                ),
                "size": top_k,
            }
            if not self.return_embedding:
                cur_query_body["_source"] = {"excludes": [self.embedding_field]}
            body.append(headers)
            body.append(cur_query_body)

            if len(body) == 2 * batch_size:
                cur_documents = self._execute_msearch(index=index, searches=body, scale_score=scale_score)
                all_documents.extend(cur_documents)
                body = []

        if len(body) > 0:
            cur_documents = self._execute_msearch(index=index, searches=body, scale_score=scale_score)
            all_documents.extend(cur_documents)

        return all_documents

    def _construct_query_body(
        self,
        query: Optional[str],
        filters: Optional[FilterType],
        custom_query: Optional[str],
        all_terms_must_match: bool,
    ) -> Dict[str, Any]:
        # Naive retrieval without BM25, only filtering
        if query is None:
            es_query: dict = {"bool": {"must": {"match_all": {}}}}
            if filters:
                es_query["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

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
            es_query = json.loads(custom_query_json)

        # Default Retrieval via BM25 using the user query on `self.search_fields`
        else:
            if not isinstance(query, str):
                logger.warning(
                    "The provided query doesn't seem to be a string, but an object of type %s. "
                    "This can cause the query to fail.",
                    type(query),
                )
            operator = "AND" if all_terms_must_match else "OR"
            es_query = {
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
            }
            if filters:
                es_query["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        return es_query

    def _execute_msearch(self, index: str, searches: List[Dict[str, Any]], scale_score: bool) -> List[List[Document]]:
        responses = self.client.msearch(index=index, searches=searches)
        documents = []
        for response in responses["responses"]:
            result = response["hits"]["hits"]
            cur_documents = [self._convert_es_hit_to_document(hit, scale_score=scale_score) for hit in result]
            documents.append(cur_documents)

        return documents

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
        :param top_k: How many documents to return per query.
        :param index: The name of the index to retrieve the documents from. If not specified, `self.index` is used.
        :param return_embedding: Whether to return the documents with their embeddings.
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If `True` (default), similarity scores (e.g. cosine or dot_product) which naturally have a
                            different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise, raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        if not self.embedding_field:
            raise RuntimeError("Please specify arg `embedding_field` in ElasticsearchDocumentStore()")

        es_query = self._construct_dense_query_body(query_emb=query_emb, filters=filters)
        excluded_fields = self._get_excluded_fields(return_embedding=return_embedding)

        try:
            result = self.client.options(headers=headers).search(
                index=index, query=es_query, size=top_k, source_excludes=excluded_fields
            )["hits"]["hits"]
            if len(result) == 0:
                count_documents = self.get_document_count(index=index, headers=headers)
                if count_documents == 0:
                    logger.warning("Index is empty. First add some documents to search them.")
                count_embeddings = self.get_embedding_count(index=index, headers=headers)
                if count_embeddings == 0:
                    logger.warning("No documents with embeddings. Run the document store's update_embeddings() method.")
        except RequestError as e:
            if e.error == "search_phase_execution_exception":
                error_message: str = (
                    "search_phase_execution_exception: Likely some of your stored documents don't have embeddings. "
                    "Run the document store's update_embeddings() method."
                )
                raise RequestError(message=error_message, meta=e.meta, body=e.info)
            raise e

        documents = [
            self._convert_es_hit_to_document(hit, adapt_score_for_embedding=True, scale_score=scale_score)
            for hit in result
        ]
        return documents

    def query_by_embedding_batch(
        self,
        query_embs: Union[List[np.ndarray], np.ndarray],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
        batch_size: Optional[int] = None,
    ) -> List[List[Document]]:
        """
        Find the documents that are most similar to the provided query embeddings by using a vector similarity metric.

        :param query_embs: Embeddings of the queries (for example gathered from the EmbeddingRetriever).
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
        :param top_k: How many documents to return per query.
        :param index: The name of the index to retrieve the documents from. If not specified, `self.index` is used.
        :param return_embedding: Whether to return the document embedding with the document or not.
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If `True` (default), similarity scores (e.g. cosine or dot_product) which naturally have a
                            different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise, raw similarity scores (e.g. cosine or dot_product) will be used.
        :param batch_size: Number of query embeddings to process at once. If not specified, `self.batch_size` is used.
        """
        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        if headers is None:
            headers = {}

        batch_size = batch_size or self.batch_size

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

        body: List[dict] = []
        all_documents = []
        for query_emb, cur_filters in zip(query_embs, filters):
            cur_query_body = {
                "query": self._construct_dense_query_body(query_emb=query_emb, filters=cur_filters),
                "size": top_k,
            }
            if not return_embedding:
                cur_query_body["_source"] = {"excludes": [self.embedding_field]}
            body.append(headers)
            body.append(cur_query_body)

            if len(body) >= batch_size * 2:
                logger.debug("Retriever query: %s", body)
                cur_documents = self._execute_msearch(index=index, searches=body, scale_score=scale_score)
                all_documents.extend(cur_documents)
                body = []

        if len(body) > 0:
            logger.debug("Retriever query: %s", body)
            cur_documents = self._execute_msearch(index=index, searches=body, scale_score=scale_score)
            all_documents.extend(cur_documents)

        return all_documents

    def update_embeddings(
        self,
        retriever: DenseRetriever,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        update_existing_embeddings: bool = True,
        batch_size: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Updates the embeddings in the document store using the encoding model specified in the retriever.
        This can be useful if you want to add or change the embeddings for your documents (e.g. after changing the
        retriever config).

        :param retriever: Retriever to use to update the embeddings.
        :param index: Index name to update.
        :param update_existing_embeddings: Whether to update existing embeddings of the documents. If set to `False`,
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
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
        """
        if index is None:
            index = self.index

        batch_size = batch_size or self.batch_size

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

                self._bulk(documents=doc_updates, refresh=self.refresh_type, headers=headers)
                progress_bar.update(batch_size)

    def _embed_documents(self, documents: List[Document], retriever: DenseRetriever) -> np.ndarray:
        """
        Embed a list of documents using a Retriever.
        :param documents: List of documents to embed.
        :param retriever: Retriever to use for embedding.
        """
        embeddings = retriever.embed_documents(documents)
        self._validate_embeddings_shape(
            embeddings=embeddings, num_documents=len(documents), embedding_dim=self.embedding_dim
        )

        return embeddings

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the documents from. If not specified, the DocumentStore's default index
                      (`self.index`) will be used.
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
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
        """
        index = index or self.index
        es_query: Dict[str, Any] = {}
        if filters:
            es_query["bool"] = {"filter": LogicalFilterClause.parse(filters).convert_to_elasticsearch()}

            if ids:
                es_query["bool"]["must"] = {"ids": {"values": ids}}

        elif ids:
            es_query["ids"] = {"values": ids}
        else:
            es_query = {"match_all": {}}
        self.client.options(headers=headers, ignore_status=404).delete_by_query(index=index, query=es_query)
        # We want to be sure that all docs are deleted before continuing (delete_by_query doesn't support wait_for)
        if self.refresh_type == "wait_for":
            self.client.indices.refresh(index=index)

    def delete_labels(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete labels in an index. All labels are deleted if no filters are passed.

        :param index: Index name to delete the labels from. If not specified, the DocumentStore's default label index
                      (`self.label_index`) will be used.
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
        :param headers: Custom HTTP headers to pass to the Elasticsearch client
                        (for example `{'Authorization': 'Basic YWRtaW46cm9vdA=='}`).
                        Check out [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html)
                        for more information.
        """
        index = index or self.label_index
        self.delete_documents(index=index, ids=ids, filters=filters, headers=headers)

    def delete_index(self, index: str):
        """
        Delete an existing search index. The index including all data will be removed.

        :param index: The name of the index to delete.
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
            self.client.options(ignore_status=[400, 404]).indices.delete(index=index)
            logger.info("Index '%s' deleted.", index)

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
                "Failed to create documents from the content of the document store. Make sure the index you specified "
                "contains documents."
            ) from e
        return document

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
                mapping["mappings"]["properties"][self.embedding_field] = {
                    "type": "dense_vector",
                    "dims": self.embedding_dim,
                }

        try:
            self.client.options(headers=headers).indices.create(index=index_name, **mapping)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self._index_exists(index_name, headers=headers):
                raise e

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
            self.client.options(headers=headers).indices.create(index=index_name, **mapping)
        except RequestError as e:
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
        indices = self.client.options(headers=headers).indices.get(index=index_name)

        if not any(indices):
            logger.warning(
                "To use an index, you must create it first. The index called '%s' doesn't exist. "
                "You can create it by setting `create_index=True` on init or by calling `write_documents()` if you "
                "prefer to create it on demand. Note that this instance doesn't validate the index after you create it.",
                index_name,
            )

        # If the index name is an alias that groups multiple existing indices, each of them must have an embedding_field.
        for index_id, index_info in indices.items():
            dynamic_templates = (
                index_info["mappings"]["dynamic_templates"] if "dynamic_templates" in index_info["mappings"] else []
            )
            properties = index_info["mappings"]["properties"] if "properties" in index_info["mappings"] else {}
            if self.search_fields:
                for search_field in self.search_fields:
                    if search_field in properties:
                        if properties[search_field]["type"] != "text":
                            raise DocumentStoreError(
                                f"Remove '{search_field}' from `search_fields` or use another index if you want to "
                                f"query it using full text search.  '{search_field}' of index '{index_id}' has type "
                                f"'{properties[search_field]['type']}' but needs 'text' for full text search. "
                                f"This error might occur if you are trying to use Haystack 1.0 and above with an "
                                f"existing Elasticsearch index created with a previous version of Haystack. "
                                f"Recreating the index with `recreate_index=True` will fix your environment. "
                                f"Note that you'll lose all data stored in the index."
                            )
                    else:
                        properties[search_field] = (
                            {"type": "text", "analyzer": "synonym"} if self.synonyms else {"type": "text"}
                        )
                        self.client.options(headers=headers).indices.put_mapping(
                            index=index_id, dynamic_templates=dynamic_templates, properties=properties
                        )

            if self.embedding_field:
                if self.embedding_field in properties and properties[self.embedding_field]["type"] != "dense_vector":
                    raise DocumentStoreError(
                        f"Update the document store to use a different name for the `embedding_field` parameter. "
                        f"The index '{index_id}' in Elasticsearch already has a field called '{self.embedding_field}' "
                        f"of type '{properties[self.embedding_field]['type']}'."
                    )
                properties[self.embedding_field] = {"type": "dense_vector", "dims": self.embedding_dim}
                self.client.options(headers=headers).indices.put_mapping(
                    index=index_id, dynamic_templates=dynamic_templates, properties=properties
                )

    def _construct_dense_query_body(self, query_emb: np.ndarray, filters: Optional[FilterType]) -> Dict:
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
                "Invalid value for similarity in ElasticSearchDocumentStore constructor. Choose between 'cosine', "
                "'dot_product', and 'l2'"
            )

        # To handle scenarios where embeddings may be missing
        script_score_query: dict = {"match_all": {}}
        if self.skip_missing_embeddings:
            script_score_query = {"bool": {"filter": {"bool": {"must": [{"exists": {"field": self.embedding_field}}]}}}}

        if filters:
            filter_ = LogicalFilterClause.parse(filters).convert_to_elasticsearch()
            if self.skip_missing_embeddings:
                script_score_query["bool"]["filter"]["bool"]["must"].append(filter_)
            else:
                script_score_query = {"bool": {"filter": filter_}}

        query = {
            "script_score": {
                "query": script_score_query,
                "script": {
                    # offset score to ensure a positive range as required by Elasticsearch
                    "source": f"{similarity_fn_name}(params.query_vector,'{self.embedding_field}') + 1000",
                    "params": {"query_vector": query_emb.tolist()},
                },
            }
        }

        return query

    @staticmethod
    def _get_raw_similarity_score(score):
        return score - 1000

    @staticmethod
    def _split_list(documents: List[dict], number_of_lists: int) -> Generator[List[dict], None, None]:
        chunk_size = max((len(documents) + 1) // number_of_lists, 1)
        for i in range(0, len(documents), chunk_size):
            yield documents[i : i + chunk_size]
