# pylint: disable=too-many-public-methods

from typing import List, Optional, Type, Union, Dict, Any, Generator

import json
import logging
import time
from copy import deepcopy
from string import Template

import numpy as np
from scipy.special import expit
from tqdm.auto import tqdm

try:
    from elasticsearch import Elasticsearch, RequestsHttpConnection, Connection, Urllib3HttpConnection
    from elasticsearch.helpers import bulk, scan
    from elasticsearch.exceptions import RequestError
except (ImportError, ModuleNotFoundError) as ie:
    from haystack.utils.import_utils import _optional_component_not_installed

    _optional_component_not_installed(__name__, "elasticsearch", ie)

from haystack.document_stores import KeywordDocumentStore
from haystack.schema import Document, Label
from haystack.document_stores.base import get_batches_from_generator
from haystack.document_stores.filter_utils import LogicalFilterClause
from haystack.errors import DocumentStoreError, HaystackError

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
        aws4auth=None,
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
        verify_certs: bool = True,
        recreate_index: bool = False,
        create_index: bool = True,
        refresh_type: str = "wait_for",
        similarity: str = "dot_product",
        timeout: int = 30,
        return_embedding: bool = False,
        duplicate_documents: str = "overwrite",
        index_type: str = "flat",
        scroll: str = "1d",
        skip_missing_embeddings: bool = True,
        synonyms: Optional[List] = None,
        synonym_type: str = "synonym",
        use_system_proxy: bool = False,
    ):
        """
        A DocumentStore using Elasticsearch to store and query the documents for our search.

            * Keeps all the logic to store and query documents from Elastic, incl. mapping of fields, adding filters or boosts to your queries, and storing embeddings
            * You can either use an existing Elasticsearch index or create a new one via haystack
            * Retrievers operate on top of this DocumentStore to find the relevant documents for a query

        :param host: url(s) of elasticsearch nodes
        :param port: port(s) of elasticsearch nodes
        :param username: username (standard authentication via http_auth)
        :param password: password (standard authentication via http_auth)
        :param api_key_id: ID of the API key (altenative authentication mode to the above http_auth)
        :param api_key: Secret value of the API key (altenative authentication mode to the above http_auth)
        :param aws4auth: Authentication for usage with aws elasticsearch (can be generated with the requests-aws4auth package)
        :param index: Name of index in elasticsearch to use for storing the documents that we want to search. If not existing yet, we will create one.
        :param label_index: Name of index in elasticsearch to use for storing labels. If not existing yet, we will create one.
        :param search_fields: Name of fields used by BM25Retriever to find matches in the docs to our incoming query (using elastic's multi_match query), e.g. ["title", "full_text"]
        :param content_field: Name of field that might contain the answer and will therefore be passed to the Reader Model (e.g. "full_text").
                           If no Reader is used (e.g. in FAQ-Style QA) the plain content of this field will just be returned.
        :param name_field: Name of field that contains the title of the the doc
        :param embedding_field: Name of field containing an embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
        :param embedding_dim: Dimensionality of embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
        :param custom_mapping: If you want to use your own custom mapping for creating a new index in Elasticsearch, you can supply it here as a dictionary.
        :param analyzer: Specify the default analyzer from one of the built-ins when creating a new Elasticsearch Index.
                         Elasticsearch also has built-in analyzers for different languages (e.g. impacting tokenization). More info at:
                         https://www.elastic.co/guide/en/elasticsearch/reference/7.9/analysis-analyzers.html
        :param excluded_meta_data: Name of fields in Elasticsearch that should not be returned (e.g. [field_one, field_two]).
                                   Helpful if you have fields with long, irrelevant content that you don't want to display in results (e.g. embedding vectors).
        :param scheme: 'https' or 'http', protocol used to connect to your elasticsearch instance
        :param ca_certs: Root certificates for SSL: it is a path to certificate authority (CA) certs on disk. You can use certifi package with certifi.where() to find where the CA certs file is located in your machine.
        :param verify_certs: Whether to be strict about ca certificates
        :param recreate_index: If set to True, an existing elasticsearch index will be deleted and a new one will be
            created using the config you are using for initialization. Be aware that all data in the old index will be
            lost if you choose to recreate the index. Be aware that both the document_index and the label_index will
            be recreated.
        :param create_index:
            Whether to try creating a new index (If the index of that name is already existing, we will just continue in any case)
            ..deprecated:: 2.0
                This param is deprecated. In the next major version we will always try to create an index if there is no
                existing index (the current behaviour when create_index=True). If you are looking to recreate an
                existing index by deleting it first if it already exist use param recreate_index.
        :param refresh_type: Type of ES refresh used to control when changes made by a request (e.g. bulk) are made visible to search.
                             If set to 'wait_for', continue only after changes are visible (slow, but safe).
                             If set to 'false', continue directly (fast, but sometimes unintuitive behaviour when docs are not immediately available after ingestion).
                             More info at https://www.elastic.co/guide/en/elasticsearch/reference/6.8/docs-refresh.html
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default since it is
                           more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence BERT model.
        :param timeout: Number of seconds after which an ElasticSearch request times out.
        :param return_embedding: To return document embedding
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param index_type: The type of index to be created. Choose from 'flat' and 'hnsw'. Currently the
                           ElasticsearchDocumentStore does not support HNSW but OpenDistroElasticsearchDocumentStore does.
        :param scroll: Determines how long the current index is fixed, e.g. during updating all documents with embeddings.
                       Defaults to "1d" and should not be larger than this. Can also be in minutes "5m" or hours "15h"
                       For details, see https://www.elastic.co/guide/en/elasticsearch/reference/current/scroll-api.html
        :param skip_missing_embeddings: Parameter to control queries based on vector similarity when indexed documents miss embeddings.
                                        Parameter options: (True, False)
                                        False: Raises exception if one or more documents do not have embeddings at query time
                                        True: Query will ignore all documents without embeddings (recommended if you concurrently index and query)
        :param synonyms: List of synonyms can be passed while elasticsearch initialization.
                         For example: [ "foo, bar => baz",
                                        "foozball , foosball" ]
                         More info at https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-tokenfilter.html
        :param synonym_type: Synonym filter type can be passed.
                             Synonym or Synonym_graph to handle synonyms, including multi-word synonyms correctly during the analysis process.
                             More info at https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-graph-tokenfilter.html
        :param use_system_proxy: Whether to use system proxy.

        """
        super().__init__()

        self.client = self._init_elastic_client(
            host=host,
            port=port,
            username=username,
            password=password,
            api_key=api_key,
            api_key_id=api_key_id,
            aws4auth=aws4auth,
            scheme=scheme,
            ca_certs=ca_certs,
            verify_certs=verify_certs,
            timeout=timeout,
            use_system_proxy=use_system_proxy,
        )

        # configure mappings to ES fields that will be used for querying / displaying results
        if type(search_fields) == str:
            search_fields = [search_fields]

        # TODO we should implement a more flexible interal mapping here that simplifies the usage of additional,
        # custom fields (e.g. meta data you want to return)
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
        if similarity in ["cosine", "dot_product", "l2"]:
            self.similarity: str = similarity
        else:
            raise DocumentStoreError(
                f"Invalid value {similarity} for similarity in ElasticSearchDocumentStore constructor. Choose between 'cosine', 'l2' and 'dot_product'"
            )
        if index_type in ["flat", "hnsw"]:
            self.index_type = index_type
        else:
            raise Exception("Invalid value for index_type in constructor. Choose between 'flat' and 'hnsw'")
        if index_type == "hnsw" and type(self) == ElasticsearchDocumentStore:
            raise Exception(
                "The HNSW algorithm for approximate nearest neighbours calculation is currently not available in the ElasticSearchDocumentStore. "
                "Try the OpenSearchDocumentStore instead."
            )
        if recreate_index:
            self._delete_index(index)
            self._delete_index(label_index)

        if create_index or recreate_index:
            self._create_document_index(index)
            self._create_label_index(label_index)

        self.duplicate_documents = duplicate_documents
        self.refresh_type = refresh_type

    @classmethod
    def _init_elastic_client(
        cls,
        host: Union[str, List[str]],
        port: Union[int, List[int]],
        username: str,
        password: str,
        api_key_id: Optional[str],
        api_key: Optional[str],
        aws4auth,
        scheme: str,
        ca_certs: Optional[str],
        verify_certs: bool,
        timeout: int,
        use_system_proxy: bool,
    ) -> Elasticsearch:

        hosts = cls._prepare_hosts(host, port)

        if (api_key or api_key_id) and not (api_key and api_key_id):
            raise ValueError("You must provide either both or none of `api_key_id` and `api_key`")

        connection_class: Type[Connection] = Urllib3HttpConnection
        if use_system_proxy:
            connection_class = RequestsHttpConnection

        if api_key:
            # api key authentication
            client = Elasticsearch(
                hosts=hosts,
                api_key=(api_key_id, api_key),
                scheme=scheme,
                ca_certs=ca_certs,
                verify_certs=verify_certs,
                timeout=timeout,
                connection_class=connection_class,
            )
        elif aws4auth:
            # aws elasticsearch with IAM
            # see https://elasticsearch-py.readthedocs.io/en/v7.12.0/index.html?highlight=http_auth#running-on-aws-with-iam
            client = Elasticsearch(
                hosts=hosts,
                http_auth=aws4auth,
                connection_class=RequestsHttpConnection,
                use_ssl=True,
                verify_certs=True,
                timeout=timeout,
            )
        elif username:
            # standard http_auth
            client = Elasticsearch(
                hosts=hosts,
                http_auth=(username, password),
                scheme=scheme,
                ca_certs=ca_certs,
                verify_certs=verify_certs,
                timeout=timeout,
                connection_class=connection_class,
            )
        else:
            # there is no authentication for this elasticsearch instance
            client = Elasticsearch(
                hosts=hosts,
                scheme=scheme,
                ca_certs=ca_certs,
                verify_certs=verify_certs,
                timeout=timeout,
                connection_class=connection_class,
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
                        f"at `{hosts}` and that it has finished the initial ramp up (can take > 30s)."
                    )
        except Exception:
            raise ConnectionError(
                f"Initial connection to Elasticsearch failed. Make sure you run an Elasticsearch instance at `{hosts}` and that it has finished the initial ramp up (can take > 30s)."
            )
        return client

    @staticmethod
    def _prepare_hosts(host, port):
        # Create list of host(s) + port(s) to allow direct client connections to multiple elasticsearch nodes
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

    def _create_document_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        """
        Create a new index for storing documents. In case if an index with the name already exists, it ensures that
        the embedding_field is present.
        """
        # Check if index_name refers to an alias
        if self.client.indices.exists_alias(name=index_name):
            logger.debug(f"Index name {index_name} is an alias.")

        # check if the existing index has the embedding field; if not create it
        if self.client.indices.exists(index=index_name, headers=headers):
            indices = self.client.indices.get(index_name, headers=headers)
            # If the index name is an alias that groups multiple existing indices, each of them must have an embedding_field.
            for index_id, index_info in indices.items():
                mapping = index_info["mappings"]
                if self.search_fields:
                    for search_field in self.search_fields:
                        if (
                            search_field in mapping["properties"]
                            and mapping["properties"][search_field]["type"] != "text"
                        ):
                            raise Exception(
                                f"The search_field '{search_field}' of index '{index_id}' with type '{mapping['properties'][search_field]['type']}' "
                                f"does not have the right type 'text' to be queried in fulltext search. Please use only 'text' type properties as search_fields or use another index. "
                                f"This error might occur if you are trying to use haystack 1.0 and above with an existing elasticsearch index created with a previous version of haystack. "
                                f'In this case deleting the index with `delete_index(index="{index_id}")` will fix your environment. '
                                f"Note, that all data stored in the index will be lost!"
                            )
                if self.embedding_field:
                    if (
                        self.embedding_field in mapping["properties"]
                        and mapping["properties"][self.embedding_field]["type"] != "dense_vector"
                    ):
                        raise Exception(
                            f"The '{index_id}' index in Elasticsearch already has a field called '{self.embedding_field}'"
                            f" with the type '{mapping['properties'][self.embedding_field]['type']}'. Please update the "
                            f"document_store to use a different name for the embedding_field parameter."
                        )
                    mapping["properties"][self.embedding_field] = {"type": "dense_vector", "dims": self.embedding_dim}
                    self.client.indices.put_mapping(index=index_id, body=mapping, headers=headers)
            return

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
            self.client.indices.create(index=index_name, body=mapping, headers=headers)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name, headers=headers):
                raise e

    def _create_label_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        if self.client.indices.exists(index=index_name, headers=headers):
            return
        mapping = {
            "mappings": {
                "properties": {
                    "query": {"type": "text"},
                    "answer": {"type": "flattened"},  # light-weight but less search options than full object
                    "document": {"type": "flattened"},
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
            self.client.indices.create(index=index_name, body=mapping, headers=headers)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name, headers=headers):
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
        Fetch documents by specifying a list of text id strings. Be aware that passing a large number of ids might lead
        to performance issues. Note that Elasticsearch limits the number of results to 10,000 documents by default.
        """
        index = index or self.index
        query = {"size": len(ids), "query": {"ids": {"values": ids}}}
        result = self.client.search(index=index, body=query, headers=headers)["hits"]["hits"]
        documents = [self._convert_es_hit_to_document(hit, return_embedding=self.return_embedding) for hit in result]
        return documents

    def get_metadata_values_by_key(
        self,
        key: str,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
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
        :param index: Elasticsearch index where the meta values should be searched. If not supplied,
                      self.index will be used.
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
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
        Indexes documents for later queries in Elasticsearch.

        Behaviour if a document with the same ID already exists in ElasticSearch:
        a) (Default) Throw Elastic's standard error message for duplicate IDs.
        b) If `self.update_existing_documents=True` for DocumentStore: Overwrite existing documents.
        (This is only relevant if you pass your own ID when initializing a `Document`.
        If don't set custom IDs for your Documents or just pass a list of dictionaries here,
        they will automatically get UUIDs assigned. See the `Document` class for details)

        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"content": "<the-actual-text>"}.
                          Optionally: Include meta data via {"content": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
                          Advanced: If you are using your own Elasticsearch mapping, the key names in the dictionary
                          should be changed to what you have set for self.content_field and self.name_field.
        :param index: Elasticsearch index where the documents should be indexed. If not supplied, self.index will be used.
        :param batch_size: Number of documents that are passed to Elasticsearch's bulk function at a time.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return: None
        """

        if index and not self.client.indices.exists(index=index, headers=headers):
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
                bulk(self.client, documents_to_index, request_timeout=300, refresh=self.refresh_type, headers=headers)
                documents_to_index = []

        if documents_to_index:
            bulk(self.client, documents_to_index, request_timeout=300, refresh=self.refresh_type, headers=headers)

    def write_labels(
        self,
        labels: Union[List[Label], List[dict]],
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: int = 10_000,
    ):
        """Write annotation labels into document store.

        :param labels: A list of Python dictionaries or a list of Haystack Label objects.
        :param index: Elasticsearch index where the labels should be stored. If not supplied, self.label_index will be used.
        :param batch_size: Number of labels that are passed to Elasticsearch's bulk function at a time.
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """
        index = index or self.label_index
        if index and not self.client.indices.exists(index=index, headers=headers):
            self._create_label_index(index, headers=headers)

        label_list: List[Label] = [Label.from_dict(label) if isinstance(label, dict) else label for label in labels]
        duplicate_ids: list = [label.id for label in self._get_duplicate_labels(label_list, index=index)]
        if len(duplicate_ids) > 0:
            logger.warning(
                f"Duplicate Label IDs: Inserting a Label whose id already exists in this document store."
                f" This will overwrite the old Label. Please make sure Label.id is a unique identifier of"
                f" the answer annotation and not the question."
                f" Problematic ids: {','.join(duplicate_ids)}"
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
                bulk(self.client, labels_to_index, request_timeout=300, refresh=self.refresh_type, headers=headers)
                labels_to_index = []

        if labels_to_index:
            bulk(self.client, labels_to_index, request_timeout=300, refresh=self.refresh_type, headers=headers)

    def update_document_meta(
        self, id: str, meta: Dict[str, str], headers: Optional[Dict[str, str]] = None, index: str = None
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
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
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
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
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
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
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
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
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
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
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
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        """

        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        result = self._get_all_documents_in_index(index=index, filters=filters, batch_size=batch_size, headers=headers)
        for hit in result:
            document = self._convert_es_hit_to_document(hit, return_embedding=return_embedding)
            yield document

    def get_all_labels(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
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
        labels = [Label.from_dict({**hit["_source"], "id": hit["_id"]}) for hit in result]
        return labels

    def _get_all_documents_in_index(
        self,
        index: str,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        batch_size: int = 10_000,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[dict, None, None]:
        """
        Return all documents in a specific index in the document store
        """
        body: dict = {"query": {"bool": {}}}

        if filters:
            body["query"]["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        if only_documents_without_embedding:
            body["query"]["bool"]["must_not"] = [{"exists": {"field": self.embedding_field}}]

        result = scan(self.client, query=body, index=index, size=batch_size, scroll=self.scroll, headers=headers)
        yield from result

    def query(
        self,
        query: Optional[str],
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
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

        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
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

        logger.debug(f"Retriever query: {body}")
        result = self.client.search(index=index, body=body, headers=headers)["hits"]["hits"]

        documents = [
            self._convert_es_hit_to_document(hit, return_embedding=self.return_embedding, scale_score=scale_score)
            for hit in result
        ]
        return documents

    def query_batch(
        self,
        queries: List[str],
        filters: Optional[
            Union[
                Dict[str, Union[Dict, List, str, int, float, bool]],
                List[Dict[str, Union[Dict, List, str, int, float, bool]]],
            ]
        ] = None,
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
            filters = [filters] * len(queries) if filters is not None else [{}] * len(queries)

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

        logger.debug(f"Retriever query: {body}")
        responses = self.client.msearch(index=index, body=body)

        all_documents = []
        cur_documents = []
        for response in responses["responses"]:
            cur_result = response["hits"]["hits"]
            cur_documents = [
                self._convert_es_hit_to_document(hit, return_embedding=self.return_embedding, scale_score=scale_score)
                for hit in cur_result
            ]
            all_documents.append(cur_documents)

        return all_documents

    def _construct_query_body(
        self,
        query: Optional[str],
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]],
        top_k: int,
        custom_query: Optional[str],
        all_terms_must_match: bool,
    ) -> Dict[str, Any]:

        # Naive retrieval without BM25, only filtering
        if query is None:
            body = {"query": {"bool": {"must": {"match_all": {}}}}}  # type: Dict[str, Any]
            if filters:
                body["query"]["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        # Retrieval via custom query
        elif custom_query:  # substitute placeholder for query and filters for the custom_query template string
            template = Template(custom_query)
            # replace all "${query}" placeholder(s) with query
            substitutions = {"query": f'"{query}"'}
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
                    f"of type {type(query)}. This can cause Elasticsearch to fail."
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

        if self.excluded_meta_data:
            body["_source"] = {"excludes": self.excluded_meta_data}

        return body

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
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

        # +1 in similarity to avoid negative numbers (for cosine sim)
        body = {"size": top_k, "query": self._get_vector_similarity_query(query_emb, top_k)}
        if filters:
            filter_ = {"bool": {"filter": LogicalFilterClause.parse(filters).convert_to_elasticsearch()}}
            if body["query"]["script_score"]["query"] == {"match_all": {}}:
                body["query"]["script_score"]["query"] = filter_
            else:
                body["query"]["script_score"]["query"]["bool"]["filter"]["bool"]["must"].append(filter_)

        excluded_meta_data: Optional[list] = None

        if self.excluded_meta_data:
            excluded_meta_data = deepcopy(self.excluded_meta_data)

            if return_embedding is True and self.embedding_field in excluded_meta_data:
                excluded_meta_data.remove(self.embedding_field)
            elif return_embedding is False and self.embedding_field not in excluded_meta_data:
                excluded_meta_data.append(self.embedding_field)
        elif return_embedding is False:
            excluded_meta_data = [self.embedding_field]

        if excluded_meta_data:
            body["_source"] = {"excludes": excluded_meta_data}

        logger.debug(f"Retriever query: {body}")
        try:
            result = self.client.search(index=index, body=body, request_timeout=300, headers=headers)["hits"]["hits"]
            if len(result) == 0:
                count_embeddings = self.get_embedding_count(index=index, headers=headers)
                if count_embeddings == 0:
                    raise RequestError(
                        400, "search_phase_execution_exception", {"error": "No documents with embeddings."}
                    )
        except RequestError as e:
            if e.error == "search_phase_execution_exception":
                error_message: str = (
                    "search_phase_execution_exception: Likely some of your stored documents don't have embeddings."
                    " Run the document store's update_embeddings() method."
                )
                raise RequestError(e.status_code, error_message, e.info)
            raise e

        documents = [
            self._convert_es_hit_to_document(
                hit, adapt_score_for_embedding=True, return_embedding=return_embedding, scale_score=scale_score
            )
            for hit in result
        ]
        return documents

    def _get_vector_similarity_query(self, query_emb: np.ndarray, top_k: int):
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
            raise Exception(
                "Invalid value for similarity in ElasticSearchDocumentStore constructor. Choose between 'cosine', 'dot_product' and 'l2'"
            )

        # To handle scenarios where embeddings may be missing
        script_score_query: dict = {"match_all": {}}
        if self.skip_missing_embeddings:
            script_score_query = {"bool": {"filter": {"bool": {"must": [{"exists": {"field": self.embedding_field}}]}}}}

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

    def _convert_es_hit_to_document(
        self, hit: dict, return_embedding: bool, adapt_score_for_embedding: bool = False, scale_score: bool = True
    ) -> Document:
        # We put all additional data of the doc into meta_data and return it in the API
        meta_data = {
            k: v
            for k, v in hit["_source"].items()
            if k not in (self.content_field, "content_type", self.embedding_field)
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
        if return_embedding:
            embedding_list = hit["_source"].get(self.embedding_field)
            if embedding_list:
                embedding = np.asarray(embedding_list, dtype=np.float32)

        doc_dict = {
            "id": hit["_id"],
            "content": hit["_source"].get(self.content_field),
            "content_type": hit["_source"].get("content_type", None),
            "meta": meta_data,
            "score": score,
            "embedding": embedding,
        }
        document = Document.from_dict(doc_dict)

        return document

    def _get_raw_similarity_score(self, score):
        return score - 1000

    def update_embeddings(
        self,
        retriever,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
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
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :return: None
        """
        if index is None:
            index = self.index

        if self.refresh_type == "false":
            self.client.indices.refresh(index=index, headers=headers)

        if not self.embedding_field:
            raise RuntimeError("Specify the arg `embedding_field` when initializing ElasticsearchDocumentStore()")

        if update_existing_embeddings:
            document_count = self.get_document_count(index=index, headers=headers)
            logger.info(f"Updating embeddings for all {document_count} docs ...")
        else:
            document_count = self.get_document_count(
                index=index, filters=filters, only_documents_without_embedding=True, headers=headers
            )
            logger.info(f"Updating embeddings for {document_count} docs without embeddings ...")

        result = self._get_all_documents_in_index(
            index=index,
            filters=filters,
            batch_size=batch_size,
            only_documents_without_embedding=not update_existing_embeddings,
            headers=headers,
        )

        logging.getLogger("elasticsearch").setLevel(logging.CRITICAL)

        with tqdm(total=document_count, position=0, unit=" Docs", desc="Updating embeddings") as progress_bar:
            for result_batch in get_batches_from_generator(result, batch_size):
                document_batch = [self._convert_es_hit_to_document(hit, return_embedding=False) for hit in result_batch]
                embeddings = retriever.embed_documents(document_batch)  # type: ignore
                assert len(document_batch) == len(embeddings)

                if embeddings[0].shape[0] != self.embedding_dim:
                    raise RuntimeError(
                        f"Embedding dim. of model ({embeddings[0].shape[0]})"
                        f" doesn't match embedding dim. in DocumentStore ({self.embedding_dim})."
                        "Specify the arg `embedding_dim` when initializing ElasticsearchDocumentStore()"
                    )
                doc_updates = []
                for doc, emb in zip(document_batch, embeddings):
                    update = {
                        "_op_type": "update",
                        "_index": index,
                        "_id": doc.id,
                        "doc": {self.embedding_field: emb.tolist()},
                    }
                    doc_updates.append(update)

                bulk(self.client, doc_updates, request_timeout=300, refresh=self.refresh_type, headers=headers)
                progress_bar.update(batch_size)

    def delete_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
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
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
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
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
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
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
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
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
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
        :param headers: Custom HTTP headers to pass to elasticsearch client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                Check out https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html for more information.
        :return: None
        """
        index = index or self.label_index
        self.delete_documents(index=index, ids=ids, filters=filters, headers=headers)

    def delete_index(self, index: str):
        """
        Delete an existing elasticsearch index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        if index == self.index:
            logger.warning(
                f"Deletion of default index '{index}' detected. "
                f"If you plan to use this index again, please reinstantiate '{self.__class__.__name__}' in order to avoid side-effects."
            )
        self._delete_index(index)

    def _delete_index(self, index: str):
        if self.client.indices.exists(index):
            self.client.indices.delete(index=index, ignore=[400, 404])
            logger.info(f"Index '{index}' deleted.")


class OpenSearchDocumentStore(ElasticsearchDocumentStore):
    def __init__(
        self,
        scheme: str = "https",  # Mind this different default param
        username: str = "admin",  # Mind this different default param
        password: str = "admin",  # Mind this different default param
        host: Union[str, List[str]] = "localhost",
        port: Union[int, List[int]] = 9200,
        api_key_id: Optional[str] = None,
        api_key: Optional[str] = None,
        aws4auth=None,
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
        ca_certs: Optional[str] = None,
        verify_certs: bool = False,  # Mind this different default param
        recreate_index: bool = False,
        create_index: bool = True,
        refresh_type: str = "wait_for",
        similarity: str = "dot_product",
        timeout: int = 30,
        return_embedding: bool = False,
        duplicate_documents: str = "overwrite",
        index_type: str = "flat",
        scroll: str = "1d",
        skip_missing_embeddings: bool = True,
        synonyms: Optional[List] = None,
        synonym_type: str = "synonym",
        use_system_proxy: bool = False,
    ):
        """
        Document Store using OpenSearch (https://opensearch.org/). It is compatible with the AWS Elasticsearch Service.

        In addition to native Elasticsearch query & filtering, it provides efficient vector similarity search using
        the KNN plugin that can scale to a large number of documents.

        :param host: url(s) of elasticsearch nodes
        :param port: port(s) of elasticsearch nodes
        :param username: username (standard authentication via http_auth)
        :param password: password (standard authentication via http_auth)
        :param api_key_id: ID of the API key (altenative authentication mode to the above http_auth)
        :param api_key: Secret value of the API key (altenative authentication mode to the above http_auth)
        :param aws4auth: Authentication for usage with aws elasticsearch (can be generated with the requests-aws4auth package)
        :param index: Name of index in elasticsearch to use for storing the documents that we want to search. If not existing yet, we will create one.
        :param label_index: Name of index in elasticsearch to use for storing labels. If not existing yet, we will create one.
        :param search_fields: Name of fields used by BM25Retriever to find matches in the docs to our incoming query (using elastic's multi_match query), e.g. ["title", "full_text"]
        :param content_field: Name of field that might contain the answer and will therefore be passed to the Reader Model (e.g. "full_text").
                           If no Reader is used (e.g. in FAQ-Style QA) the plain content of this field will just be returned.
        :param name_field: Name of field that contains the title of the the doc
        :param embedding_field: Name of field containing an embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
                                Note, that in OpenSearch the similarity type for efficient approximate vector similarity calculations is tied to the embedding field's data type which cannot be changed after creation.
        :param embedding_dim: Dimensionality of embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
        :param custom_mapping: If you want to use your own custom mapping for creating a new index in Elasticsearch, you can supply it here as a dictionary.
        :param analyzer: Specify the default analyzer from one of the built-ins when creating a new Elasticsearch Index.
                         Elasticsearch also has built-in analyzers for different languages (e.g. impacting tokenization). More info at:
                         https://www.elastic.co/guide/en/elasticsearch/reference/7.9/analysis-analyzers.html
        :param excluded_meta_data: Name of fields in Elasticsearch that should not be returned (e.g. [field_one, field_two]).
                                   Helpful if you have fields with long, irrelevant content that you don't want to display in results (e.g. embedding vectors).
        :param scheme: 'https' or 'http', protocol used to connect to your elasticsearch instance
        :param ca_certs: Root certificates for SSL: it is a path to certificate authority (CA) certs on disk. You can use certifi package with certifi.where() to find where the CA certs file is located in your machine.
        :param verify_certs: Whether to be strict about ca certificates
        :param create_index: Whether to try creating a new index (If the index of that name is already existing, we will just continue in any case
        :param refresh_type: Type of ES refresh used to control when changes made by a request (e.g. bulk) are made visible to search.
                             If set to 'wait_for', continue only after changes are visible (slow, but safe).
                             If set to 'false', continue directly (fast, but sometimes unintuitive behaviour when docs are not immediately available after ingestion).
                             More info at https://www.elastic.co/guide/en/elasticsearch/reference/6.8/docs-refresh.html
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default since it is
                           more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence BERT model.
                           Note, that the use of efficient approximate vector calculations in OpenSearch is tied to embedding_field's data type which cannot be changed after creation.
                           You won't be able to use approximate vector calculations on an embedding_field which was created with a different similarity value.
                           In such cases a fallback to exact but slow vector calculations will happen and a warning will be displayed.
        :param timeout: Number of seconds after which an ElasticSearch request times out.
        :param return_embedding: To return document embedding
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param index_type: The type of index to be created. Choose from 'flat' and 'hnsw'.
                           As OpenSearch currently does not support all similarity functions (e.g. dot_product) in exact vector similarity calculations,
                           we don't make use of exact vector similarity when index_type='flat'. Instead we use the same approximate vector similarity calculations like in 'hnsw', but further optimized for accuracy.
                           Exact vector similarity is only used as fallback when there's a mismatch between certain requested and indexed similarity types.
                           In these cases however, a warning will be displayed. See similarity param for more information.
        :param scroll: Determines how long the current index is fixed, e.g. during updating all documents with embeddings.
                       Defaults to "1d" and should not be larger than this. Can also be in minutes "5m" or hours "15h"
                       For details, see https://www.elastic.co/guide/en/elasticsearch/reference/current/scroll-api.html
        :param skip_missing_embeddings: Parameter to control queries based on vector similarity when indexed documents miss embeddings.
                                        Parameter options: (True, False)
                                        False: Raises exception if one or more documents do not have embeddings at query time
                                        True: Query will ignore all documents without embeddings (recommended if you concurrently index and query)
        :param synonyms: List of synonyms can be passed while elasticsearch initialization.
                         For example: [ "foo, bar => baz",
                                        "foozball , foosball" ]
                         More info at https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-tokenfilter.html
        :param synonym_type: Synonym filter type can be passed.
                             Synonym or Synonym_graph to handle synonyms, including multi-word synonyms correctly during the analysis process.
                             More info at https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-graph-tokenfilter.html
        """
        self.embeddings_field_supports_similarity = False
        self.similarity_to_space_type = {"cosine": "cosinesimil", "dot_product": "innerproduct", "l2": "l2"}
        self.space_type_to_similarity = {v: k for k, v in self.similarity_to_space_type.items()}
        super().__init__(
            scheme=scheme,
            username=username,
            password=password,
            host=host,
            port=port,
            api_key_id=api_key_id,
            api_key=api_key,
            aws4auth=aws4auth,
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
            ca_certs=ca_certs,
            verify_certs=verify_certs,
            recreate_index=recreate_index,
            create_index=create_index,
            refresh_type=refresh_type,
            similarity=similarity,
            timeout=timeout,
            return_embedding=return_embedding,
            duplicate_documents=duplicate_documents,
            index_type=index_type,
            scroll=scroll,
            skip_missing_embeddings=skip_missing_embeddings,
            synonyms=synonyms,
            synonym_type=synonym_type,
            use_system_proxy=use_system_proxy,
        )

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
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
        # +1 in similarity to avoid negative numbers (for cosine sim)
        body: Dict[str, Any] = {"size": top_k, "query": self._get_vector_similarity_query(query_emb, top_k)}
        if filters:
            body["query"]["bool"]["filter"] = LogicalFilterClause.parse(filters).convert_to_elasticsearch()

        excluded_meta_data: Optional[list] = None

        if self.excluded_meta_data:
            excluded_meta_data = deepcopy(self.excluded_meta_data)

            if return_embedding is True and self.embedding_field in excluded_meta_data:
                excluded_meta_data.remove(self.embedding_field)
            elif return_embedding is False and self.embedding_field not in excluded_meta_data:
                excluded_meta_data.append(self.embedding_field)
        elif return_embedding is False:
            excluded_meta_data = [self.embedding_field]

        if excluded_meta_data:
            body["_source"] = {"excludes": excluded_meta_data}

        logger.debug(f"Retriever query: {body}")
        result = self.client.search(index=index, body=body, request_timeout=300, headers=headers)["hits"]["hits"]

        documents = [
            self._convert_es_hit_to_document(
                hit, adapt_score_for_embedding=True, return_embedding=return_embedding, scale_score=scale_score
            )
            for hit in result
        ]
        return documents

    def _create_document_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        """
        Create a new index for storing documents.
        """
        # Check if index_name refers to an alias
        if self.client.indices.exists_alias(name=index_name):
            logger.debug(f"Index name {index_name} is an alias.")

        # check if the existing index has the embedding field; if not create it
        if self.client.indices.exists(index=index_name, headers=headers):
            indices = self.client.indices.get(index_name, headers=headers)
            # If the index name is an alias that groups multiple existing indices, each of them must have an embedding_field.
            for index_id, index_info in indices.items():
                mappings = index_info["mappings"]
                index_settings = index_info["settings"]["index"]
                if self.search_fields:
                    for search_field in self.search_fields:
                        if (
                            search_field in mappings["properties"]
                            and mappings["properties"][search_field]["type"] != "text"
                        ):
                            raise Exception(
                                f"The search_field '{search_field}' of index '{index_id}' with type '{mappings['properties'][search_field]['type']}' "
                                f"does not have the right type 'text' to be queried in fulltext search. Please use only 'text' type properties as search_fields or use another index. "
                                f"This error might occur if you are trying to use haystack 1.0 and above with an existing elasticsearch index created with a previous version of haystack. "
                                f'In this case deleting the index with `delete_index(index="{index_id}")` will fix your environment. '
                                f"Note, that all data stored in the index will be lost!"
                            )

                # embedding field will be created
                if self.embedding_field not in mappings["properties"]:
                    mappings["properties"][self.embedding_field] = self._get_embedding_field_mapping(
                        similarity=self.similarity
                    )
                    self.client.indices.put_mapping(index=index_id, body=mappings, headers=headers)
                    self.embeddings_field_supports_similarity = True
                else:
                    # bad embedding field
                    if mappings["properties"][self.embedding_field]["type"] != "knn_vector":
                        raise Exception(
                            f"The '{index_id}' index in OpenSearch already has a field called '{self.embedding_field}'"
                            f" with the type '{mappings['properties'][self.embedding_field]['type']}'. Please update the "
                            f"document_store to use a different name for the embedding_field parameter."
                        )
                    # embedding field with global space_type setting
                    if "method" not in mappings["properties"][self.embedding_field]:
                        embedding_field_space_type = index_settings["knn.space_type"]
                    # embedding field with local space_type setting
                    else:
                        # embedding field with global space_type setting
                        if "method" not in mappings["properties"][self.embedding_field]:
                            embedding_field_space_type = index_settings["knn.space_type"]
                        # embedding field with local space_type setting
                        else:
                            embedding_field_space_type = mappings["properties"][self.embedding_field]["method"][
                                "space_type"
                            ]

                        embedding_field_similarity = self.space_type_to_similarity[embedding_field_space_type]
                        if embedding_field_similarity == self.similarity:
                            self.embeddings_field_supports_similarity = True
                        else:
                            logger.warning(
                                f"Embedding field '{self.embedding_field}' is optimized for similarity '{embedding_field_similarity}'. "
                                f"Falling back to slow exact vector calculation. "
                                f"Consider cloning the embedding field optimized for '{embedding_field_similarity}' by calling clone_embedding_field(similarity='{embedding_field_similarity}', ...) "
                                f"or creating a new index optimized for '{self.similarity}' by setting `similarity='{self.similarity}'` the first time you instantiate OpenSearchDocumentStore for the new index, "
                                f"e.g. `OpenSearchDocumentStore(index='my_new_{self.similarity}_index', similarity='{self.similarity}')`."
                            )

                # Adjust global ef_search setting. If not set, default is 512.
                ef_search = index_settings.get("knn.algo_param", {"ef_search": 512}).get("ef_search", 512)
                if self.index_type == "hnsw" and ef_search != 20:
                    body = {"knn.algo_param.ef_search": 20}
                    self.client.indices.put_settings(index=index_id, body=body, headers=headers)
                elif self.index_type == "flat" and ef_search != 512:
                    body = {"knn.algo_param.ef_search": 512}
                    self.client.indices.put_settings(index=index_id, body=body, headers=headers)

            return

        if self.custom_mapping:
            index_definition = self.custom_mapping
        else:
            index_definition = {
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
                    index_definition["mappings"]["properties"].update({field: {"type": "text", "analyzer": "synonym"}})
                index_definition["mappings"]["properties"][self.content_field] = {"type": "text", "analyzer": "synonym"}

                index_definition["settings"]["analysis"]["analyzer"]["synonym"] = {
                    "tokenizer": "whitespace",
                    "filter": ["lowercase", "synonym"],
                }
                index_definition["settings"]["analysis"]["filter"] = {
                    "synonym": {"type": self.synonym_type, "synonyms": self.synonyms}
                }

            else:
                for field in self.search_fields:
                    index_definition["mappings"]["properties"].update({field: {"type": "text"}})

            if self.embedding_field:
                index_definition["settings"]["index"] = {"knn": True}
                if self.index_type == "hnsw":
                    index_definition["settings"]["index"]["knn.algo_param.ef_search"] = 20
                index_definition["mappings"]["properties"][self.embedding_field] = self._get_embedding_field_mapping(
                    similarity=self.similarity
                )

        try:
            self.client.indices.create(index=index_name, body=index_definition, headers=headers)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name, headers=headers):
                raise e

    def _get_embedding_field_mapping(self, similarity: str):
        space_type = self.similarity_to_space_type[similarity]
        method: dict = {"space_type": space_type, "name": "hnsw", "engine": "nmslib"}

        if self.index_type == "flat":
            # use default parameters from https://opensearch.org/docs/1.2/search-plugins/knn/knn-index/
            # we need to set them explicitly as aws managed instances starting from version 1.2 do not support empty parameters
            method["parameters"] = {"ef_construction": 512, "m": 16}
        elif self.index_type == "hnsw":
            method["parameters"] = {"ef_construction": 80, "m": 64}
        else:
            logger.error("Please set index_type to either 'flat' or 'hnsw'")

        embeddings_field_mapping = {"type": "knn_vector", "dimension": self.embedding_dim, "method": method}
        return embeddings_field_mapping

    def _create_label_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        if self.client.indices.exists(index=index_name, headers=headers):
            return
        mapping = {
            "mappings": {
                "properties": {
                    "query": {"type": "text"},
                    "answer": {
                        "type": "nested"
                    },  # In elasticsearch we use type:flattened, but this is not supported in opensearch
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
            self.client.indices.create(index=index_name, body=mapping, headers=headers)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name, headers=headers):
                raise e

    def _get_vector_similarity_query(self, query_emb: np.ndarray, top_k: int):
        """
        Generate Elasticsearch query for vector similarity.
        """
        if self.embeddings_field_supports_similarity:
            query: dict = {
                "bool": {"must": [{"knn": {self.embedding_field: {"vector": query_emb.tolist(), "k": top_k}}}]}
            }
        else:
            # if we do not have a proper similarity field we have to fall back to exact but slow vector similarity calculation
            query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "knn_score",
                        "lang": "knn",
                        "params": {
                            "field": self.embedding_field,
                            "query_value": query_emb.tolist(),
                            "space_type": self.similarity_to_space_type[self.similarity],
                        },
                    },
                }
            }
        return query

    def _get_raw_similarity_score(self, score):
        # adjust scores according to https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn
        # and https://opensearch.org/docs/latest/search-plugins/knn/knn-score-script/
        if self.similarity == "dot_product":
            if score > 1:
                score = score - 1
            else:
                score = -(1 / score - 1)
        elif self.similarity == "l2":
            score = 1 / score - 1
        elif self.similarity == "cosine":
            if self.embeddings_field_supports_similarity:
                score = -(1 / score - 2)
            else:
                score = score - 1

        return score

    def clone_embedding_field(
        self,
        new_embedding_field: str,
        similarity: str,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ):
        mapping = self.client.indices.get(self.index, headers=headers)[self.index]["mappings"]
        if new_embedding_field in mapping["properties"]:
            raise Exception(
                f"{new_embedding_field} already exists with mapping {mapping['properties'][new_embedding_field]}"
            )
        mapping["properties"][new_embedding_field] = self._get_embedding_field_mapping(similarity=similarity)
        self.client.indices.put_mapping(index=self.index, body=mapping, headers=headers)

        document_count = self.get_document_count(headers=headers)
        result = self._get_all_documents_in_index(index=self.index, batch_size=batch_size, headers=headers)

        logging.getLogger("elasticsearch").setLevel(logging.CRITICAL)

        with tqdm(total=document_count, position=0, unit=" Docs", desc="Cloning embeddings") as progress_bar:
            for result_batch in get_batches_from_generator(result, batch_size):
                document_batch = [self._convert_es_hit_to_document(hit, return_embedding=True) for hit in result_batch]
                doc_updates = []
                for doc in document_batch:
                    if doc.embedding is not None:
                        update = {
                            "_op_type": "update",
                            "_index": self.index,
                            "_id": doc.id,
                            "doc": {new_embedding_field: doc.embedding.tolist()},
                        }
                        doc_updates.append(update)

                bulk(self.client, doc_updates, request_timeout=300, refresh=self.refresh_type, headers=headers)
                progress_bar.update(batch_size)


class OpenDistroElasticsearchDocumentStore(OpenSearchDocumentStore):
    """
    A DocumentStore which has an Open Distro for Elasticsearch service behind it.
    """

    def __init__(
        self,
        scheme: str = "https",
        username: str = "admin",
        password: str = "admin",
        host: Union[str, List[str]] = "localhost",
        port: Union[int, List[int]] = 9200,
        api_key_id: Optional[str] = None,
        api_key: Optional[str] = None,
        aws4auth=None,
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
        ca_certs: Optional[str] = None,
        verify_certs: bool = False,
        recreate_index: bool = False,
        create_index: bool = True,
        refresh_type: str = "wait_for",
        similarity: str = "cosine",  # Mind this different default param
        timeout: int = 30,
        return_embedding: bool = False,
        duplicate_documents: str = "overwrite",
        index_type: str = "flat",
        scroll: str = "1d",
        skip_missing_embeddings: bool = True,
        synonyms: Optional[List] = None,
        synonym_type: str = "synonym",
        use_system_proxy: bool = False,
    ):
        logger.warning(
            "Open Distro for Elasticsearch has been replaced by OpenSearch! "
            "See https://opensearch.org/faq/ for details. "
            "We recommend using the OpenSearchDocumentStore instead."
        )
        super().__init__(
            scheme=scheme,
            username=username,
            password=password,
            host=host,
            port=port,
            api_key_id=api_key_id,
            api_key=api_key,
            aws4auth=aws4auth,
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
            ca_certs=ca_certs,
            verify_certs=verify_certs,
            recreate_index=recreate_index,
            create_index=create_index,
            refresh_type=refresh_type,
            similarity=similarity,
            timeout=timeout,
            return_embedding=return_embedding,
            duplicate_documents=duplicate_documents,
            index_type=index_type,
            scroll=scroll,
            skip_missing_embeddings=skip_missing_embeddings,
            synonyms=synonyms,
            synonym_type=synonym_type,
            use_system_proxy=use_system_proxy,
        )
