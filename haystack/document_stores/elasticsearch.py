from typing import List, Optional, Union, Dict, Any, Generator

import json
import logging
import time
from copy import deepcopy
from string import Template
import numpy as np
from scipy.special import expit
from tqdm.auto import tqdm
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch.helpers import bulk, scan
from elasticsearch.exceptions import RequestError
import pandas as pd

from haystack.document_stores import BaseDocumentStore
from haystack.schema import Document, Label
from haystack.document_stores.base import get_batches_from_generator


logger = logging.getLogger(__name__)


class ElasticsearchDocumentStore(BaseDocumentStore):
    def __init__(
        self,
        host: Union[str, List[str]] = "localhost",
        port: Union[int, List[int]] = 9200,
        username: str = "",
        password: str = "",
        api_key_id: Optional[str] = None,
        api_key: Optional[str] = None,
        aws4auth = None,
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
        create_index: bool = True,
        refresh_type: str = "wait_for",
        similarity="dot_product",
        timeout=30,
        return_embedding: bool = False,
        duplicate_documents: str = 'overwrite',
        index_type: str = "flat",
        scroll: str = "1d",
        skip_missing_embeddings: bool = True,
        synonyms: Optional[List] = None,
        synonym_type: str = "synonym"
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
        :param search_fields: Name of fields used by ElasticsearchRetriever to find matches in the docs to our incoming query (using elastic's multi_match query), e.g. ["title", "full_text"]
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
        :param create_index: Whether to try creating a new index (If the index of that name is already existing, we will just continue in any case
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

        """
        # save init parameters to enable export of component config as YAML
        self.set_config(
            host=host, port=port, username=username, password=password, api_key_id=api_key_id, api_key=api_key,
            aws4auth=aws4auth, index=index, label_index=label_index, search_fields=search_fields, content_field=content_field,
            name_field=name_field, embedding_field=embedding_field, embedding_dim=embedding_dim,
            custom_mapping=custom_mapping, excluded_meta_data=excluded_meta_data, analyzer=analyzer, scheme=scheme,
            ca_certs=ca_certs, verify_certs=verify_certs, create_index=create_index,
            duplicate_documents=duplicate_documents, refresh_type=refresh_type, similarity=similarity,
            timeout=timeout, return_embedding=return_embedding, index_type=index_type, scroll=scroll,
            skip_missing_embeddings=skip_missing_embeddings, synonyms=synonyms,synonym_type=synonym_type
        )

        self.client = self._init_elastic_client(host=host, port=port, username=username, password=password,
                                           api_key=api_key, api_key_id=api_key_id, aws4auth=aws4auth, scheme=scheme,
                                           ca_certs=ca_certs, verify_certs=verify_certs,timeout=timeout)

        # configure mappings to ES fields that will be used for querying / displaying results
        if type(search_fields) == str:
            search_fields = [search_fields]

        #TODO we should implement a more flexible interal mapping here that simplifies the usage of additional,
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
            self.similarity = similarity
        else:
            raise Exception(f"Invalid value {similarity} for similarity in ElasticSearchDocumentStore constructor. Choose between 'cosine', 'l2' and 'dot_product'")
        if index_type in ["flat", "hnsw"]:
            self.index_type = index_type
        else:
            raise Exception("Invalid value for index_type in constructor. Choose between 'flat' and 'hnsw'")
        if index_type == "hnsw" and type(self) == ElasticsearchDocumentStore:
            raise Exception("The HNSW algorithm for approximate nearest neighbours calculation is currently not available in the ElasticSearchDocumentStore. "
                            "Try the OpenSearchDocumentStore instead.")
        if create_index:
            self._create_document_index(index)
            self._create_label_index(label_index)

        self.duplicate_documents = duplicate_documents
        self.refresh_type = refresh_type


    def _init_elastic_client(self,
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
                             timeout: int) -> Elasticsearch:

        hosts = self._prepare_hosts(host, port)

        if (api_key or api_key_id) and not(api_key and api_key_id):
            raise ValueError("You must provide either both or none of `api_key_id` and `api_key`")

        if api_key:
            # api key authentication
            client = Elasticsearch(hosts=hosts, api_key=(api_key_id, api_key),
                                        scheme=scheme, ca_certs=ca_certs, verify_certs=verify_certs, timeout=timeout)
        elif aws4auth:
            # aws elasticsearch with IAM
            # see https://elasticsearch-py.readthedocs.io/en/v7.12.0/index.html?highlight=http_auth#running-on-aws-with-iam
            client = Elasticsearch(
                hosts=hosts, http_auth=aws4auth, connection_class=RequestsHttpConnection, use_ssl=True, verify_certs=True, timeout=timeout)
        elif username:
            # standard http_auth
            client = Elasticsearch(hosts=hosts, http_auth=(username, password),
                                        scheme=scheme, ca_certs=ca_certs, verify_certs=verify_certs,
                                        timeout=timeout)
        else:
            # there is no authentication for this elasticsearch instance
            client = Elasticsearch(hosts=hosts, scheme=scheme, ca_certs=ca_certs, verify_certs=verify_certs,
                                        timeout=timeout)

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
                f"Initial connection to Elasticsearch failed. Make sure you run an Elasticsearch instance at `{hosts}` and that it has finished the initial ramp up (can take > 30s).")
        return client

    def _prepare_hosts(self, host, port):
        # Create list of host(s) + port(s) to allow direct client connections to multiple elasticsearch nodes
        if isinstance(host, list):
            if isinstance(port, list):
                if not len(port) == len(host):
                    raise ValueError("Length of list `host` must match length of list `port`")
                hosts = [{"host":h, "port":p} for h, p in zip(host,port)]
            else:
                hosts = [{"host": h, "port": port} for h in host]
        else:
            hosts = [{"host": host, "port": port}]
        return hosts

    def _create_document_index(self, index_name: str):
        """
        Create a new index for storing documents. In case if an index with the name already exists, it ensures that
        the embedding_field is present.
        """
        # check if the existing index has the embedding field; if not create it
        if self.client.indices.exists(index=index_name):
            if self.embedding_field:
                mapping = self.client.indices.get(index_name)[index_name]["mappings"]
                if self.embedding_field in mapping["properties"] and mapping["properties"][self.embedding_field]["type"] != "dense_vector":
                    raise Exception(f"The '{index_name}' index in Elasticsearch already has a field called '{self.embedding_field}'"
                                    f" with the type '{mapping['properties'][self.embedding_field]['type']}'. Please update the "
                                    f"document_store to use a different name for the embedding_field parameter.")
                mapping["properties"][self.embedding_field] = {"type": "dense_vector", "dims": self.embedding_dim}
                self.client.indices.put_mapping(index=index_name, body=mapping)
            return

        if self.custom_mapping:
            mapping = self.custom_mapping
        else:
            mapping = {
                "mappings": {
                    "properties": {
                        self.name_field: {"type": "keyword"},
                        self.content_field: {"type": "text"},
                    },
                    "dynamic_templates": [
                        {
                            "strings": {
                                "path_match": "*",
                                "match_mapping_type": "string",
                                "mapping": {"type": "keyword"}}}
                    ],
                },
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": self.analyzer,
                            }
                        }
                    }
                }
            }
            if self.synonyms:
                mapping["mappings"]["properties"][self.content_field] = {"type": "text", "analyzer": "synonym"}
                mapping["settings"]["analysis"]["analyzer"]["synonym"] = {"tokenizer": "whitespace",
                                                                          "filter": ["lowercase",
                                                                                     "synonym"]}
                mapping["settings"]["analysis"]["filter"] = {"synonym": {"type": self.synonym_type, "synonyms": self.synonyms}}

            if self.embedding_field:
                mapping["mappings"]["properties"][self.embedding_field] = {"type": "dense_vector", "dims": self.embedding_dim}

        try:
            self.client.indices.create(index=index_name, body=mapping)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name):
                raise e

    def _create_label_index(self, index_name: str):
        if self.client.indices.exists(index=index_name):
            return
        mapping = {
            "mappings": {
                "properties": {
                    "query": {"type": "text"},
                    "answer": {"type": "flattened"}, #light-weight but less search options than full object
                    "document": {"type": "flattened"},
                    "is_correct_answer": {"type": "boolean"},
                    "is_correct_document": {"type": "boolean"},
                    "origin": {"type": "keyword"},  # e.g. user-feedback or gold-label
                    "document_id": {"type": "keyword"},
                    "no_answer": {"type": "boolean"},
                    "pipeline_id": {"type": "keyword"},
                    "created_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"},
                    "updated_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"}
                    #TODO add pipeline_hash and pipeline_name once we migrated the REST API to pipelines
                }
            }
        }
        try:
            self.client.indices.create(index=index_name, body=mapping)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name):
                raise e

    # TODO: Add flexibility to define other non-meta and meta fields expected by the Document class
    def _create_document_field_map(self) -> Dict:
        return {
            self.content_field: "content",
            self.embedding_field: "embedding"
        }

    def get_document_by_id(self, id: str, index: Optional[str] = None) -> Optional[Document]:
        """Fetch a document by specifying its text id string"""
        index = index or self.index
        documents = self.get_documents_by_id([id], index=index)
        if documents:
            return documents[0]
        else:
            return None

    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None) -> List[Document]:  # type: ignore
        """Fetch documents by specifying a list of text id strings"""
        index = index or self.index
        query = {"query": {"ids": {"values": ids}}}
        result = self.client.search(index=index, body=query)["hits"]["hits"]
        documents = [self._convert_es_hit_to_document(hit, return_embedding=self.return_embedding) for hit in result]
        return documents

    def get_metadata_values_by_key(
        self,
        key: str,
        query: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        index: Optional[str] = None,
    ) -> List[dict]:
        """
        Get values associated with a metadata key. The output is in the format:
            [{"value": "my-value-1", "count": 23}, {"value": "my-value-2", "count": 12}, ... ]

        :param key: the meta key name to get the values for.
        :param query: narrow down the scope to documents matching the query string.
        :param filters: narrow down the scope to documents that match the given filters.
        :param index: Elasticsearch index where the meta values should be searched. If not supplied,
                      self.index will be used.
        """
        body: dict = {"size": 0, "aggs": {"metadata_agg": {"terms": {"field": key}}}}
        if query:
            body["query"] = {
                "bool": {
                    "should": [{"multi_match": {"query": query, "type": "most_fields", "fields": self.search_fields, }}]
                }
            }
        if filters:
            filter_clause = []
            for key, values in filters.items():
                filter_clause.append({"terms": {key: values}})
            if not body.get("query"):
                body["query"] = {"bool": {}}
            body["query"]["bool"].update({"filter": filter_clause})
        result = self.client.search(body=body, index=index)
        buckets = result["aggregations"]["metadata_agg"]["buckets"]
        for bucket in buckets:
            bucket["count"] = bucket.pop("doc_count")
            bucket["value"] = bucket.pop("key")
        return buckets

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,
                        batch_size: int = 10_000, duplicate_documents: Optional[str] = None):
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
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return: None
        """

        if index and not self.client.indices.exists(index=index):
            self._create_document_index(index)

        if index is None:
            index = self.index
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert duplicate_documents in self.duplicate_documents_options, \
            f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(documents=document_objects,
                                                            index=index,
                                                            duplicate_documents=duplicate_documents)
        documents_to_index = []
        for doc in document_objects:
            _doc = {
                "_op_type": "index" if duplicate_documents == 'overwrite' else "create",
                "_index": index,
                **doc.to_dict(field_map=self._create_document_field_map())
            }  # type: Dict[str, Any]

            # cast embedding type as ES cannot deal with np.array
            if _doc[self.embedding_field] is not None:
                if type(_doc[self.embedding_field]) == np.ndarray:
                    _doc[self.embedding_field] = _doc[self.embedding_field].tolist()

            # rename id for elastic
            _doc["_id"] = str(_doc.pop("id"))

            # don't index query score and empty fields
            _ = _doc.pop("score", None)
            _doc = {k:v for k,v in _doc.items() if v is not None}

            # In order to have a flat structure in elastic + similar behaviour to the other DocumentStores,
            # we "unnest" all value within "meta"
            if "meta" in _doc.keys():
                for k, v in _doc["meta"].items():
                    _doc[k] = v
                _doc.pop("meta")
            documents_to_index.append(_doc)

            # Pass batch_size number of documents to bulk
            if len(documents_to_index) % batch_size == 0:
                bulk(self.client, documents_to_index, request_timeout=300, refresh=self.refresh_type)
                documents_to_index = []

        if documents_to_index:
            bulk(self.client, documents_to_index, request_timeout=300, refresh=self.refresh_type)

    def write_labels(
        self, labels: Union[List[Label], List[dict]], index: Optional[str] = None, batch_size: int = 10_000
    ):
        """Write annotation labels into document store.

        :param labels: A list of Python dictionaries or a list of Haystack Label objects.
        :param index: Elasticsearch index where the labels should be stored. If not supplied, self.label_index will be used.
        :param batch_size: Number of labels that are passed to Elasticsearch's bulk function at a time.
        """
        index = index or self.label_index
        if index and not self.client.indices.exists(index=index):
            self._create_label_index(index)

        labels = [Label.from_dict(label) if isinstance(label, dict) else label for label in labels]
        duplicate_ids: list = [label.id for label in self._get_duplicate_labels(labels, index=index)]
        if len(duplicate_ids) > 0:
            logger.warning(f"Duplicate Label IDs: Inserting a Label whose id already exists in this document store."
                           f" This will overwrite the old Label. Please make sure Label.id is a unique identifier of"
                           f" the answer annotation and not the question."
                           f" Problematic ids: {','.join(duplicate_ids)}")
        labels_to_index = []
        for label in labels:
            # create timestamps if not available yet
            if not label.created_at:  # type: ignore
                label.created_at = time.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore
            if not label.updated_at:  # type: ignore
                label.updated_at = label.created_at  # type: ignore

            _label = {
                "_op_type": "index" if self.duplicate_documents == "overwrite" or label.id in duplicate_ids else  # type: ignore
                "create",
                "_index": index,
                **label.to_dict()  # type: ignore
            }  # type: Dict[str, Any]

            # rename id for elastic
            if label.id is not None:  # type: ignore
                _label["_id"] = str(_label.pop("id"))  # type: ignore

            labels_to_index.append(_label)

            # Pass batch_size number of labels to bulk
            if len(labels_to_index) % batch_size == 0:
                bulk(self.client, labels_to_index, request_timeout=300, refresh=self.refresh_type)
                labels_to_index = []

        if labels_to_index:
            bulk(self.client, labels_to_index, request_timeout=300, refresh=self.refresh_type)

    def update_document_meta(self, id: str, meta: Dict[str, str]):
        """
        Update the metadata dictionary of a document by specifying its string id
        """
        body = {"doc": meta}
        self.client.update(index=self.index, id=id, body=body, refresh=self.refresh_type)

    def get_document_count(self, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None,
                           only_documents_without_embedding: bool = False) -> int:
        """
        Return the number of documents in the document store.
        """
        index = index or self.index

        body: dict = {"query": {"bool": {}}}
        if only_documents_without_embedding:
            body['query']['bool']['must_not'] = [{"exists": {"field": self.embedding_field}}]

        if filters:
            filter_clause = []
            for key, values in filters.items():
                if type(values) != list:
                    raise ValueError(
                        f'Wrong filter format for key "{key}": Please provide a list of allowed values for each key. '
                        'Example: {"name": ["some", "more"], "category": ["only_one"]} ')
                filter_clause.append(
                    {
                        "terms": {key: values}
                    }
                )
            body["query"]["bool"]["filter"] = filter_clause

        result = self.client.count(index=index, body=body)
        count = result["count"]
        return count

    def get_label_count(self, index: Optional[str] = None) -> int:
        """
        Return the number of labels in the document store
        """
        index = index or self.label_index
        return self.get_document_count(index=index)

    def get_embedding_count(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None) -> int:
        """
        Return the count of embeddings in the document store.
        """

        index = index or self.index

        body: dict = {"query": {"bool": {"must": [{"exists": {"field": self.embedding_field}}]}}}
        if filters:
            filter_clause = []
            for key, values in filters.items():
                if type(values) != list:
                    raise ValueError(
                        f'Wrong filter format for key "{key}": Please provide a list of allowed values for each key. '
                        'Example: {"name": ["some", "more"], "category": ["only_one"]} ')
                filter_clause.append(
                    {
                        "terms": {key: values}
                    }
                )
            body["query"]["bool"]["filter"] = filter_clause

        result = self.client.count(index=index, body=body)
        count = result["count"]
        return count

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """
        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """

        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        result = self._get_all_documents_in_index(index=index, filters=filters, batch_size=batch_size)
        for hit in result:
            document = self._convert_es_hit_to_document(hit, return_embedding=return_embedding)
            yield document

    def get_all_labels(
        self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, batch_size: int = 10_000
    ) -> List[Label]:
        """
        Return all labels in the document store
        """
        index = index or self.label_index
        result = list(self._get_all_documents_in_index(index=index, filters=filters, batch_size=batch_size))
        labels = [Label.from_dict({**hit["_source"], "id": hit["_id"]}) for hit in result]
        return labels

    def _get_all_documents_in_index(
        self,
        index: str,
        filters: Optional[Dict[str, List[str]]] = None,
        batch_size: int = 10_000,
        only_documents_without_embedding: bool = False,
    ) -> Generator[dict, None, None]:
        """
        Return all documents in a specific index in the document store
        """
        body: dict = {"query": {"bool": {}}}

        if filters:
            filter_clause = []
            for key, values in filters.items():
                filter_clause.append(
                    {
                        "terms": {key: values}
                    }
                )
            body["query"]["bool"]["filter"] = filter_clause

        if only_documents_without_embedding:
            body['query']['bool']['must_not'] = [{"exists": {"field": self.embedding_field}}]

        result = scan(self.client, query=body, index=index, size=batch_size, scroll=self.scroll)
        yield from result

    def query(
        self,
        query: Optional[str],
        filters: Optional[Dict[str, List[str]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query as defined by the BM25 algorithm.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """

        if index is None:
            index = self.index

        # Naive retrieval without BM25, only filtering
        if query is None:
            body = {"query":
                        {"bool": {"must":
                                      {"match_all": {}}}}}  # type: Dict[str, Any]
            if filters:
                filter_clause = []
                for key, values in filters.items():
                    filter_clause.append(
                        {
                            "terms": {key: values}
                        }
                    )
                body["query"]["bool"]["filter"] = filter_clause

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
                logger.warning("The query provided seems to be not a string, but an object "
                               f"of type {type(query)}. This can cause Elasticsearch to fail.")
            body = {
                "size": str(top_k),
                "query": {
                    "bool": {
                        "should": [{"multi_match": {"query": query, "type": "most_fields", "fields": self.search_fields}}]
                    }
                },
            }

            if filters:
                filter_clause = []
                for key, values in filters.items():
                    if type(values) != list:
                        raise ValueError(f'Wrong filter format: "{key}": {values}. Provide a list of values for each key. '
                                         'Example: {"name": ["some", "more"], "category": ["only_one"]} ')
                    filter_clause.append(
                        {
                            "terms": {key: values}
                        }
                    )
                body["query"]["bool"]["filter"] = filter_clause

        if self.excluded_meta_data:
            body["_source"] = {"excludes": self.excluded_meta_data}

        logger.debug(f"Retriever query: {body}")
        result = self.client.search(index=index, body=body)["hits"]["hits"]

        documents = [self._convert_es_hit_to_document(hit, return_embedding=self.return_embedding) for hit in result]
        return documents

    def query_by_embedding(self,
                           query_emb: np.ndarray,
                           filters: Optional[Dict[str, List[str]]] = None,
                           top_k: int = 10,
                           index: Optional[str] = None,
                           return_embedding: Optional[bool] = None) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param filters: Optional filters to narrow down the search space.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param top_k: How many documents to return
        :param index: Index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :return:
        """
        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        if not self.embedding_field:
            raise RuntimeError("Please specify arg `embedding_field` in ElasticsearchDocumentStore()")
        else:
            # +1 in similarity to avoid negative numbers (for cosine sim)
            body = {
                "size": top_k,
                "query": self._get_vector_similarity_query(query_emb, top_k)
            }
            if filters:
                filter_clause = []
                for key, values in filters.items():
                    if type(values) != list:
                        raise ValueError(f'Wrong filter format for key "{key}": Please provide a list of allowed values for each key. '
                                         'Example: {"name": ["some", "more"], "category": ["only_one"]} ')
                    filter_clause.append(
                        {
                            "terms": {key: values}
                        }
                    )
                body["query"]["script_score"]["query"] = {"bool": {"filter": filter_clause}}

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
                result = self.client.search(index=index, body=body, request_timeout=300)["hits"]["hits"]
                if len(result) == 0:
                    count_embeddings = self.get_embedding_count(index=index)
                    if count_embeddings == 0:
                        raise RequestError(400, "search_phase_execution_exception",
                                           {"error": "No documents with embeddings."})
            except RequestError as e:
                if e.error == "search_phase_execution_exception":
                    error_message: str = "search_phase_execution_exception: Likely some of your stored documents don't have embeddings." \
                                         " Run the document store's update_embeddings() method."
                    raise RequestError(e.status_code, error_message, e.info)
                else:
                    raise e

            documents = [
                self._convert_es_hit_to_document(hit, adapt_score_for_embedding=True, return_embedding=return_embedding)
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
            if type(self) == OpenSearchDocumentStore:
                similarity_fn_name = "innerproduct"
            elif type(self) == ElasticsearchDocumentStore:
                similarity_fn_name = "dotProduct"
        else:
            raise Exception("Invalid value for similarity in ElasticSearchDocumentStore constructor. Choose between \'cosine\' and \'dot_product\'")

        # To handle scenarios where embeddings may be missing
        script_score_query: dict = {"match_all": {}}
        if self.skip_missing_embeddings:
            script_score_query = {
                "bool": {
                    "filter": {
                        "bool": {
                            "must": [
                                {
                                    "exists": {
                                        "field": self.embedding_field
                                    }
                                }
                            ]
                        }
                    }
                }
            }

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
            self,
            hit: dict,
            return_embedding: bool,
            adapt_score_for_embedding: bool = False,

    ) -> Document:
        # We put all additional data of the doc into meta_data and return it in the API
        meta_data = {k:v for k,v in hit["_source"].items() if k not in (self.content_field, "content_type", self.embedding_field)}
        name = meta_data.pop(self.name_field, None)
        if name:
            meta_data["name"] = name

        score = hit["_score"] if hit["_score"] else None
        if score:
            if adapt_score_for_embedding:
                score = self._scale_embedding_score(score)
                if self.similarity == "cosine":
                    score = (score + 1) / 2  # scaling probability from cosine similarity
                elif self.similarity == "dot_product":
                    score = float(expit(np.asarray(score / 100)))  # scaling probability from dot product
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
            "embedding": embedding
        }
        document = Document.from_dict(doc_dict)

        return document

    def _scale_embedding_score(self, score):
        return score - 1000

    def describe_documents(self, index=None):
        """
        Return a summary of the documents in the document store
        """
        if index is None:
            index = self.index
        docs = self.get_all_documents(index)

        l = [len(d.content) for d in docs]
        stats = {"count": len(docs),
                 "chars_mean": np.mean(l),
                 "chars_max": max(l),
                 "chars_min": min(l),
                 "chars_median": np.median(l),
                 }
        return stats

    def update_embeddings(
        self,
        retriever,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        update_existing_embeddings: bool = True,
        batch_size: int = 10_000
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
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return: None
        """
        if index is None:
            index = self.index

        if self.refresh_type == 'false':
            self.client.indices.refresh(index=index)

        if not self.embedding_field:
            raise RuntimeError("Specify the arg `embedding_field` when initializing ElasticsearchDocumentStore()")

        if update_existing_embeddings:
            document_count = self.get_document_count(index=index)
            logger.info(f"Updating embeddings for all {document_count} docs ...")
        else:
            document_count = self.get_document_count(index=index, filters=filters,
                                                     only_documents_without_embedding=True)
            logger.info(f"Updating embeddings for {document_count} docs without embeddings ...")

        result = self._get_all_documents_in_index(
            index=index,
            filters=filters,
            batch_size=batch_size,
            only_documents_without_embedding=not update_existing_embeddings
        )

        logging.getLogger("elasticsearch").setLevel(logging.CRITICAL)

        with tqdm(total=document_count, position=0, unit=" Docs", desc="Updating embeddings") as progress_bar:
            for result_batch in get_batches_from_generator(result, batch_size):
                document_batch = [self._convert_es_hit_to_document(hit, return_embedding=False) for hit in result_batch]
                embeddings = retriever.embed_documents(document_batch)  # type: ignore
                assert len(document_batch) == len(embeddings)

                if embeddings[0].shape[0] != self.embedding_dim:
                    raise RuntimeError(f"Embedding dim. of model ({embeddings[0].shape[0]})"
                                       f" doesn't match embedding dim. in DocumentStore ({self.embedding_dim})."
                                       "Specify the arg `embedding_dim` when initializing ElasticsearchDocumentStore()")
                doc_updates = []
                for doc, emb in zip(document_batch, embeddings):
                    update = {"_op_type": "update",
                              "_index": index,
                              "_id": doc.id,
                              "doc": {self.embedding_field: emb.tolist()},
                              }
                    doc_updates.append(update)

                bulk(self.client, doc_updates, request_timeout=300, refresh=self.refresh_type)
                progress_bar.update(batch_size)

    def delete_all_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the document from.
        :param filters: Optional filters to narrow down the documents to be deleted.
        :return: None
        """
        logger.warning(
                """DEPRECATION WARNINGS: 
                1. delete_all_documents() method is deprecated, please use delete_documents method
                For more details, please refer to the issue: https://github.com/deepset-ai/haystack/issues/1045
                """
        )
        self.delete_documents(index, None, filters)

    def delete_documents(self, index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the documents from. If None, the
                      DocumentStore's default index (self.index) will be used
        :param ids: Optional list of IDs to narrow down the documents to be deleted.
        :param filters: Optional filters to narrow down the documents to be deleted.
            Example filters: {"name": ["some", "more"], "category": ["only_one"]}.
            If filters are provided along with a list of IDs, this method deletes the
            intersection of the two query results (documents that match the filters and
            have their ID in the list).
        :return: None
        """
        index = index or self.index
        query: Dict[str, Any] = {"query": {}}
        if filters:
            filter_clause = []
            for key, values in filters.items():
                filter_clause.append(
                        {
                            "terms": {key: values}
                        }
                )
                query["query"]["bool"] = {"filter": filter_clause}

            if ids:
                query["query"]["bool"]["must"] = {"ids": {"values": ids}}

        elif ids:
            query["query"]["ids"] = {"values": ids}
        else:
            query["query"] = {"match_all": {}}
        self.client.delete_by_query(index=index, body=query, ignore=[404])
        # We want to be sure that all docs are deleted before continuing (delete_by_query doesn't support wait_for)
        if self.refresh_type == "wait_for":
            time.sleep(2)

    def delete_labels(self, index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete labels in an index. All labels are deleted if no filters are passed.

        :param index: Index name to delete the labels from. If None, the
                      DocumentStore's default label index (self.label_index) will be used
        :param ids: Optional list of IDs to narrow down the labels to be deleted.
        :param filters: Optional filters to narrow down the labels to be deleted.
            Example filters: {"id": ["9a196e41-f7b5-45b4-bd19-5feb7501c159", "9a196e41-f7b5-45b4-bd19-5feb7501c159"]} or {"query": ["question2"]}
        :return: None
        """
        index = index or self.label_index
        self.delete_documents(index=index, ids=ids, filters=filters)


class OpenSearchDocumentStore(ElasticsearchDocumentStore):
    """
    Document Store using OpenSearch (https://opensearch.org/). It is compatible with the AWS Elasticsearch Service.

    In addition to native Elasticsearch query & filtering, it provides efficient vector similarity search using
    the KNN plugin that can scale to a large number of documents.
    """

    def __init__(self,
                 verify_certs=False,
                 scheme="https",
                 username="admin",
                 password="admin",
                 port=9200,
                 **kwargs):

        # Overwrite default kwarg values of parent class so that in default cases we can initialize
        # an OpenSearchDocumentStore without provding any arguments

        super(OpenSearchDocumentStore, self).__init__(verify_certs=verify_certs,
                                                      scheme=scheme,
                                                      username=username,
                                                      password=password,
                                                      port=port,
                                                      **kwargs)


    def query_by_embedding(self,
                        query_emb: np.ndarray,
                        filters: Optional[Dict[str, List[str]]] = None,
                        top_k: int = 10,
                        index: Optional[str] = None,
                        return_embedding: Optional[bool] = None) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param filters: Optional filters to narrow down the search space.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param top_k: How many documents to return
        :param index: Index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :return:
        """
        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        if not self.embedding_field:
            raise RuntimeError("Please specify arg `embedding_field` in ElasticsearchDocumentStore()")
        else:
            # +1 in similarity to avoid negative numbers (for cosine sim)
            body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [
                            self._get_vector_similarity_query(query_emb, top_k)
                        ]
                    }
                }
            }
            if filters:
                filter_clause = []
                for key, values in filters.items():
                    if type(values) != list:
                        raise ValueError(
                            f'Wrong filter format for key "{key}": Please provide a list of allowed values for each key. '
                            'Example: {"name": ["some", "more"], "category": ["only_one"]} ')
                    filter_clause.append(
                        {
                            "terms": {key: values}
                        }
                    )
                body["query"]["bool"]["filter"] = filter_clause         # type: ignore

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
            result = self.client.search(index=index, body=body, request_timeout=300)["hits"]["hits"]

            documents = [
                self._convert_es_hit_to_document(hit, adapt_score_for_embedding=True, return_embedding=return_embedding)
                for hit in result
            ]
            return documents

    def _create_document_index(self, index_name: str):
        """
        Create a new index for storing documents.
        """

        if self.custom_mapping:
            mapping = self.custom_mapping
        else:
            mapping = {
                "mappings": {
                    "properties": {
                        self.name_field: {"type": "keyword"},
                        self.content_field: {"type": "text"},
                    },
                    "dynamic_templates": [
                        {
                            "strings": {
                                "path_match": "*",
                                "match_mapping_type": "string",
                                "mapping": {"type": "keyword"}}}
                    ],
                },
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": self.analyzer,
                            }
                        }
                    }
                }
            }
            if self.embedding_field:

                if self.similarity == "cosine":
                    similarity_space_type = "cosinesimil"
                elif self.similarity == "dot_product":
                    similarity_space_type = "innerproduct"
                elif self.similarity == "l2":
                    similarity_space_type = "l2"

                mapping["settings"]["index"] = {}
                mapping["settings"]["index"]["knn"] = True
                mapping["settings"]["index"]["knn.space_type"] = similarity_space_type

                mapping["mappings"]["properties"][self.embedding_field] = {
                    "type": "knn_vector",
                    "dimension": self.embedding_dim,
                }

                if self.index_type == "flat":
                    pass
                elif self.index_type == "hnsw":
                    mapping["settings"]["index"]["knn.algo_param"] = {}
                    mapping["settings"]["index"]["knn.algo_param"]["ef_search"] = 20
                    mapping["mappings"]["properties"][self.embedding_field]["method"] = {
                        "space_type": similarity_space_type,
                        "name": "hnsw",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 80,
                            "m": 64
                        }
                    }
                else:
                    logger.error("Please set index_type to either 'flat' or 'hnsw'")

        try:
            self.client.indices.create(index=index_name, body=mapping)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name):
                raise e

    def _create_label_index(self, index_name: str):
        if self.client.indices.exists(index=index_name):
            return
        mapping = {
            "mappings": {
                "properties": {
                    "query": {"type": "text"},
                    "answer": {"type": "nested"}, # In elasticsearch we use type:flattened, but this is not supported in opensearch
                    "document": {"type": "nested"},
                    "is_correct_answer": {"type": "boolean"},
                    "is_correct_document": {"type": "boolean"},
                    "origin": {"type": "keyword"},  # e.g. user-feedback or gold-label
                    "document_id": {"type": "keyword"},
                    "no_answer": {"type": "boolean"},
                    "pipeline_id": {"type": "keyword"},
                    "created_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"},
                    "updated_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"}
                    #TODO add pipeline_hash and pipeline_name once we migrated the REST API to pipelines
                }
            }
        }
        try:
            self.client.indices.create(index=index_name, body=mapping)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name):
                raise e

    def _get_vector_similarity_query(self, query_emb: np.ndarray, top_k: int):
        """
        Generate Elasticsearch query for vector similarity.
        """
        query = {"knn": {self.embedding_field: {"vector": query_emb.tolist(), "k": top_k}}}
        return query

    def _scale_embedding_score(self, score):
        return score


class OpenDistroElasticsearchDocumentStore(OpenSearchDocumentStore):
    """
    A DocumentStore which has an Open Distro for Elasticsearch service behind it.
    """
    def __init__(self, host="https://admin:admin@localhost:9200/", similarity="cosine", **kwargs):
        logger.warning("Open Distro for Elasticsearch has been replaced by OpenSearch! "
                       "See https://opensearch.org/faq/ for details. "
                       "We recommend using the OpenSearchDocumentStore instead.")
        super(OpenDistroElasticsearchDocumentStore, self).__init__(host=host,
                                                                   similarity=similarity,
                                                                   **kwargs)
    def _prepare_hosts(self, host, port):
        return host
