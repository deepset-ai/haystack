import logging
from typing import Dict, List, Optional, Type, Union

import numpy as np

try:
    from elasticsearch import Connection, Elasticsearch, RequestsHttpConnection, Urllib3HttpConnection
    from elasticsearch.helpers import bulk, scan
    from elasticsearch.exceptions import RequestError
except (ImportError, ModuleNotFoundError) as ie:
    from haystack.utils.import_utils import _optional_component_not_installed

    _optional_component_not_installed(__name__, "elasticsearch", ie)

from haystack.errors import DocumentStoreError
from haystack.schema import Document, FilterType
from haystack.document_stores.filter_utils import LogicalFilterClause

from .search_engine import SearchEngineDocumentStore, prepare_hosts

logger = logging.getLogger(__name__)


class ElasticsearchDocumentStore(SearchEngineDocumentStore):
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
        # Base constructor might need the client to be ready, create it first
        client = self._init_elastic_client(
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
        )

        # Let the base class trap the right exception from the elasticpy client
        self._RequestError = RequestError

    def _do_bulk(self, *args, **kwargs):
        """Override the base class method to use the Elasticsearch client"""
        return bulk(*args, **kwargs)

    def _do_scan(self, *args, **kwargs):
        """Override the base class method to use the Elasticsearch client"""
        return scan(*args, **kwargs)

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
        hosts = prepare_hosts(host, port)

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
            if username:
                logger.warning(
                    "aws4auth and a username are passed to the ElasticsearchDocumentStore. The username will be ignored and aws4auth will be used for authentication."
                )
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
            result = self.client.search(index=index, body=body, request_timeout=300, headers=headers)["hits"]["hits"]
            if len(result) == 0:
                count_documents = self.get_document_count(index=index, headers=headers)
                if count_documents == 0:
                    logger.warning("Index is empty. First add some documents to search them.")
                count_embeddings = self.get_embedding_count(index=index, headers=headers)
                if count_embeddings == 0:
                    logger.warning("No documents with embeddings. Run the document store's update_embeddings() method.")
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
        body = {"size": top_k, "query": self._get_vector_similarity_query(query_emb, top_k)}
        if filters:
            filter_ = {"bool": {"filter": LogicalFilterClause.parse(filters).convert_to_elasticsearch()}}
            if body["query"]["script_score"]["query"] == {"match_all": {}}:
                body["query"]["script_score"]["query"] = filter_
            else:
                body["query"]["script_score"]["query"]["bool"]["filter"]["bool"]["must"].append(filter_)

        excluded_fields = self._get_excluded_fields(return_embedding=return_embedding)
        if excluded_fields:
            body["_source"] = {"excludes": excluded_fields}

        return body

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
            self.client.indices.create(index=index_name, body=mapping, headers=headers)
        except self._RequestError as e:
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
            self.client.indices.create(index=index_name, body=mapping, headers=headers)
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
        indices = self.client.indices.get(index_name, headers=headers)

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
                        self.client.indices.put_mapping(index=index_id, body=mapping, headers=headers)

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
                mapping["properties"][self.embedding_field] = {"type": "dense_vector", "dims": self.embedding_dim}
                self.client.indices.put_mapping(index=index_id, body=mapping, headers=headers)

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
            raise DocumentStoreError(
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

    def _get_raw_similarity_score(self, score):
        return score - 1000
