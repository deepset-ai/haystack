import logging
from typing import List, Optional, Union, Dict, Any

from haystack.lazy_imports import LazyImport
from haystack import Document

with LazyImport("Run 'pip install farm-haystack[elasticsearch8]'") as es_import:
    from elasticsearch import Elasticsearch, RequestError
    from elasticsearch.helpers import bulk, scan
    from elastic_transport import RequestsHttpNode, Urllib3HttpNode

from .base import _ElasticsearchDocumentStore

logger = logging.getLogger(__name__)


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


class ElasticsearchDocumentStore(_ElasticsearchDocumentStore):
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
        :param batch_size: Number of Documents to index at once / Number of queries to execute at once. If you face
                           memory issues, decrease the batch_size.

        """
        # Ensure all the required inputs were successful
        es_import.check()
        # Let the base class trap the right exception from the specific client
        self._RequestError = RequestError
        # Initiate the Elasticsearch client for version 8.x
        client = ElasticsearchDocumentStore._init_elastic_client(
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

        self._validate_server_version(expected_version=8)

    def _do_bulk(self, *args, **kwargs):
        """Override the base class method to use the Elasticsearch client"""
        return bulk(*args, **kwargs)

    def _do_scan(self, *args, **kwargs):
        """Override the base class method to use the Elasticsearch client"""
        return scan(*args, **kwargs)

    @staticmethod
    def _init_elastic_client(
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
        aws4auth: Optional[str] = "",
    ) -> Elasticsearch:
        hosts = _prepare_hosts(host, port, scheme)

        if aws4auth:
            logger.warning("AWS authentication is not supported in Elasticsearch version 8 and later!")

        if (api_key or api_key_id) and not (api_key and api_key_id):
            raise ValueError("You must provide either both or none of `api_key_id` and `api_key`.")

        node_class = RequestsHttpNode if use_system_proxy else Urllib3HttpNode

        if api_key_id and api_key:
            # api key authentication
            if ca_certs is not None:
                client = Elasticsearch(
                    hosts=hosts,
                    api_key=(api_key_id, api_key),
                    ca_certs=ca_certs,
                    verify_certs=verify_certs,
                    request_timeout=timeout,
                    node_class=node_class,
                )
            else:
                client = Elasticsearch(
                    hosts=hosts,
                    api_key=(api_key_id, api_key),
                    verify_certs=verify_certs,
                    request_timeout=timeout,
                    node_class=node_class,
                )
        elif username:
            # standard http_auth
            if ca_certs is not None:
                client = Elasticsearch(
                    hosts=hosts,
                    basic_auth=(username, password),
                    ca_certs=ca_certs,
                    verify_certs=verify_certs,
                    request_timeout=timeout,
                    node_class=node_class,
                )
            else:
                client = Elasticsearch(
                    hosts=hosts,
                    basic_auth=(username, password),
                    verify_certs=verify_certs,
                    request_timeout=timeout,
                    node_class=node_class,
                )
        else:
            # there is no authentication for this elasticsearch instance
            if ca_certs is not None:
                client = Elasticsearch(
                    hosts=hosts,
                    ca_certs=ca_certs,
                    verify_certs=verify_certs,
                    request_timeout=timeout,
                    node_class=node_class,
                )
            else:
                client = Elasticsearch(
                    hosts=hosts,
                    basic_auth=(username, password),
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

    def _index_exists(self, index_name: str, headers: Optional[Dict[str, str]] = None) -> bool:
        if logger.isEnabledFor(logging.DEBUG):
            if self.client.options(headers=headers).indices.exists_alias(name=index_name):
                logger.debug("Index name %s is an alias.", index_name)

        return self.client.options(headers=headers).indices.exists(index=index_name)

    def _index_delete(self, index):
        if self._index_exists(index):
            self.client.options(ignore_status=[400, 404]).indices.delete(index=index)
            logger.info("Index '%s' deleted.", index)

    def _index_refresh(self, index, headers):
        if self._index_exists(index):
            self.client.options(headers=headers).indices.refresh(index=index)

    def _index_create(self, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        return self.client.options(headers=headers).indices.create(*args, **kwargs)

    def _index_get(self, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        return self.client.options(headers=headers).indices.get(*args, **kwargs)

    def _index_put_mapping(self, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        body = kwargs.pop("body", {})
        return self.client.options(headers=headers).indices.put_mapping(*args, **kwargs, **body)

    def _search(self, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        return self.client.options(headers=headers).search(*args, **kwargs)

    def _update(self, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        return self.client.options(headers=headers).update(*args, **kwargs)

    def _count(self, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        body = kwargs.pop("body", {})
        return self.client.options(headers=headers).count(*args, **kwargs, **body)

    def _delete_by_query(self, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        ignore_status = kwargs.pop("ignore", [])
        body = kwargs.pop("body", {})
        return self.client.options(headers=headers, ignore_status=ignore_status).delete_by_query(
            *args, **kwargs, **body
        )

    def _execute_msearch(self, index: str, body: List[Dict[str, Any]], scale_score: bool) -> List[List[Document]]:
        responses = self.client.msearch(index=index, body=body)
        documents = []
        for response in responses["responses"]:
            result = response["hits"]["hits"]
            cur_documents = [self._convert_es_hit_to_document(hit, scale_score=scale_score) for hit in result]
            documents.append(cur_documents)

        return documents
