import logging
from typing import Dict, List, Optional, Type, Union

from haystack import Document
from haystack.document_stores.elasticsearch import BaseElasticsearchDocumentStore
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install farm-haystack[elasticsearch8]'") as es_import:
    from elasticsearch import Elasticsearch, RequestError
    from elasticsearch.helpers import bulk, scan
    from elastic_transport import BaseNode, RequestsHttpNode, Urllib3HttpNode


logger = logging.getLogger(__name__)


class ElasticsearchDocumentStore(BaseElasticsearchDocumentStore):
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
        es_import.check()

        client = self._init_elastic_client(
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

        self._RequestError = RequestError

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

    def _do_bulk(
        self, actions: Union[List[Dict], List[Document]], client: "Elasticsearch", refresh: str, headers: Optional[Dict]  # type: ignore[override]
    ):
        """Override the base class method to use the Elasticsearch client"""
        client = client.options(headers=headers)
        return bulk(actions=actions, client=client, refresh=refresh)

    def _do_scan(self, *args, **kwargs):
        """Override the base class method to use the Elasticsearch client"""
        if "client" in kwargs and "headers" in kwargs and kwargs["headers"] is not None:
            kwargs["client"] = kwargs.get("client").options(headers=kwargs.get("headers"))
        return scan(*args, **kwargs)

    def _client_indices_exists(self, index: str, headers: Optional[Dict]) -> bool:
        return self.client.options(headers=headers).indices.exists(index=index)

    def _client_indices_exists_alias(self, index: str, headers: Optional[Dict]) -> bool:
        return self.client.indices.options(headers=headers).exists_alias(name=index)

    def _client_search(self, index: str, body: Dict, headers: Optional[Dict]) -> Dict:
        return self.client.options(headers=headers).search(index=index, **body)

    def _client_update(self, index: str, doc_id: str, meta: Dict, headers: Optional[Dict]):
        self.client.options(headers=headers).update(index=index, id=doc_id, **meta, refresh=self.refresh_type)

    def _client_count(self, index: str, body: Dict, headers: Optional[Dict]) -> Dict:
        return self.client.options(headers=headers).count(index=index, **body, headers=headers)

    def _client_indices_refresh(self, index: str, headers: Optional[Dict]):
        self.client.options(headers=headers).indices.refresh(index=index)

    def _client_delete_by_query(self, index: str, body: Dict, headers: Optional[Dict]):
        self.client.options(headers=headers, ignore_status=404).delete_by_query(index=index, **body, headers=headers)

    def _client_indices_delete(self, index: str, headers: Optional[Dict]):
        self.client.options(headers=headers, ignore_status=[400, 404]).indices.delete(index=index)

    def _client_indices_create(self, index: str, mapping: Dict, headers: Optional[Dict]):
        self.client.options(headers=headers).indices.create(index=index, **mapping)

    def _client_indices_get(self, index: str, headers: Optional[Dict]) -> Dict:
        return self.client.options(headers=headers).indices.get(index=index)

    def _client_indices_put_mapping(self, index: str, mapping: Dict, headers: Optional[Dict]):
        self.client.options(headers=headers).indices.put_mapping(index=index, **mapping)
