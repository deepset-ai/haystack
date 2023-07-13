import logging
from typing import List, Optional, Type, Union

from haystack.lazy_imports import LazyImport
from haystack.document_stores.search_engine import prepare_hosts
from .base import _ElasticsearchDocumentStore


failed_import_message = (
    "Run 'pip install farm-haystack[elasticsearch]'.\n"
    "Note that this installs the Elasticsearch 7 client and requires a running Elasticsearch 7 instance. Starting from "
    "Haystack version 1.21 , the 'elasticsearch' extra will install the Elasticsearch 8 client and will require a "
    "running Elasticsearch 8 instance. To continue using Elasticsearch 7 with Haystack 1.21 and beyond, install the "
    "'elasticsearch7' extra.\n"
    "If you want to already start using Elasticsearch 8, install the 'elasticsearch8' extra"
)
with LazyImport(failed_import_message) as es_import:
    from elasticsearch import Connection, Elasticsearch, RequestsHttpConnection, Urllib3HttpConnection
    from elasticsearch.helpers import bulk, scan
    from elasticsearch.exceptions import RequestError

logger = logging.getLogger(__name__)


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
        # Initiate the Elasticsearch client for version 7.x
        client = ElasticsearchDocumentStore._init_elastic_client(
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
            batch_size=batch_size,
        )

        self._validate_server_version(expected_version=7)

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
        aws4auth,
        scheme: str,
        ca_certs: Optional[str],
        verify_certs: bool,
        timeout: int,
        use_system_proxy: bool,
    ) -> "Elasticsearch":
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
