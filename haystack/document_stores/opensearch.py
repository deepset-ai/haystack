from typing import List, Optional, Union, Dict, Any

import logging

import numpy as np
from tqdm.auto import tqdm
from tenacity import retry, wait_exponential, retry_if_not_result

try:
    from opensearchpy import OpenSearch, Urllib3HttpConnection, RequestsHttpConnection, NotFoundError, RequestError
    from opensearchpy.helpers import bulk, scan
except (ImportError, ModuleNotFoundError) as e:
    from haystack.utils.import_utils import _optional_component_not_installed

    _optional_component_not_installed(__name__, "opensearch", e)


from haystack.schema import Document, FilterType
from haystack.document_stores.base import get_batches_from_generator
from haystack.document_stores.filter_utils import LogicalFilterClause
from haystack.errors import DocumentStoreError
from haystack.nodes.retriever import DenseRetriever

from .search_engine import SearchEngineDocumentStore, prepare_hosts

logger = logging.getLogger(__name__)


SIMILARITY_SPACE_TYPE_MAPPINGS = {
    "nmslib": {"cosine": "cosinesimil", "dot_product": "innerproduct", "l2": "l2"},
    "score_script": {"cosine": "cosinesimil", "dot_product": "innerproduct", "l2": "l2"},
    "faiss": {"cosine": "innerproduct", "dot_product": "innerproduct", "l2": "l2"},
}


class OpenSearchDocumentStore(SearchEngineDocumentStore):
    valid_index_types = ["flat", "hnsw", "ivf", "ivf_pq"]

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
        knn_engine: str = "nmslib",
        knn_parameters: Optional[Dict] = None,
        ivf_train_size: Optional[int] = None,
    ):
        """
        Document Store using OpenSearch (https://opensearch.org/). It is compatible with the Amazon OpenSearch Service.

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
        :param index_type: The type of index you want to create. Choose from 'flat', 'hnsw', 'ivf', or 'ivf_pq'.
                           'ivf_pq' is an IVF index optimized for memory through product quantization.
                           ('ivf' and 'ivf_pq' are only available with 'faiss' as knn_engine.)
                           If index_type='flat', we use OpenSearch's default index settings (which is an hnsw index
                           optimized for accuracy and memory footprint), since OpenSearch does not require a special
                           index for exact vector similarity calculations. Note that OpenSearchDocumentStore will only
                           perform exact vector calculations if the selected knn_engine supports it (currently only
                           knn_engine='score_script'). For the other knn_engines we use hnsw, as this usually achieves
                           the best balance between nearly as good accuracy and latency.
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
        :param knn_engine: The engine you want to use for the nearest neighbor search by OpenSearch's KNN plug-in. Possible values: "nmslib", "faiss" or "score_script". Defaults to "nmslib".
                        For more information, see [k-NN Index](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/).
        :param knn_parameters: Custom parameters for the KNN engine. Parameter names depend on the index type you use.
                               Configurable parameters for indices of type...
                                 - `hnsw`: `"ef_construction"`, `"ef_search"`, `"m"`
                                 - `ivf`: `"nlist"`, `"nprobes"`
                                 - `ivf_pq`: `"nlist"`, `"nprobes"`, `"m"`, `"code_size"`
                               If you don't specify any parameters, the OpenSearch's default values are used.
                               (With the exception of index_type='hnsw', where we use values other than OpenSearch's
                               default ones to achieve comparability throughout DocumentStores in Haystack.)
                               For more information on configuration of knn indices, see
                               [OpenSearch Documentation](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/#method-definitions).
        :param ivf_train_size: Number of embeddings to use for training the IVF index. Training starts automatically
                               once the number of indexed embeddings exceeds ivf_train_size. If `None`, the minimum
                               number of embeddings recommended for training by FAISS is used (depends on the desired
                               index type and knn parameters). If `0`, training doesn't happen automatically but needs
                               to be triggered manually via the `train_index` method.
                               Default: `None`
        """
        # These parameters aren't used by Opensearch at the moment but could be in the future, see
        # https://github.com/opensearch-project/security/issues/1504. Let's not deprecate them for
        # now but send a warning to the user.
        if api_key or api_key_id:
            logger.warning("api_key and api_key_id will be ignored by the Opensearch client")

        # Base constructor needs the client to be ready, create it before calling super()
        client = self._init_client(
            host=host,
            port=port,
            username=username,
            password=password,
            aws4auth=aws4auth,
            scheme=scheme,
            ca_certs=ca_certs,
            verify_certs=verify_certs,
            timeout=timeout,
            use_system_proxy=use_system_proxy,
        )

        # Test the connection
        try:
            client.indices.get(index)
        except NotFoundError:
            # We don't know which permissions the user has but we can assume they can write to the given index, so
            # if we get a NotFoundError it means at least the connection is working.
            pass
        except Exception as e:
            # If we get here, there's something fundamentally wrong with the connection and we can't continue
            raise ConnectionError(
                f"Initial connection to Opensearch failed with error '{e}'\n"
                f"Make sure an Opensearch instance is running at `{host}` and that it has finished booting (can take > 30s)."
            )

        if knn_engine not in {"nmslib", "faiss", "score_script"}:
            raise ValueError(f"knn_engine must be either 'nmslib', 'faiss' or 'score_script' but was {knn_engine}")

        if index_type in self.valid_index_types:
            if index_type in ["ivf", "ivf_pq"] and knn_engine != "faiss":
                raise DocumentStoreError("Use 'faiss' as knn_engine when using 'ivf' as index_type.")
            self.index_type = index_type
        else:
            raise DocumentStoreError(
                f"Invalid value for index_type in constructor. Choose one of these values: {self.valid_index_types}."
            )

        self.knn_engine = knn_engine
        self.knn_parameters = {} if knn_parameters is None else knn_parameters
        if ivf_train_size is not None:
            if ivf_train_size <= 0:
                raise DocumentStoreError("`ivf_train_on_write_size` must be None or a positive integer.")
            self.ivf_train_size = ivf_train_size
        elif self.index_type in ["ivf", "ivf_pq"]:
            self.ivf_train_size = self._recommended_ivf_train_size()
        self.space_type = SIMILARITY_SPACE_TYPE_MAPPINGS[knn_engine][similarity]
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

        # Let the base class catch the right error from the Opensearch client
        self._RequestError = RequestError

    def _do_bulk(self, *args, **kwargs):
        """Override the base class method to use the Opensearch client"""
        return bulk(*args, **kwargs)

    def _do_scan(self, *args, **kwargs):
        """Override the base class method to use the Opensearch client"""
        return scan(*args, **kwargs)

    @classmethod
    def _init_client(
        cls,
        host: Union[str, List[str]],
        port: Union[int, List[int]],
        username: str,
        password: str,
        aws4auth,
        scheme: str,
        ca_certs: Optional[str],
        verify_certs: bool,
        timeout: int,
        use_system_proxy: bool,
    ) -> OpenSearch:
        """
        Create an instance of the Opensearch client
        """
        hosts = prepare_hosts(host, port)
        connection_class = Urllib3HttpConnection
        if use_system_proxy:
            connection_class = RequestsHttpConnection  # type: ignore [assignment]

        if aws4auth:
            # Sign requests to Opensearch with IAM credentials
            # see https://docs.aws.amazon.com/opensearch-service/latest/developerguide/request-signing.html#request-signing-python
            if username:
                logger.warning(
                    "aws4auth and a username or the default username 'admin' are passed to the OpenSearchDocumentStore. The username will be ignored and aws4auth will be used for authentication."
                )
            client = OpenSearch(
                hosts=hosts,
                http_auth=aws4auth,
                connection_class=RequestsHttpConnection,
                use_ssl=True,
                verify_certs=True,
                timeout=timeout,
            )
        elif username:
            # standard http_auth
            client = OpenSearch(
                hosts=hosts,
                http_auth=(username, password),
                scheme=scheme,
                ca_certs=ca_certs,
                verify_certs=verify_certs,
                timeout=timeout,
                connection_class=connection_class,
            )
        else:
            # no authentication needed
            client = OpenSearch(
                hosts=hosts,
                scheme=scheme,
                ca_certs=ca_certs,
                verify_certs=verify_certs,
                timeout=timeout,
                connection_class=connection_class,
            )

        return client

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Indexes documents for later queries in OpenSearch.

        If a document with the same ID already exists in OpenSearch:
        a) (Default) Throw Elastic's standard error message for duplicate IDs.
        b) If `self.update_existing_documents=True` for DocumentStore: Overwrite existing documents.
        (This is only relevant if you pass your own ID when initializing a `Document`.
        If you don't set custom IDs for your Documents or just pass a list of dictionaries here,
        they automatically get UUIDs assigned. See the `Document` class for details.)

        :param documents: A list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"content": "<the-actual-text>"}.
                          Optionally: Include meta data via {"content": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          You can use it for filtering and you can access it in the responses of the Finder.
                          Advanced: If you are using your own OpenSearch mapping, change the key names in the dictionary
                          to what you have set for self.content_field and self.name_field.
        :param index: OpenSearch index where the documents should be indexed. If you don't specify it, self.index is used.
        :param batch_size: Number of documents that are passed to OpenSearch's bulk function at a time.
        :param duplicate_documents: Handle duplicate documents based on parameter options.
                                    Parameter options: ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicate documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: Raises an error if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to OpenSearch client (for example {'Authorization': 'Basic YWRtaW46cm9vdA=='})
                For more information, see [HTTP/REST clients and security](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html).
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return: None
        """
        if index is None:
            index = self.index

        if self.knn_engine == "faiss" and self.similarity == "cosine":
            field_map = self._create_document_field_map()
            documents = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
            embeddings_to_index = np.array([d.embedding for d in documents], dtype="float32")
            self.normalize_embedding(embeddings_to_index)
            for document, embedding in zip(documents, embeddings_to_index):
                document.embedding = None if np.isnan(embedding).any() else embedding

        super().write_documents(
            documents=documents,
            index=index,
            batch_size=batch_size,
            duplicate_documents=duplicate_documents,
            headers=headers,
        )

        # Train IVF index if number of embeddings exceeds ivf_train_size
        if (
            self.index_type in ["ivf", "ivf_pq"]
            and not index.startswith(".")
            and not self._ivf_model_exists(index=index)
        ):
            if self.get_embedding_count(index=index, headers=headers) >= self.ivf_train_size:
                train_docs = self.get_all_documents(index=index, return_embedding=True, headers=headers)
                self._train_ivf_index(index=index, documents=train_docs, headers=headers)

    def _embed_documents(self, documents: List[Document], retriever: DenseRetriever) -> np.ndarray:
        """
        Embed a list of documents using a Retriever.
        :param documents: List of documents to embed.
        :param retriever: Retriever to use for embedding.
        :return: embeddings of documents.
        """
        embeddings = super()._embed_documents(documents, retriever)
        if self.knn_engine == "faiss" and self.similarity == "cosine":
            self.normalize_embedding(embeddings)
        return embeddings

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

        if self.index_type in ["ivf", "ivf_pq"] and not self._ivf_model_exists(index=index):
            self._ivf_index_not_trained_error(index=index, headers=headers)

        if not self.embedding_field:
            raise DocumentStoreError("Please set a valid `embedding_field` for OpenSearchDocumentStore")
        body = self._construct_dense_query_body(
            query_emb=query_emb, filters=filters, top_k=top_k, return_embedding=return_embedding
        )

        logger.debug("Retriever query: %s", body)
        result = self.client.search(index=index, body=body, request_timeout=300, headers=headers)["hits"]["hits"]

        documents = [
            self._convert_es_hit_to_document(hit, adapt_score_for_embedding=True, scale_score=scale_score)
            for hit in result
        ]

        if self.index_type == "hnsw":
            ef_search = self._get_ef_search_value()
            if top_k > ef_search:
                logger.warning(
                    "top_k (%i) is greater than ef_search (%i). "
                    "We recommend setting ef_search >= top_k for optimal performance.",
                    top_k,
                    ef_search,
                )

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

        if self.index_type in ["ivf", "ivf_pq"] and not self._ivf_model_exists(index=index):
            self._ivf_index_not_trained_error(index=index, headers=headers)

        return super().query_by_embedding_batch(
            query_embs, filters, top_k, index, return_embedding, headers, scale_score
        )

    def _construct_dense_query_body(
        self, query_emb: np.ndarray, return_embedding: bool, filters: Optional[FilterType] = None, top_k: int = 10
    ):
        body: Dict[str, Any] = {"size": top_k, "query": self._get_vector_similarity_query(query_emb, top_k)}
        if filters:
            filter_ = LogicalFilterClause.parse(filters).convert_to_elasticsearch()
            if "script_score" in body["query"]:
                # set filter for pre-filtering (see https://opensearch.org/docs/latest/search-plugins/knn/knn-score-script/)
                body["query"]["script_score"]["query"] = {"bool": {"filter": filter_}}
            else:
                body["query"]["bool"]["filter"] = filter_

        excluded_fields = self._get_excluded_fields(return_embedding=return_embedding)
        if excluded_fields:
            body["_source"] = {"excludes": excluded_fields}

        return body

    def _create_document_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        """
        Create a new index for storing documents.
        """
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
                index_definition["settings"]["index"] = {"knn": True}  # TODO: option to turn off for script scoring
                # global ef_search setting affects only nmslib, for faiss it is set in the field mapping
                if self.knn_engine == "nmslib" and self.index_type == "hnsw":
                    ef_search = self._get_ef_search_value()
                    index_definition["settings"]["index"]["knn.algo_param.ef_search"] = ef_search
                index_definition["mappings"]["properties"][self.embedding_field] = self._get_embedding_field_mapping(
                    index=index_name
                )

        try:
            self.client.indices.create(index=index_name, body=index_definition, headers=headers)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self._index_exists(index_name, headers=headers):
                raise e

    def train_index(
        self,
        documents: Optional[Union[List[dict], List[Document]]] = None,
        embeddings: Optional[np.ndarray] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Trains an IVF index on the provided Documents or embeddings if the index hasn't been trained yet.

        The train vectors should come from the same distribution as your final vectors.
        You can pass either Documents (including embeddings) or just plain embeddings you want to train the index on.

        :param documents: Documents (including the embeddings) you want to train the index on.
        :param embeddings: Plain embeddings you want to train the index on.
        :param index: Name of the index to train. If `None`, the DocumentStore's default index (self.index) is used.
        :param headers: Custom HTTP headers to pass to the OpenSearch client (for example {'Authorization': 'Basic YWRtaW46cm9vdA=='}).
                For more information, see [HTTP/REST clients and security](https://www.elastic.co/guide/en/elasticsearch/reference/current/http-clients.html).
        """
        if self.index_type not in ["ivf", "ivf_pq"]:
            raise DocumentStoreError(
                "You can only train an index if you set `index_type` to 'ivf' or 'ivf_pq' in your DocumentStore. "
                "Other index types don't require training."
            )

        if index is None:
            index = self.index

        if isinstance(embeddings, np.ndarray) and documents:
            raise ValueError("Pass either `documents` or `embeddings`. You passed both.")

        if documents:
            document_objects = [Document.from_dict(d) if isinstance(d, dict) else d for d in documents]
            document_objects = [doc for doc in document_objects if doc.embedding is not None]
            self._train_ivf_index(index=index, documents=document_objects, headers=headers)
        elif isinstance(embeddings, np.ndarray):
            document_objects = [
                Document(content=f"Embedding {i}", embedding=embedding) for i, embedding in enumerate(embeddings)
            ]
            self._train_ivf_index(index=index, documents=document_objects, headers=headers)
        else:
            logger.warning(
                "When calling `train_index`, you must provide either Documents or embeddings. "
                "Because none of these values was provided, the index won't be trained."
            )

    def delete_index(self, index: str):
        """
        Delete an existing search index. The index together with all data will be removed.
        If the index is of type `"ivf"` or `"ivf_pq"`, this method also deletes the corresponding IVF and PQ model.

        :param index: The name of the index to delete.
        :return: None
        """
        # Check if index uses an IVF model and delete it
        index_mapping = self.client.indices.get(index)[index]["mappings"]["properties"]
        if self.embedding_field in index_mapping and "model_id" in index_mapping[self.embedding_field]:
            model_id = index_mapping[self.embedding_field]["model_id"]
            self.client.transport.perform_request("DELETE", f"/_plugins/_knn/models/{model_id}")

        super().delete_index(index)

    def _validate_and_adjust_document_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
        """
        Validates an existing document index. If there's no embedding field, we'll add it.
        """
        indices = self.client.indices.get(index_name, headers=headers)

        if not any(indices):
            # We don't want to raise here as creating a query-only document store before the index being created asynchronously is a valid use case.
            logger.warning(
                "Before you can use an index, you must create it first. The index '%s' doesn't exist. "
                "You can create it by setting `create_index=True` on init or by calling `write_documents()` if you prefer to create it on demand. "
                "Note that this instance doesn't validate the index after you created it.",
                index_name,
            )

        # If the index name is an alias that groups multiple existing indices, each of them must have an embedding_field.
        for index_id, index_info in indices.items():
            mappings = index_info["mappings"]
            index_settings = index_info["settings"]["index"]

            # validate fulltext fields
            if self.search_fields:
                for search_field in self.search_fields:
                    if search_field in mappings["properties"]:
                        if mappings["properties"][search_field]["type"] != "text":
                            raise DocumentStoreError(
                                f"The index '{index_id}' needs the 'text' type for the search_field '{search_field}' to run full text search, but got type '{mappings['properties'][search_field]['type']}'. "
                                f"You can fix this issue in one of the following ways: "
                                f" - Recreate the index by setting `recreate_index=True` (Note that you'll lose all data stored in the index.) "
                                f" - Use another index name by setting `index='my_index_name'`. "
                                f" - Remove '{search_field}' from `search_fields`. "
                            )
                    else:
                        mappings["properties"][search_field] = (
                            {"type": "text", "analyzer": "synonym"} if self.synonyms else {"type": "text"}
                        )
                        self.client.indices.put_mapping(index=index_id, body=mappings, headers=headers)

            # validate embedding field
            existing_embedding_field = mappings["properties"].get(self.embedding_field, None)

            if existing_embedding_field is None:
                # create embedding field
                mappings["properties"][self.embedding_field] = self._get_embedding_field_mapping(index=index_name)
                self.client.indices.put_mapping(index=index_id, body=mappings, headers=headers)
            else:
                # check type of existing embedding field
                if existing_embedding_field["type"] != "knn_vector":
                    raise DocumentStoreError(
                        f"The index '{index_id}' needs the 'knn_vector' type for the embedding_field '{self.embedding_field}' to run vector search, but got type '{mappings['properties'][self.embedding_field]['type']}'. "
                        f"You can fix it in one of these ways: "
                        f" - Recreate the index by setting `recreate_index=True` (Note that you'll lose all data stored in the index.) "
                        f" - Use another index name by setting `index='my_index_name'`. "
                        f" - Use another embedding field name by setting `embedding_field='my_embedding_field_name'`. "
                    )

                # Check if existing embedding field fits desired knn settings
                training_required = self.index_type in ["ivf", "ivf_pq"] and "model_id" not in existing_embedding_field
                if self.knn_engine != "score_script" and not training_required:
                    self._validate_approximate_knn_settings(existing_embedding_field, index_settings, index_id)

            # Adjust global ef_search setting (nmslib only).
            if self.knn_engine == "nmslib":
                ef_search = index_settings.get("knn.algo_param", {}).get("ef_search", 512)
                desired_ef_search = self._get_ef_search_value()
                if self.index_type == "hnsw" and ef_search != desired_ef_search:
                    body = {"knn.algo_param.ef_search": desired_ef_search}
                    self.client.indices.put_settings(index=index_id, body=body, headers=headers)
                    logger.info("Set ef_search to 20 for hnsw index '%s'.", index_id)
                elif self.index_type == "flat" and ef_search != 512:
                    body = {"knn.algo_param.ef_search": 512}
                    self.client.indices.put_settings(index=index_id, body=body, headers=headers)
                    logger.info("Set ef_search to 512 for hnsw index '%s'.", index_id)

    def _validate_approximate_knn_settings(
        self, existing_embedding_field: Dict[str, Any], index_settings: Dict[str, Any], index_id: str
    ):
        """
        Checks if the existing embedding field fits the desired approximate knn settings.
        If not, it will raise an error.
        If settings are not specified we infer the same default values as https://opensearch.org/docs/latest/search-plugins/knn/knn-index/
        """
        method = existing_embedding_field.get("method", {})
        if "model_id" in existing_embedding_field:
            embedding_field_knn_engine = "faiss"
        else:
            embedding_field_knn_engine = method.get("engine", "nmslib")
        embedding_field_space_type = method.get("space_type", "l2")

        # Validate knn engine
        if embedding_field_knn_engine != self.knn_engine:
            raise DocumentStoreError(
                f"Existing embedding field '{self.embedding_field}' of OpenSearch index '{index_id}' has knn_engine "
                f"'{embedding_field_knn_engine}', but knn_engine was set to '{self.knn_engine}'. "
                f"To switch knn_engine to '{self.knn_engine}' consider one of these options: "
                f" - Clone the embedding field in the same index, for example,  `clone_embedding_field(knn_engine='{self.knn_engine}', ...)`. "
                f" - Create a new index by selecting a different index name, for example, `index='my_new_{self.knn_engine}_index'`. "
                f" - Overwrite the existing index by setting `recreate_index=True`. Note that you'll lose all existing data."
            )

        # Validate space type
        if embedding_field_space_type != self.space_type:
            supported_similaries = [
                k
                for k, v in SIMILARITY_SPACE_TYPE_MAPPINGS[embedding_field_knn_engine].items()
                if v == embedding_field_space_type
            ]
            raise DocumentStoreError(
                f"Set `similarity` to one of '{supported_similaries}' to properly use the embedding field '{self.embedding_field}' of index '{index_id}'. "
                f"Similarity '{self.similarity}' is not compatible with embedding field's space type '{embedding_field_space_type}', it requires '{self.space_type}'. "
                f"If you do want to switch `similarity` of an existing index, note that the dense retriever models have an affinity for a specific similarity function. "
                f"Switching the similarity function might degrade the performance of your model. "
                f"\n"
                f"If you don't want to change the existing index, you can still use similarity '{self.similarity}' by setting `knn_engine='score_script'`. "
                f"This might be slower because of the exact vector calculation. "
                f"For a fast ANN search with similarity '{self.similarity}', consider one of these options: "
                f" - Clone the embedding field in the same index, for example, `clone_embedding_field(similarity='{self.similarity}', ...)`. "
                f" - Create a new index by selecting a different index name, for example,  `index='my_new_{self.similarity}_index'`. "
                f" - Overwrite the existing index by setting `recreate_index=True`. Note that you'll lose all existing data."
            )

        # Validate HNSW indices
        if self.index_type in ["flat", "hnsw"]:
            self._validate_hnsw_settings(existing_embedding_field, index_settings, index_id)
        # Validate IVF indices
        elif self.index_type in ["ivf", "ivf_pq"]:
            self._validate_ivf_settings(existing_embedding_field, index_settings, index_id)
        else:
            raise DocumentStoreError("Unknown index_type. Must be one of 'flat', 'hnsw', 'ivf', or 'ivf_pq'.")

    def _validate_hnsw_settings(
        self, existing_embedding_field: Dict[str, Any], index_settings: Dict[str, Any], index_id: str
    ):
        method = existing_embedding_field.get("method", {})
        parameters = method.get("parameters", {})
        embedding_field_method_name = method.get("name", "hnsw")
        embedding_field_ef_construction = parameters.get("ef_construction", 512)
        embedding_field_m = parameters.get("m", 16)
        embedding_field_knn_engine = method.get("engine", "nmslib")
        # ef_search is configured in the index settings and not in the mapping for nmslib
        if embedding_field_knn_engine == "nmslib":
            embedding_field_ef_search = index_settings.get("knn.algo_param", {}).get("ef_search", 512)
        else:
            embedding_field_ef_search = parameters.get("ef_search", 512)

        # Check method params according to requested index_type
        # Indices of type "flat" that don't use "score_script" as knn_engine use an HNSW index optimized for accuracy
        if self.index_type == "flat":
            self._assert_embedding_param(
                name="method.name", actual=embedding_field_method_name, expected="hnsw", index_id=index_id
            )
            self._assert_embedding_param(
                name="ef_construction", actual=embedding_field_ef_construction, expected=512, index_id=index_id
            )
            self._assert_embedding_param(name="m", actual=embedding_field_m, expected=16, index_id=index_id)
            if self.knn_engine == "faiss":
                self._assert_embedding_param(
                    name="ef_search", actual=embedding_field_ef_search, expected=512, index_id=index_id
                )

        elif self.index_type == "hnsw":
            expected_ef_construction = self.knn_parameters.get("ef_construction", 80)
            expected_m = self.knn_parameters.get("m", 64)
            expected_ef_search = self.knn_parameters.get("ef_search", 20)
            self._assert_embedding_param(
                name="method.name", actual=embedding_field_method_name, expected="hnsw", index_id=index_id
            )
            self._assert_embedding_param(
                name="ef_construction",
                actual=embedding_field_ef_construction,
                expected=expected_ef_construction,
                index_id=index_id,
            )
            self._assert_embedding_param(name="m", actual=embedding_field_m, expected=expected_m, index_id=index_id)
            if self.knn_engine == "faiss":
                self._assert_embedding_param(
                    name="ef_search", actual=embedding_field_ef_search, expected=expected_ef_search, index_id=index_id
                )

    def _validate_ivf_settings(
        self, existing_embedding_field: Dict[str, Any], index_settings: Dict[str, Any], index_id: str
    ):
        # Index is not trained yet and should therefore be an HNSW index with default settings until index is trained
        if "model_id" in existing_embedding_field:
            model_endpoint = f"/_plugins/_knn/models/{existing_embedding_field['model_id']}"
            response = self.client.transport.perform_request("GET", url=model_endpoint)
            model_settings_list = [setting.split(":") for setting in response["description"].split()]
            model_settings = {k: (int(v) if v.isnumeric() else v) for k, v in model_settings_list}

            embedding_field_nlist = model_settings.get("nlist")
            embedding_field_nprobes = model_settings.get("nprobes")
            expected_nlist = self.knn_parameters.get("nlist", 4)
            expected_nprobes = self.knn_parameters.get("nprobes", 1)
            self._assert_embedding_param(
                name="nlist", actual=embedding_field_nlist, expected=expected_nlist, index_id=index_id
            )
            self._assert_embedding_param(
                name="nprobes", actual=embedding_field_nprobes, expected=expected_nprobes, index_id=index_id
            )
            if self.index_type == "ivf_pq":
                embedding_field_m = model_settings.get("m")
                embedding_field_code_size = model_settings.get("code_size")
                expected_m = self.knn_parameters.get("m", 1)
                expected_code_size = self.knn_parameters.get("code_size", 8)
                self._assert_embedding_param(name="m", actual=embedding_field_m, expected=expected_m, index_id=index_id)
                self._assert_embedding_param(
                    name="code_size", actual=embedding_field_code_size, expected=expected_code_size, index_id=index_id
                )

    def _assert_embedding_param(self, name: str, actual: Any, expected: Any, index_id: str) -> None:
        if actual != expected:
            message = (
                f"The index_type '{self.index_type}' needs '{expected}' as {name} value. "
                f"Currently, the value for embedding field '{self.embedding_field}' of index '{index_id}' is '{actual}'. "
                f"To use your embeddings with index_type '{self.index_type}', you can do one of the following: "
                f" - Clone the embedding field in the same index, for example, `clone_embedding_field(index_type='{self.index_type}', ...)`. "
                f" - Create a new index by selecting a different index name, for example,  `index='my_new_{self.index_type}_index'`. "
                f" - Overwrite the existing index by setting `recreate_index=True`. Note that you'll lose all existing data."
            )
            raise DocumentStoreError(message)

    def _get_embedding_field_mapping(
        self,
        knn_engine: Optional[str] = None,
        space_type: Optional[str] = None,
        index_type: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        index: Optional[str] = None,
    ) -> Dict[str, Any]:
        if space_type is None:
            space_type = self.space_type
        if knn_engine is None:
            knn_engine = self.knn_engine
        if index_type is None:
            index_type = self.index_type
        if embedding_dim is None:
            embedding_dim = self.embedding_dim
        if index is None:
            index = self.index

        embeddings_field_mapping = {"type": "knn_vector", "dimension": embedding_dim}
        if knn_engine != "score_script":
            method: dict = {"space_type": space_type, "engine": knn_engine}

            ef_construction = (
                80 if "ef_construction" not in self.knn_parameters else self.knn_parameters["ef_construction"]
            )
            ef_search = self._get_ef_search_value()
            m = 64 if "m" not in self.knn_parameters else self.knn_parameters["m"]

            if index_type == "flat":
                # We're using HNSW with knn_engines nmslib and faiss as they do not support exact knn.
                method["name"] = "hnsw"
                # use default parameters from https://opensearch.org/docs/1.2/search-plugins/knn/knn-index/
                # we need to set them explicitly as aws managed instances starting from version 1.2 do not support empty parameters
                method["parameters"] = {"ef_construction": 512, "m": 16}
            elif index_type == "hnsw":
                method["name"] = "hnsw"
                method["parameters"] = {"ef_construction": ef_construction, "m": m}
                # for nmslib this is a global index setting
                if knn_engine == "faiss":
                    method["parameters"]["ef_search"] = ef_search
            elif index_type in ["ivf", "ivf_pq"]:
                if knn_engine != "faiss":
                    raise DocumentStoreError("To use 'ivf' or 'ivf_pq as index_type, set knn_engine to 'faiss'.")
                # Check if IVF model already exists
                if self._ivf_model_exists(index):
                    logger.info("Using existing IVF model '%s-ivf' for index '%s'.", index, index)
                    embeddings_field_mapping = {"type": "knn_vector", "model_id": f"{index}-ivf"}
                    method = {}
                else:
                    # IVF indices require training before they can be initialized. Setting index_type to HNSW until
                    # index is trained
                    logger.info("Using index of type 'flat' for index '%s' until IVF model is trained.", index)
                    method = {}
            else:
                logger.error("Set index_type to either 'flat', 'hnsw', 'ivf', or 'ivf_pq'.")
                method["name"] = "hnsw"

            if method:
                embeddings_field_mapping["method"] = method

        return embeddings_field_mapping

    def _ivf_model_exists(self, index: str) -> bool:
        if self._index_exists(".opensearch-knn-models"):
            response = self.client.transport.perform_request("GET", "/_plugins/_knn/models/_search")
            existing_ivf_models = set(
                model["_source"]["model_id"]
                for model in response["hits"]["hits"]
                if model["_source"]["state"] != "failed"
            )
        else:
            existing_ivf_models = set()

        return f"{index}-ivf" in existing_ivf_models

    def _ivf_index_not_trained_error(self, index: str, headers: Optional[Dict[str, str]] = None):
        add_num_of_embs = ""
        if self.ivf_train_size != 0:
            embs_to_add = self.get_embedding_count(index=index, headers=headers) - self.ivf_train_size
            add_num_of_embs = (
                f"or add at least {embs_to_add} more embeddings to automatically start the " f"training process "
            )
        raise DocumentStoreError(
            f"Index of type '{self.index_type}' is not trained yet. Train the index manually using "
            f"`train_index` {add_num_of_embs}before querying it."
        )

    def _create_label_index(self, index_name: str, headers: Optional[Dict[str, str]] = None):
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
            if not self._index_exists(index_name, headers=headers):
                raise e

    def _get_vector_similarity_query(self, query_emb: np.ndarray, top_k: int):
        """
        Generate Elasticsearch query for vector similarity.
        """
        if self.knn_engine == "score_script":
            query: dict = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "knn_score",
                        "lang": "knn",
                        "params": {
                            "field": self.embedding_field,
                            "query_value": query_emb.tolist(),
                            "space_type": self.space_type,
                        },
                    },
                }
            }
        else:
            if self.knn_engine == "faiss" and self.similarity == "cosine":
                self.normalize_embedding(query_emb)

            query = {"bool": {"must": [{"knn": {self.embedding_field: {"vector": query_emb.tolist(), "k": top_k}}}]}}

        return query

    def _get_raw_similarity_score(self, score):
        # adjust scores according to https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn
        # and https://opensearch.org/docs/latest/search-plugins/knn/knn-score-script/

        # space type is required as criterion as there is no consistent similarity-to-space-type mapping accross knn engines
        if self.space_type == "innerproduct":
            if score > 1:
                score = score - 1
            else:
                score = -(1 / score - 1)
        elif self.space_type == "l2":
            score = 1 / score - 1
        elif self.space_type == "cosinesimil":
            if self.knn_engine == "score_script":
                score = score - 1
            else:
                score = -(1 / score - 2)

        return score

    def _train_ivf_index(
        self, index: Optional[str], documents: List[Document], headers: Optional[Dict[str, str]] = None
    ):
        """
        If the provided index is not an IVF index yet, this method trains it on the provided Documents
        and converts the index to an IVF index.
        """
        if index is None:
            index = self.index

        # Check if IVF index is already trained by checking if embedding mapping contains a model_id field
        if "model_id" in self.client.indices.get(index)[index]["mappings"]["properties"][self.embedding_field]:
            logger.info("IVF index '%s' is already trained. Skipping training.", index)
        # IVF model is not trained yet -> train it and convert HNSW index to IVF index
        else:
            nlist = self.knn_parameters.get("nlist", 4)
            nprobes = self.knn_parameters.get("nprobes", 1)
            recommended_train_size = self._recommended_ivf_train_size()
            documents = [doc for doc in documents if doc.embedding is not None]
            if len(documents) < nlist:
                raise DocumentStoreError(
                    f"IVF training requires the number of training samples to be greater than or "
                    f"equal to `nlist`. Number of provided training samples is `{len(documents)}` "
                    f"and nlist is `{nlist}`."
                )
            if len(documents) < recommended_train_size:
                logger.warning(
                    "Consider increasing the number of training samples to at least "
                    "`%i` to get a reliable %s index.",
                    recommended_train_size,
                    self.index_type,
                )
            # Create temporary index containing training embeddings
            self._create_document_index(index_name=f".{index}_ivf_training", headers=headers)
            self.write_documents(documents=documents, index=f".{index}_ivf_training", headers=headers)

            settings = f"index_type:{self.index_type} nlist:{nlist} nprobes:{nprobes}"
            training_req_body: Dict = {
                "training_index": f".{index}_ivf_training",
                "training_field": self.embedding_field,
                "dimension": self.embedding_dim,
                "method": {
                    "name": "ivf",
                    "engine": "faiss",
                    "space_type": self.space_type,
                    "parameters": {"nlist": nlist, "nprobes": nprobes},
                },
            }
            # Add product quantization
            if self.index_type == "ivf_pq":
                m = self.knn_parameters.get("m", 1)
                code_size = self.knn_parameters.get("code_size", 8)
                if code_size > 8:
                    raise DocumentStoreError(
                        f"code_size parameter for product quantization must be less than or equal to 8. "
                        f"Provided code_size is `{code_size}`."
                    )

                # see FAISS doc for details: https://github.com/facebookresearch/faiss/wiki/FAQ#can-i-ignore-warning-clustering-xxx-points-to-yyy-centroids
                n_clusters = 2**code_size
                if len(documents) < n_clusters:
                    raise DocumentStoreError(
                        f"PQ training requires the number of training samples to be greater than or "
                        f"equal to the number of clusters. Number of provided training samples is `{len(documents)}` "
                        f"and the number of clusters is `{n_clusters}`."
                    )

                encoder = {"name": "pq", "parameters": {"m": m, "code_size": code_size}}
                settings += f" m:{m} code_size:{code_size}"
                training_req_body["method"]["parameters"]["encoder"] = encoder

            training_req_body["description"] = settings
            logger.info("Training IVF index '%s' using {len(documents)} embeddings.", index)
            train_endpoint = f"/_plugins/_knn/models/{index}-ivf/_train"
            response = self.client.transport.perform_request(
                "POST", url=train_endpoint, headers=headers, body=training_req_body
            )
            ivf_model = response["model_id"]

            # Wait until model training is finished, _knn_model_trained uses a retry decorator
            if self._knn_model_trained(ivf_model, headers=headers):
                logger.info("Training of IVF index '%s' finished.", index)

            # Delete temporary training index
            self.client.indices.delete(index=f".{index}_ivf_training", headers=headers)

            # Clone original index to temporary one
            self.client.indices.add_block(index=index, block="read_only")
            self.client.indices.clone(
                index=index, target=f".{index}_temp", body={"settings": {"index": {"blocks": {"read_only": False}}}}
            )
            self.client.indices.put_settings(index=index, body={"index": {"blocks": {"read_only": False}}})
            self.client.indices.delete(index=index)

            # Reindex original index to newly created IVF index
            self._create_document_index(index_name=index, headers=headers)
            self.client.reindex(
                body={"source": {"index": f".{index}_temp"}, "dest": {"index": index}},
                params={"request_timeout": 24 * 60 * 60},
            )
            self.client.indices.delete(index=f".{index}_temp")

    def _recommended_ivf_train_size(self) -> int:
        """
        Calculates the minumum recommended number of training samples for IVF training as suggested in FAISS docs.
        https://github.com/facebookresearch/faiss/wiki/FAQ#can-i-ignore-warning-clustering-xxx-points-to-yyy-centroids
        """
        min_points_per_cluster = 39
        if self.index_type == "ivf":
            n_clusters = self.knn_parameters.get("nlist", 4)
            return n_clusters * min_points_per_cluster
        elif self.index_type == "ivf_pq":
            n_clusters = 2 ** self.knn_parameters.get("code_size", 8)
            return n_clusters * min_points_per_cluster
        else:
            raise DocumentStoreError(f"Invalid index type '{self.index_type}'.")

    @retry(retry=retry_if_not_result(bool), wait=wait_exponential(min=1, max=10))
    def _knn_model_trained(self, model_name: str, headers: Optional[Dict[str, str]] = None) -> bool:
        model_state_endpoint = f"/_plugins/_knn/models/{model_name}"
        response = self.client.transport.perform_request("GET", url=model_state_endpoint, headers=headers)
        model_state = response["state"]
        if model_state == "created":
            return True
        elif model_state == "failed":
            error_message = response["error"]
            raise DocumentStoreError(f"Failed to train the KNN model. Error message: {error_message}")

        return False

    def _get_ef_search_value(self) -> int:
        ef_search = 20 if "ef_search" not in self.knn_parameters else self.knn_parameters["ef_search"]
        return ef_search

    def _delete_index(self, index: str):
        if self._index_exists(index):
            self.client.indices.delete(index=index, ignore=[400, 404])
            self._delete_ivf_model(index)
            logger.info("Index '%s' deleted.", index)

    def _delete_ivf_model(self, index: str):
        """
        If index is an index of type 'ivf' or 'ivf_pq', this method deletes the corresponding IVF model.
        """
        if self._index_exists(".opensearch-knn-models"):
            response = self.client.transport.perform_request("GET", "/_plugins/_knn/models/_search")
            existing_ivf_models = set(model["_source"]["model_id"] for model in response["hits"]["hits"])
            if f"{index}-ivf" in existing_ivf_models:
                self.client.transport.perform_request("DELETE", f"/_plugins/_knn/models/{index}-ivf")

    def clone_embedding_field(
        self,
        new_embedding_field: str,
        similarity: str,
        batch_size: int = 10_000,
        knn_engine: Optional[str] = None,
        index_type: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        if knn_engine is None:
            knn_engine = self.knn_engine

        mapping = self.client.indices.get(self.index, headers=headers)[self.index]["mappings"]
        if new_embedding_field in mapping["properties"]:
            raise DocumentStoreError(
                f"{new_embedding_field} already exists with mapping {mapping['properties'][new_embedding_field]}"
            )

        space_type = SIMILARITY_SPACE_TYPE_MAPPINGS[knn_engine][similarity]
        mapping["properties"][new_embedding_field] = self._get_embedding_field_mapping(
            space_type=space_type, knn_engine=knn_engine, index_type=index_type
        )
        self.client.indices.put_mapping(index=self.index, body=mapping, headers=headers)

        document_count = self.get_document_count(headers=headers)
        result = self._get_all_documents_in_index(index=self.index, batch_size=batch_size, headers=headers)

        opensearch_logger = logging.getLogger("opensearch")
        original_log_level = opensearch_logger.getEffectiveLevel()
        try:
            opensearch_logger.setLevel(logging.CRITICAL)
            with tqdm(total=document_count, position=0, unit=" Docs", desc="Cloning embeddings") as progress_bar:
                for result_batch in get_batches_from_generator(result, batch_size):
                    document_batch = [self._convert_es_hit_to_document(hit) for hit in result_batch]
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
        finally:
            opensearch_logger.setLevel(original_log_level)


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
