from typing import Any, Dict, Generator, List, Optional, Union

import logging
import warnings
import numpy as np

from tqdm.auto import tqdm

try:
    from pymilvus import FieldSchema, CollectionSchema, Collection, connections, utility
    from pymilvus.client.abstract import QueryResult
    from pymilvus.client.types import DataType
except (ImportError, ModuleNotFoundError) as ie:
    from haystack.utils.import_utils import _optional_component_not_installed

    _optional_component_not_installed(__name__, "milvus2", ie)

from haystack.schema import Document, FilterType
from haystack.document_stores.sql import SQLDocumentStore
from haystack.document_stores.base import get_batches_from_generator
from haystack.nodes.retriever import DenseRetriever


logger = logging.getLogger(__name__)


class MilvusDocumentStore(SQLDocumentStore):
    """
    Limitations:
    Milvus 2.0 so far doesn't support the deletion of documents (https://github.com/milvus-io/milvus/issues/7130).
    Therefore, delete_documents() and update_embeddings() won't work yet.

    Differences to 1.x:
    Besides big architectural changes that impact performance and reliability 2.0 supports the filtering by scalar data types.
    For Haystack users this means you can now run a query using vector similarity and filter for some meta data at the same time!
    (See https://milvus.io/docs/v2.0.x/comparison.md for more details)

    Usage:
    1. Start a Milvus service via docker (see https://milvus.io/docs/v2.0.x/install_standalone-docker.md)
    2. Run pip install farm-haystack[milvus]
    3. Init a MilvusDocumentStore() in Haystack

    Overview:
    Milvus (https://milvus.io/) is a highly reliable, scalable Document Store specialized on storing and processing vectors.
    Therefore, it is particularly suited for Haystack users that work with dense retrieval methods (like DPR).

    In contrast to FAISS, Milvus ...
     - runs as a separate service (e.g. a Docker container) and can scale easily in a distributed environment
     - allows dynamic data management (i.e. you can insert/delete vectors without recreating the whole index)
     - encapsulates multiple ANN libraries (FAISS, ANNOY ...)

    This class uses Milvus for all vector related storage, processing and querying.
    The meta-data (e.g. for filtering) and the document text are however stored in a separate SQL Database as Milvus
    does not allow these data types (yet).
    """

    def __init__(
        self,
        sql_url: str = "sqlite:///",
        host: str = "localhost",
        port: str = "19530",
        connection_pool: str = "SingletonThread",
        index: str = "document",
        vector_dim: Optional[int] = None,
        embedding_dim: int = 768,
        index_file_size: int = 1024,
        similarity: str = "dot_product",
        index_type: str = "IVF_FLAT",
        index_param: Optional[Dict[str, Any]] = None,
        search_param: Optional[Dict[str, Any]] = None,
        return_embedding: bool = False,
        embedding_field: str = "embedding",
        id_field: str = "id",
        custom_fields: Optional[List[Any]] = None,
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        isolation_level: Optional[str] = None,
        consistency_level: int = 0,
        recreate_index: bool = False,
    ):
        """
        :param sql_url: SQL connection URL for storing document texts and metadata. It defaults to a local, file based SQLite DB. For large scale
                        deployment, Postgres is recommended. If using MySQL then same server can also be used for
                        Milvus metadata. For more details see https://milvus.io/docs/v1.1.0/data_manage.md.
        :param milvus_url: Milvus server connection URL for storing and processing vectors.
                           Protocol, host and port will automatically be inferred from the URL.
                           See https://milvus.io/docs/v2.0.x/install_standalone-docker.md for instructions to start a Milvus instance.
        :param connection_pool: Connection pool type to connect with Milvus server. Default: "SingletonThread".
        :param index: Index name for text, embedding and metadata (in Milvus terms, this is the "collection name").
        :param vector_dim: Deprecated. Use embedding_dim instead.
        :param embedding_dim: The embedding vector size. Default: 768.
        :param index_file_size: Specifies the size of each segment file that is stored by Milvus and its default value is 1024 MB.
         When the size of newly inserted vectors reaches the specified volume, Milvus packs these vectors into a new segment.
         Milvus creates one index file for each segment. When conducting a vector search, Milvus searches all index files one by one.
         As a rule of thumb, we would see a 30% ~ 50% increase in the search performance after changing the value of index_file_size from 1024 to 2048.
         Note that an overly large index_file_size value may cause failure to load a segment into the memory or graphics memory.
         (From https://milvus.io/docs/v2.0.x/performance_faq.md)
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default and recommended for DPR embeddings.
                           'cosine' is recommended for Sentence Transformers, but is not directly supported by Milvus.
                           However, you can normalize your embeddings and use `dot_product` to get the same results.
                           See https://milvus.io/docs/v2.0.x/metric.md.
        :param index_type: Type of approximate nearest neighbour (ANN) index used. The choice here determines your tradeoff between speed and accuracy.
                           Some popular options:
                           - FLAT (default): Exact method, slow
                           - IVF_FLAT, inverted file based heuristic, fast
                           - HSNW: Graph based, fast
                           - ANNOY: Tree based, fast
                           See: https://milvus.io/docs/v2.0.x/index.md
        :param index_param: Configuration parameters for the chose index_type needed at indexing time.
                            For example: {"nlist": 16384} as the number of cluster units to create for index_type IVF_FLAT.
                            See https://milvus.io/docs/v2.0.x/index.md
        :param search_param: Configuration parameters for the chose index_type needed at query time
                             For example: {"nprobe": 10} as the number of cluster units to query for index_type IVF_FLAT.
                             See https://milvus.io/docs/v2.0.x/index.md
        :param return_embedding: To return document embedding.
        :param embedding_field: Name of field containing an embedding vector.
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param isolation_level: see SQLAlchemy's `isolation_level` parameter for `create_engine()` (https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.isolation_level)
        :param recreate_index: If set to True, an existing Milvus index will be deleted and a new one will be
            created using the config you are using for initialization. Be aware that all data in the old index will be
            lost if you choose to recreate the index. Be aware that both the document_index and the label_index will
            be recreated.
        """
        super().__init__(
            url=sql_url, index=index, duplicate_documents=duplicate_documents, isolation_level=isolation_level
        )

        connections.add_connection(default={"host": host, "port": port})
        connections.connect()

        if vector_dim is not None:
            warnings.warn(
                message="The 'vector_dim' parameter is deprecated, use 'embedding_dim' instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            self.embedding_dim = vector_dim
        else:
            self.embedding_dim = embedding_dim

        self.index_file_size = index_file_size
        self.similarity = similarity
        self.cosine = False

        if similarity == "dot_product":
            self.metric_type = "IP"
        elif similarity == "l2":
            self.metric_type = "L2"
        elif similarity == "cosine":
            self.metric_type = "IP"
            self.cosine = True
        else:
            raise ValueError(
                "The Milvus document store can currently only support dot_product, cosine and L2 similarity. "
                'Please set similarity="dot_product" or "cosine" or "l2"'
            )

        self.index_type = index_type
        self.index_param = index_param or {"nlist": 16384}
        self.search_param = search_param or {"nprobe": 10}
        self.index = index
        self.embedding_field = embedding_field
        self.id_field = id_field
        self.custom_fields = custom_fields

        self.collection = self._create_collection_and_index(
            self.index, consistency_level, recreate_index=recreate_index
        )

        self.return_embedding = return_embedding
        self.progress_bar = progress_bar

    def _create_collection_and_index(
        self,
        index: Optional[str] = None,
        consistency_level: int = 0,
        index_param: Optional[Dict[str, Any]] = None,
        recreate_index: bool = False,
    ):
        index = index or self.index
        index_param = index_param or self.index_param
        custom_fields = self.custom_fields or []

        if recreate_index:
            self._delete_index(index)
            super().delete_labels()

        has_collection = utility.has_collection(collection_name=index)
        if not has_collection:
            fields = [
                FieldSchema(name=self.id_field, dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name=self.embedding_field, dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ]

            for field in custom_fields:
                if field.name == self.id_field or field.name == self.embedding_field:
                    logger.warning("Skipping '%s' as it is similar to 'id_field' or 'embedding_field'", field.name)
                else:
                    fields.append(field)

            collection_schema = CollectionSchema(fields=fields)
        else:
            collection_schema = None

        collection = Collection(name=index, schema=collection_schema, consistency_level=consistency_level)

        has_index = collection.has_index()
        if not has_index:
            collection.create_index(
                field_name=self.embedding_field,
                index_params={"index_type": self.index_type, "metric_type": self.metric_type, "params": index_param},
            )

        collection.load()

        return collection

    def _create_document_field_map(self) -> Dict:
        return {self.index: self.embedding_field}

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        index_param: Optional[Dict[str, Any]] = None,
    ):
        """
        Add new documents to the DocumentStore.

        :param documents: List of `Dicts` or List of `Documents`. If they already contain the embeddings, we'll index
                                  them right away in Milvus. If not, you can later call `update_embeddings()` to create & index them.
        :param index: (SQL) index name for storing the docs and metadata
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return:
        """
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.index
        index_param = index_param or self.index_param
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert (
            duplicate_documents in self.duplicate_documents_options
        ), f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"
        field_map = self._create_document_field_map()

        if len(documents) == 0:
            logger.warning("Calling DocumentStore.write_documents() with empty list")
            return

        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(document_objects, duplicate_documents)
        add_vectors = False if document_objects[0].embedding is None else True

        batched_documents = get_batches_from_generator(document_objects, batch_size)
        with tqdm(total=len(document_objects), disable=not self.progress_bar) as progress_bar:
            mutation_result: Any = None

            for document_batch in batched_documents:
                if add_vectors:
                    doc_ids = []
                    embeddings = []
                    for doc in document_batch:
                        doc_ids.append(doc.id)
                        if isinstance(doc.embedding, np.ndarray):
                            if self.cosine:
                                embedding = doc.embedding / np.linalg.norm(doc.embedding)
                                embeddings.append(embedding.tolist())
                            else:
                                embeddings.append(doc.embedding.tolist())
                        elif isinstance(doc.embedding, list):
                            if self.cosine:
                                embedding = np.array(doc.embedding)
                                embedding /= np.linalg.norm(embedding)
                                embeddings.append(embedding.tolist())
                            else:
                                embeddings.append(doc.embedding)
                        else:
                            raise AttributeError(
                                f"Format of supplied document embedding {type(doc.embedding)} is not "
                                f"supported. Please use list or numpy.ndarray"
                            )
                    if duplicate_documents == "overwrite":
                        existing_docs = super().get_documents_by_id(ids=doc_ids, index=index)
                        self._delete_vector_ids_from_milvus(documents=existing_docs, index=index)

                    mutation_result = self.collection.insert([embeddings])

                docs_to_write_in_sql = []

                for idx, doc in enumerate(document_batch):
                    meta = doc.meta
                    if add_vectors and mutation_result is not None:
                        meta["vector_id"] = str(mutation_result.primary_keys[idx])
                    docs_to_write_in_sql.append(doc)

                super().write_documents(docs_to_write_in_sql, index=index, duplicate_documents=duplicate_documents)
                progress_bar.update(batch_size)
        progress_bar.close()

        # TODO: Equivalent in 2.0?

    #        if duplicate_documents == 'overwrite':
    #            connection.compact(collection_name=index)

    def update_embeddings(
        self,
        retriever: DenseRetriever,
        index: Optional[str] = None,
        batch_size: int = 10_000,
        update_existing_embeddings: bool = True,
        filters: Optional[FilterType] = None,
    ):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to get embeddings for text
        :param index: (SQL) index name for storing the docs and metadata
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param update_existing_embeddings: Whether to update existing embeddings of the documents. If set to False,
                                           only documents without embeddings are processed. This mode can be used for
                                           incremental updating of embeddings, wherein, only newly indexed documents
                                           get processed.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :return: None
        """
        index = index or self.index

        document_count = self.get_document_count(index=index)
        if document_count == 0:
            logger.warning("Calling DocumentStore.update_embeddings() on an empty index")
            return

        logger.info("Updating embeddings for %s docs...", document_count)

        result = self._query(
            index=index,
            vector_ids=None,
            batch_size=batch_size,
            filters=filters,
            only_documents_without_embedding=not update_existing_embeddings,
        )
        batched_documents = get_batches_from_generator(result, batch_size)
        with tqdm(
            total=document_count, disable=not self.progress_bar, position=0, unit=" docs", desc="Updating Embedding"
        ) as progress_bar:
            for document_batch in batched_documents:
                self._delete_vector_ids_from_milvus(documents=document_batch, index=index)

                embeddings = retriever.embed_documents(document_batch)
                self._validate_embeddings_shape(
                    embeddings=embeddings, num_documents=len(document_batch), embedding_dim=self.embedding_dim
                )

                if self.cosine:
                    self.normalize_embedding(embeddings)

                mutation_result = self.collection.insert([embeddings.tolist()])

                vector_id_map = {}
                for vector_id, doc in zip(mutation_result.primary_keys, document_batch):
                    vector_id_map[doc.id] = str(vector_id)

                self.update_vector_ids(vector_id_map, index=index)
                progress_bar.set_description_str("Documents Processed")
                progress_bar.update(batch_size)

        # TODO: Equivalent in 2.0?
        # self.milvus_server.compact(collection_name=index)

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
        :param filters: Optional filters to narrow down the search space.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param top_k: How many documents to return
        :param index: (SQL) index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :return:
        """
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.index
        has_collection = utility.has_collection(collection_name=index)
        if not has_collection:
            raise Exception("No index exists. Use 'update_embeddings()` to create an index.")

        if return_embedding is None:
            return_embedding = self.return_embedding

        query_emb = query_emb.reshape(-1).astype(np.float32)
        if self.cosine:
            query_emb = query_emb / np.linalg.norm(query_emb)

        search_result: QueryResult = self.collection.search(
            data=[query_emb.tolist()],
            anns_field=self.embedding_field,
            param={"metric_type": self.metric_type, **self.search_param},
            limit=top_k,
        )

        vector_ids_for_query = []
        scores_for_vector_ids: Dict[str, float] = {}
        for vector_id, distance in zip(search_result[0].ids, search_result[0].distances):
            vector_ids_for_query.append(str(vector_id))
            scores_for_vector_ids[str(vector_id)] = distance

        documents = self.get_documents_by_vector_ids(vector_ids_for_query, index=index)

        if return_embedding:
            self._populate_embeddings_to_docs(index=index, docs=documents)

        for doc in documents:
            score = scores_for_vector_ids[doc.meta["vector_id"]]
            if scale_score:
                score = self.scale_to_unit_interval(score, self.similarity)
            doc.score = score

        return documents

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: int = 10_000,
    ):
        """
        Delete all documents (from SQL AND Milvus).
        :param index: (SQL) index name for storing the docs and metadata
        :param filters: Optional filters to narrow down the search space.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :return: None
        """
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        if ids:
            self._delete_vector_ids_from_milvus(ids=ids, index=index)
        elif filters:
            batch = []
            for existing_docs in super().get_all_documents_generator(
                filters=filters, index=index, batch_size=batch_size
            ):
                batch.append(existing_docs)
                if len(batch) == batch_size:
                    self._delete_vector_ids_from_milvus(documents=batch, index=index)
            if len(batch) != 0:
                self._delete_vector_ids_from_milvus(documents=batch, index=index)
        else:
            self.collection = self._create_collection_and_index(self.index, recreate_index=True)

        index = index or self.index
        super().delete_documents(index=index, filters=filters, ids=ids)

    def delete_index(self, index: str):
        """
        Delete an existing index. The index including all data will be removed.

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
        if utility.has_collection(collection_name=index):
            utility.drop_collection(collection_name=index)
            logger.info("Index '%s' deleted.", index)
        super().delete_index(index)

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        Get all documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.index
        documents = super().get_all_documents_generator(index=index, filters=filters, batch_size=batch_size)
        if return_embedding is None:
            return_embedding = self.return_embedding

        for doc in documents:
            if return_embedding:
                self._populate_embeddings_to_docs(index=index, docs=[doc])
            yield doc

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Get documents from the document store (optionally using filter criteria).

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.index
        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def get_document_by_id(
        self, id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> Optional[Document]:
        """
        Fetch a document by specifying its text id string

        :param id: ID of the document
        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        """
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        documents = self.get_documents_by_id([id], index)
        document = documents[0] if documents else None
        return document

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Fetch multiple documents by specifying their IDs (strings)

        :param ids: List of IDs of the documents
        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.index
        documents = super().get_documents_by_id(ids=ids, index=index, batch_size=batch_size)
        if self.return_embedding:
            self._populate_embeddings_to_docs(index=index, docs=documents)

        return documents

    def _populate_embeddings_to_docs(self, docs: List[Document], index: Optional[str] = None):
        index = index or self.index
        docs_with_vector_ids = []
        for doc in docs:
            if doc.meta and doc.meta.get("vector_id") is not None:
                docs_with_vector_ids.append(doc)

        if len(docs_with_vector_ids) == 0:
            return

        ids = []
        vector_id_map = {}

        for doc in docs_with_vector_ids:
            vector_id: str = doc.meta["vector_id"]  # type: ignore
            # vector_id is always a string, but it isn't part of type hint
            ids.append(str(vector_id))
            vector_id_map[int(vector_id)] = doc

        search_result: QueryResult = self.collection.query(
            expr=f'{self.id_field} in [ {",".join(ids)} ]', output_fields=[self.embedding_field]
        )

        for result in search_result:
            doc = vector_id_map[result["id"]]
            doc.embedding = np.array(result["embedding"], "float32")

    def _delete_vector_ids_from_milvus(
        self, documents: Optional[List[Document]] = None, ids: Optional[List[str]] = None, index: Optional[str] = None
    ):
        index = index or self.index
        if ids is None:
            ids = []
            if documents is None:
                raise ValueError("You must either specify documents or ids to delete.")
            for doc in documents:
                if "vector_id" in doc.meta:
                    ids.append(str(doc.meta["vector_id"]))
        else:
            docs = super().get_documents_by_id(ids=ids, index=index)
            ids = [doc.meta["vector_id"] for doc in docs if "vector_id" in doc.meta]

        expr = f"{self.id_field} in [{','.join(ids)}]"

        self.collection.delete(expr)

    def get_embedding_count(self, index: Optional[str] = None, filters: Optional[FilterType] = None) -> int:
        """
        Return the count of embeddings in the document store.
        """
        if filters:
            raise Exception("filters are not supported for get_embedding_count in MilvusDocumentStore.")
        return len(self.get_all_documents(index=index))
