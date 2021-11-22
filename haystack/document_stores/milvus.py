from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union

if TYPE_CHECKING:
    from haystack.nodes.retriever import BaseRetriever

import logging
import numpy as np
from tqdm import tqdm
from scipy.special import expit

from milvus import IndexType, MetricType, Milvus, Status

from haystack.schema import Document
from haystack.document_stores.sql import SQLDocumentStore
from haystack.document_stores.base import get_batches_from_generator


logger = logging.getLogger(__name__)


class MilvusDocumentStore(SQLDocumentStore):
    """
    Milvus (https://milvus.io/) is a highly reliable, scalable Document Store specialized on storing and processing vectors.
    Therefore, it is particularly suited for Haystack users that work with dense retrieval methods (like DPR).
    In contrast to FAISS, Milvus ...
     - runs as a separate service (e.g. a Docker container) and can scale easily in a distributed environment
     - allows dynamic data management (i.e. you can insert/delete vectors without recreating the whole index)
     - encapsulates multiple ANN libraries (FAISS, ANNOY ...)

    This class uses Milvus for all vector related storage, processing and querying.
    The meta-data (e.g. for filtering) and the document text are however stored in a separate SQL Database as Milvus
    does not allow these data types (yet).

    Usage:
    1. Start a Milvus server (see https://milvus.io/docs/v1.0.0/install_milvus.md)
    2. Init a MilvusDocumentStore in Haystack
    """
    def __init__(
            self,
            sql_url: str = "sqlite:///",
            milvus_url: str = "tcp://localhost:19530",
            connection_pool: str = "SingletonThread",
            index: str = "document",
            vector_dim: int = 768,
            index_file_size: int = 1024,
            similarity: str = "dot_product",
            index_type: IndexType = IndexType.FLAT,
            index_param: Optional[Dict[str, Any]] = None,
            search_param: Optional[Dict[str, Any]] = None,
            return_embedding: bool = False,
            embedding_field: str = "embedding",
            progress_bar: bool = True,
            duplicate_documents: str = 'overwrite',
            **kwargs,
    ):
        """
        :param sql_url: SQL connection URL for storing document texts and metadata. It defaults to a local, file based SQLite DB. For large scale
                        deployment, Postgres is recommended. If using MySQL then same server can also be used for
                        Milvus metadata. For more details see https://milvus.io/docs/v1.0.0/data_manage.md.
        :param milvus_url: Milvus server connection URL for storing and processing vectors.
                           Protocol, host and port will automatically be inferred from the URL.
                           See https://milvus.io/docs/v1.0.0/install_milvus.md for instructions to start a Milvus instance.
        :param connection_pool: Connection pool type to connect with Milvus server. Default: "SingletonThread".
        :param index: Index name for text, embedding and metadata (in Milvus terms, this is the "collection name").
        :param vector_dim: The embedding vector size. Default: 768.
        :param index_file_size: Specifies the size of each segment file that is stored by Milvus and its default value is 1024 MB.
         When the size of newly inserted vectors reaches the specified volume, Milvus packs these vectors into a new segment.
         Milvus creates one index file for each segment. When conducting a vector search, Milvus searches all index files one by one.
         As a rule of thumb, we would see a 30% ~ 50% increase in the search performance after changing the value of index_file_size from 1024 to 2048.
         Note that an overly large index_file_size value may cause failure to load a segment into the memory or graphics memory.
         (From https://milvus.io/docs/v1.0.0/performance_faq.md#How-can-I-get-the-best-performance-from-Milvus-through-setting-index_file_size)
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default and recommended for DPR embeddings.
                           'cosine' is recommended for Sentence Transformers.
        :param index_type: Type of approximate nearest neighbour (ANN) index used. The choice here determines your tradeoff between speed and accuracy.
                           Some popular options:
                           - FLAT (default): Exact method, slow
                           - IVF_FLAT, inverted file based heuristic, fast
                           - HSNW: Graph based, fast
                           - ANNOY: Tree based, fast
                           See: https://milvus.io/docs/v1.0.0/index.md
        :param index_param: Configuration parameters for the chose index_type needed at indexing time.
                            For example: {"nlist": 16384} as the number of cluster units to create for index_type IVF_FLAT.
                            See https://milvus.io/docs/v1.0.0/index.md
        :param search_param: Configuration parameters for the chose index_type needed at query time
                             For example: {"nprobe": 10} as the number of cluster units to query for index_type IVF_FLAT.
                             See https://milvus.io/docs/v1.0.0/index.md
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
        """
        # save init parameters to enable export of component config as YAML
        self.set_config(
            sql_url=sql_url, milvus_url=milvus_url, connection_pool=connection_pool, index=index, vector_dim=vector_dim,
            index_file_size=index_file_size, similarity=similarity, index_type=index_type, index_param=index_param,
            search_param=search_param, duplicate_documents=duplicate_documents,
            return_embedding=return_embedding, embedding_field=embedding_field, progress_bar=progress_bar,
        )

        self.milvus_server = Milvus(uri=milvus_url, pool=connection_pool)
        self.vector_dim = vector_dim
        self.index_file_size = index_file_size

        if similarity in ("dot_product", "cosine"):
            self.metric_type = MetricType.IP
            self.similarity = similarity
        elif similarity == "l2":
            self.metric_type = MetricType.L2
            self.similarity = similarity
        else:
            raise ValueError("The Milvus document store can currently only support dot_product, cosine and L2 similarity. "
                             "Please set similarity=\"dot_product\", \"cosine\" or \"l2\"")

        self.index_type = index_type
        self.index_param = index_param or {"nlist": 16384}
        self.search_param = search_param or {"nprobe": 10}
        self.index = index
        self._create_collection_and_index_if_not_exist(self.index)
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents

        super().__init__(
            url=sql_url,
            index=index
        )

    def __del__(self):
        return self.milvus_server.close()

    def _create_collection_and_index_if_not_exist(
        self,
        index: Optional[str] = None,
        index_param: Optional[Dict[str, Any]] = None
    ):
        index = index or self.index
        index_param = index_param or self.index_param

        status, ok = self.milvus_server.has_collection(collection_name=index)
        if not ok:
            collection_param = {
                'collection_name': index,
                'dimension': self.vector_dim,
                'index_file_size': self.index_file_size,
                'metric_type': self.metric_type
            }

            status = self.milvus_server.create_collection(collection_param)
            if status.code != Status.SUCCESS:
                raise RuntimeError(f'Collection creation on Milvus server failed: {status}')

            status = self.milvus_server.create_index(index, self.index_type, index_param)
            if status.code != Status.SUCCESS:
                raise RuntimeError(f'Index creation on Milvus server failed: {status}')

    def _create_document_field_map(self) -> Dict:
        return {
            self.index: self.embedding_field,
        }

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,
                        batch_size: int = 10_000, duplicate_documents: Optional[str] = None, index_param: Optional[Dict[str, Any]] = None):
        """
        Add new documents to the DocumentStore.

        :param documents: List of `Dicts` or List of `Documents`. If they already contain the embeddings, we'll index
                                  them right away in Milvus. If not, you can later call update_embeddings() to create & index them.
        :param index: (SQL) index name for storing the docs and metadata
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return: None
        """
        index = index or self.index
        index_param = index_param or self.index_param
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert duplicate_documents in self.duplicate_documents_options, \
            f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"
        self._create_collection_and_index_if_not_exist(index)
        field_map = self._create_document_field_map()

        if len(documents) == 0:
            logger.warning("Calling DocumentStore.write_documents() with empty list")
            return

        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(documents=document_objects,
                                                            index=index,
                                                            duplicate_documents=duplicate_documents)
        add_vectors = False if document_objects[0].embedding is None else True

        batched_documents = get_batches_from_generator(document_objects, batch_size)
        with tqdm(total=len(document_objects), disable=not self.progress_bar) as progress_bar:
            for document_batch in batched_documents:
                vector_ids = []
                if add_vectors:
                    doc_ids = []
                    embeddings = []
                    for doc in document_batch:
                        doc_ids.append(doc.id)
                        if isinstance(doc.embedding, np.ndarray):
                            if self.similarity=="cosine":
                                self.normalize_embedding(doc.embedding)
                            embeddings.append(doc.embedding.tolist())
                        elif isinstance(doc.embedding, list):
                            if self.similarity=="cosine":
                                # temp conversion to ndarray
                                np_embedding = np.array(doc.embedding, dtype="float32")
                                self.normalize_embedding(np_embedding)
                                embeddings.append(np_embedding.tolist())
                            else:
                                embeddings.append(doc.embedding)
                        else:
                            raise AttributeError(f'Format of supplied document embedding {type(doc.embedding)} is not '
                                                 f'supported. Please use list or numpy.ndarray')

                    if duplicate_documents == 'overwrite':
                        existing_docs = super().get_documents_by_id(ids=doc_ids, index=index)
                        self._delete_vector_ids_from_milvus(documents=existing_docs, index=index)

                    status, vector_ids = self.milvus_server.insert(collection_name=index, records=embeddings)
                    if status.code != Status.SUCCESS:
                        raise RuntimeError(f'Vector embedding insertion failed: {status}')

                docs_to_write_in_sql = []
                for idx, doc in enumerate(document_batch):
                    meta = doc.meta
                    if add_vectors:
                        meta["vector_id"] = vector_ids[idx]
                    docs_to_write_in_sql.append(doc)

                super().write_documents(docs_to_write_in_sql, index=index, duplicate_documents=duplicate_documents)
                progress_bar.update(batch_size)
        progress_bar.close()

        self.milvus_server.flush([index])
        if duplicate_documents == 'overwrite':
            self.milvus_server.compact(collection_name=index)

        # Milvus index creating should happen after the creation of the collection and after the insertion
        # of documents for maximum efficiency.
        # See (https://github.com/milvus-io/milvus/discussions/4939#discussioncomment-809303)
        status = self.milvus_server.create_index(index, self.index_type, index_param)
        if status.code != Status.SUCCESS:
            raise RuntimeError(f'Index creation on Milvus server failed: {status}')

    def update_embeddings(
        self,
        retriever: 'BaseRetriever',
        index: Optional[str] = None,
        batch_size: int = 10_000,
        update_existing_embeddings: bool = True,
        filters: Optional[Dict[str, List[str]]] = None,
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
        self._create_collection_and_index_if_not_exist(index)

        document_count = self.get_document_count(index=index)
        if document_count == 0:
            logger.warning("Calling DocumentStore.update_embeddings() on an empty index")
            return

        logger.info(f"Updating embeddings for {document_count} docs...")

        result = self._query(
            index=index,
            vector_ids=None,
            batch_size=batch_size,
            filters=filters,
            only_documents_without_embedding=not update_existing_embeddings
        )
        batched_documents = get_batches_from_generator(result, batch_size)
        with tqdm(total=document_count, disable=not self.progress_bar, position=0, unit=" docs",
                  desc="Updating Embedding") as progress_bar:
            for document_batch in batched_documents:
                self._delete_vector_ids_from_milvus(documents=document_batch, index=index)

                embeddings = retriever.embed_documents(document_batch) # type: ignore
                if self.similarity=="cosine":
                    for embedding in embeddings:
                        self.normalize_embedding(embedding)

                embeddings_list = [embedding.tolist() for embedding in embeddings]
                assert len(document_batch) == len(embeddings_list)

                status, vector_ids = self.milvus_server.insert(collection_name=index, records=embeddings_list)
                if status.code != Status.SUCCESS:
                    raise RuntimeError(f'Vector embedding insertion failed: {status}')

                vector_id_map = {}
                for vector_id, doc in zip(vector_ids, document_batch):
                    vector_id_map[doc.id] = vector_id

                self.update_vector_ids(vector_id_map, index=index)
                progress_bar.set_description_str("Documents Processed")
                progress_bar.update(batch_size)

        self.milvus_server.flush([index])
        self.milvus_server.compact(collection_name=index)

    def query_by_embedding(self,
                           query_emb: np.ndarray,
                           filters: Optional[dict] = None,
                           top_k: int = 10,
                           index: Optional[str] = None,
                           return_embedding: Optional[bool] = None) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param filters: Optional filters to narrow down the search space.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param top_k: How many documents to return
        :param index: (SQL) index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :return: list of Documents that are the most similar to `query_emb`
        """
        if filters:
            logger.warning("Query filters are not implemented for the MilvusDocumentStore.")

        index = index or self.index
        status, ok = self.milvus_server.has_collection(collection_name=index)
        if status.code != Status.SUCCESS:
            raise RuntimeError(f'Milvus has collection check failed: {status}')
        if not ok:
            raise Exception("No index exists. Use 'update_embeddings()` to create an index.")

        if return_embedding is None:
            return_embedding = self.return_embedding
        index = index or self.index

        if self.similarity=="cosine": self.normalize_embedding(query_emb)

        query_emb = query_emb.reshape(1, -1).astype(np.float32)

        status, search_result = self.milvus_server.search(
            collection_name=index,
            query_records=query_emb,
            top_k=top_k,
            params=self.search_param
        )
        if status.code != Status.SUCCESS:
            raise RuntimeError(f'Vector embedding search failed: {status}')

        vector_ids_for_query = []
        scores_for_vector_ids: Dict[str, float] = {}
        for vector_id_list, distance_list in zip(search_result.id_array, search_result.distance_array):
            for vector_id, distance in zip(vector_id_list, distance_list):
                vector_ids_for_query.append(str(vector_id))
                scores_for_vector_ids[str(vector_id)] = distance

        documents = self.get_documents_by_vector_ids(vector_ids_for_query, index=index)

        if return_embedding:
            self._populate_embeddings_to_docs(index=index, docs=documents)

        for doc in documents:
            raw_score = scores_for_vector_ids[doc.meta["vector_id"]]
            doc.score = self.finalize_raw_score(raw_score, self.similarity)

        return documents

    def delete_all_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete all documents (from SQL AND Milvus).
        :param index: (SQL) index name for storing the docs and metadata
        :param filters: Optional filters to narrow down the search space.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
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

        :param index: Index name to delete the document from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param ids: Optional list of IDs to narrow down the documents to be deleted.
        :param filters: Optional filters to narrow down the documents to be deleted.
            Example filters: {"name": ["some", "more"], "category": ["only_one"]}.
            If filters are provided along with a list of IDs, this method deletes the
            intersection of the two query results (documents that match the filters and
            have their ID in the list).
        :return: None
        """
        index = index or self.index
        status, ok = self.milvus_server.has_collection(collection_name=index)
        if status.code != Status.SUCCESS:
            raise RuntimeError(f'Milvus has collection check failed: {status}')
        if ok:
            if not filters and not ids:
                status = self.milvus_server.drop_collection(collection_name=index)
                if status.code != Status.SUCCESS:
                    raise RuntimeError(f'Milvus drop collection failed: {status}')
            else:
                affected_docs = super().get_all_documents(filters=filters, index=index)
                if ids:
                    affected_docs = [doc for doc in affected_docs if doc.id in ids]
                self._delete_vector_ids_from_milvus(documents=affected_docs, index=index)

            self.milvus_server.flush([index])
            self.milvus_server.compact(collection_name=index)

        # Delete from SQL at the end to allow the above .get_all_documents() to work properly
        super().delete_documents(index=index, ids=ids, filters=filters)

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
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
        index = index or self.index
        documents = super().get_all_documents_generator(
            index=index, filters=filters, batch_size=batch_size
        )
        if return_embedding is None:
            return_embedding = self.return_embedding

        for doc in documents:
            if return_embedding:
                self._populate_embeddings_to_docs(index=index, docs=[doc])
            yield doc

    def get_all_documents(
            self,
            index: Optional[str] = None,
            filters: Optional[Dict[str, List[str]]] = None,
            return_embedding: Optional[bool] = None,
            batch_size: int = 10_000,
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
        index = index or self.index
        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def get_document_by_id(self, id: str, index: Optional[str] = None) -> Optional[Document]:
        """
        Fetch a document by specifying its text id string

        :param id: ID of the document
        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        """
        documents = self.get_documents_by_id([id], index)
        document = documents[0] if documents else None
        return document

    def get_documents_by_id(
            self, ids: List[str], index: Optional[str] = None, batch_size: int = 10_000
    ) -> List[Document]:
        """
        Fetch multiple documents by specifying their IDs (strings)

        :param ids: List of IDs of the documents
        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """
        index = index or self.index
        documents = super().get_documents_by_id(ids=ids, index=index)
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

        ids = [int(doc.meta.get("vector_id")) for doc in docs_with_vector_ids] # type: ignore
        status, vector_embeddings = self.milvus_server.get_entity_by_id(
            collection_name=index,
            ids=ids
        )
        if status.code != Status.SUCCESS:
            raise RuntimeError(f'Getting vector embedding by id failed: {status}')

        for embedding, doc in zip(vector_embeddings, docs_with_vector_ids):
            doc.embedding = np.array(embedding, dtype="float32")

    def _delete_vector_ids_from_milvus(self, documents: List[Document], index: Optional[str] = None):
        index = index or self.index
        existing_vector_ids = []
        for doc in documents:
            if "vector_id" in doc.meta:
                existing_vector_ids.append(int(doc.meta["vector_id"]))
        if len(existing_vector_ids) > 0:
            status = self.milvus_server.delete_entity_by_id(
                collection_name=index,
                id_array=existing_vector_ids
            )
            if status.code != Status.SUCCESS:
                raise RuntimeError(f"Existing vector ids deletion failed: {status}")

    def get_all_vectors(self, index: Optional[str] = None) -> List[np.ndarray]:
        """
        Helper function to dump all vectors stored in Milvus server.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :return: List[np.array]: List of vectors.
        """
        index = index or self.index
        status, collection_info = self.milvus_server.get_collection_stats(collection_name=index)
        if not status.OK():
            logger.info(f"Failed fetch stats from store ...")
            return list()

        logger.debug(f"collection_info = {collection_info}")

        ids = list()
        partition_list = collection_info["partitions"]
        for partition in partition_list:
            segment_list = partition["segments"]
            for segment in segment_list:
                segment_name = segment["name"]
                status, id_list = self.milvus_server.list_id_in_segment(
                    collection_name=index,
                    segment_name=segment_name)
                logger.debug(f"{status}: segment {segment_name} has {len(id_list)} vectors ...")
                ids.extend(id_list)

        if len(ids) == 0:
            logger.info(f"No documents in the store ...")
            return list()

        status, vectors = self.milvus_server.get_entity_by_id(collection_name=index, ids=ids)
        if not status.OK():
            logger.info(f"Failed fetch document for ids {ids} from store ...")
            return list()

        return vectors

    def get_embedding_count(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None) -> int:
        """
        Return the count of embeddings in the document store.
        """
        if filters:
            raise Exception("filters are not supported for get_embedding_count in MilvusDocumentStore.")
        index = index or self.index
        _, embedding_count = self.milvus_server.count_entities(index)
        if embedding_count is None:
            embedding_count = 0
        return embedding_count
