import logging
from typing import Any, Dict, Generator, List, Optional, Union
import numpy as np

from milvus import IndexType, MetricType, Milvus
from scipy.special import expit
from tqdm import tqdm

from haystack import Document
from haystack.document_store.sql import SQLDocumentStore
from haystack.retriever.base import BaseRetriever
from haystack.utils import get_batches_from_generator

logger = logging.getLogger(__name__)


class MilvusDocumentStore(SQLDocumentStore):
    """
    Document store for very large scale embedding based dense retrievers like the DPR.
    It implements the Milvus (https://github.com/milvus-io/milvus) to perform similarity search on vectors.
    The document text and meta-data (for filtering) is stored using the SQLDocumentStore, while the vector embeddings
    are indexed in a Milvus Server Index. Refer https://milvus.io/docs/v0.10.5/tuning.md for performance tuning option
    """

    def __init__(
            self,
            sql_url: str = "sqlite:///",
            server_uri: str = "tcp://localhost:19530",
            connection_pool: str = "SingletonThread",
            index: str = "document",
            vector_dim: int = 768,
            index_file_size: int = 1024,
            metric_type: MetricType = MetricType.IP,
            index_type: IndexType = IndexType.FLAT,
            index_param: Optional[Dict[str, Any]] = None,
            search_param: Optional[Dict[str, Any]] = None,
            update_existing_documents: bool = False,
            return_embedding: bool = False,
            embedding_field: str = "embedding",
            **kwargs,
    ):
        """
        :param sql_url: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
                        deployment, Postgres is recommended. If using MySQL then same server can also be used for
                        Milvus metadata. Refer for more detail https://milvus.io/docs/v0.10.5/data_manage.md.
        :param server_uri: Milvus server uri, it will automatically deduce protocol, host and port from uri.
        :param connection_pool: Connection pool type to connect with Milvus server by default it use SingletonThread.
        :param index: Index name for text, embedding and metadata.
        :param vector_dim: The embedding vector size by default it use 768 dimension.
        :param index_file_size: File size for Milvus server embedding vector store by default it use 1024 MB.
        :param metric_type: Embedding vector search metrics by default it use IP.
        :param index_type: Embedding vector indexing type by default it use FLAT.
        :param index_param: Embedding vector index creation parameter by default it use {"nlist": 16384}.
                            Refer for more information https://github.com/milvus-io/pymilvus/blob/master/doc/source/param.rst
        :param search_param: Embedding vector search parameter by default it use {"nprobe": 10}.
                             Refer for more information https://github.com/milvus-io/pymilvus/blob/master/doc/source/param.rst
        :param update_existing_documents: Whether to update any existing documents with the same ID when adding
                                          documents. When set as True, any document with an existing ID gets updated.
                                          If set to False, an error is raised if the document ID of the document being
                                          added already exists.
        :param return_embedding: To return document embedding.
        :param embedding_field: Name of field containing an embedding vector.
        """
        self.milvus_server = Milvus(uri=server_uri, pool=connection_pool)
        self.vector_dim = vector_dim
        self.index_file_size = index_file_size
        self.metric_type = metric_type
        self.index_type = index_type
        self.index_param = index_param or {"nlist": 16384}
        self.search_param = search_param or {"nprobe": 10}
        self.index = index
        self._create_collection_and_index_if_not_exist(self.index)
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field

        super().__init__(
            url=sql_url,
            update_existing_documents=update_existing_documents,
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

            self.milvus_server.create_collection(collection_param)

            self.milvus_server.create_index(index, self.index_type, index_param)

    def _create_document_field_map(self) -> Dict:
        return {
            self.index: self.embedding_field,
        }

    def write_documents(
            self, documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000
    ):
        """
        Add new documents to the DocumentStore.

        :param documents: List of `Dicts` or List of `Documents`. If they already contain the embeddings, we'll index
                                  them right away in FAISS. If not, you can later call update_embeddings() to create & index them.
        :param index: (SQL) index name for storing the docs and metadata
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return:
        """
        index = index or self.index
        self._create_collection_and_index_if_not_exist(index)
        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]

        add_vectors = False if document_objects[0].embedding is None else True

        for i in range(0, len(document_objects), batch_size):
            vector_ids = []
            if add_vectors:
                doc_ids = []
                embeddings = []
                for doc in document_objects[i: i + batch_size]:
                    doc_ids.append(doc.id)
                    if isinstance(doc.embedding, np.ndarray):
                        embeddings.append(doc.embedding.tolist())
                    elif isinstance(doc.embedding, list):
                        embeddings.append(doc.embedding)
                    else:
                        raise AttributeError("Document embedded in unrecognized format")

                if self.update_existing_documents:
                    existing_docs = super().get_documents_by_id(ids=doc_ids, index=index)
                    self._delete_vector_ids_from_milvus(documents=existing_docs, index=index)

                status, vector_ids = self.milvus_server.insert(collection_name=index, records=embeddings)

            docs_to_write_in_sql = []
            for idx, doc in enumerate(document_objects[i: i + batch_size]):
                meta = doc.meta
                if add_vectors:
                    meta["vector_id"] = vector_ids[idx]
                docs_to_write_in_sql.append(doc)

            super().write_documents(docs_to_write_in_sql, index=index)

        self.milvus_server.flush([index])
        if self.update_existing_documents:
            self.milvus_server.compact(collection_name=index)

    def update_embeddings(self, retriever: BaseRetriever, index: Optional[str] = None, batch_size: int = 10_000):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to get embeddings for text
        :param index: (SQL) index name for storing the docs and metadata
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return: None
        """
        index = index or self.index
        self._create_collection_and_index_if_not_exist(index)

        document_count = self.get_document_count(index=index)
        if document_count == 0:
            logger.warning("Calling DocumentStore.update_embeddings() on an empty index")
            return

        logger.info(f"Updating embeddings for {document_count} docs...")

        result = self.get_all_documents_generator(index=index, batch_size=batch_size, return_embedding=False)
        batched_documents = get_batches_from_generator(result, batch_size)
        with tqdm(total=document_count) as progress_bar:
            for document_batch in batched_documents:
                self._delete_vector_ids_from_milvus(documents=document_batch, index=index)

                embeddings = retriever.embed_passages(document_batch)  # type: ignore
                embeddings_list = [embedding.tolist() for embedding in embeddings]
                assert len(document_batch) == len(embeddings_list)

                status, vector_ids = self.milvus_server.insert(collection_name=index, records=embeddings_list)

                vector_id_map = {}
                for vector_id, doc in zip(vector_ids, document_batch):
                    vector_id_map[doc.id] = vector_id

                self.update_vector_ids(vector_id_map, index=index)
                progress_bar.update(batch_size)
        progress_bar.close()

        self.milvus_server.flush([index])
        self.milvus_server.compact(collection_name=index)

    def query_by_embedding(self,
                           query_emb: np.array,
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
        :return:
        """
        if filters:
            raise Exception("Query filters are not implemented for the MilvusDocumentStore.")

        index = index or self.index
        status, ok = self.milvus_server.has_collection(collection_name=index)
        if not ok:
            raise Exception("No index exists. Use 'update_embeddings()` to create an index.")

        if return_embedding is None:
            return_embedding = self.return_embedding
        index = index or self.index

        query_emb = query_emb.reshape(1, -1).astype(np.float32)
        status, search_result = self.milvus_server.search(
            collection_name=index,
            query_records=query_emb,
            top_k=top_k,
            params=self.search_param
        )
        vector_ids_for_query = []
        scores_for_vector_ids: Dict[str, float] = {}
        for vector_id_list, distance_list in zip(search_result.id_array, search_result.distance_array):
            for vector_id, distance in zip(vector_id_list, distance_list):
                vector_ids_for_query.append(vector_id)
                scores_for_vector_ids[str(vector_id)] = distance

        documents = self.get_documents_by_vector_ids(vector_ids_for_query, index=index)
        for doc in documents:
            doc.score = scores_for_vector_ids[doc.meta["vector_id"]]
            doc.probability = float(expit(np.asarray(doc.score / 100)))
            if return_embedding is True:
                doc.embedding = self.milvus_server.get_entity_by_id(
                    collection_name=index,
                    ids=[int(doc.meta["vector_id"])]
                )

        return documents

    def delete_all_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        index = index or self.index
        super().delete_all_documents(index=index, filters=filters)
        status, ok = self.milvus_server.has_collection(collection_name=index)
        if ok:
            self.milvus_server.drop_collection(collection_name=index)
            self.milvus_server.flush([index])
            self.milvus_server.compact(collection_name=index)

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
                if doc.meta and doc.meta.get("vector_id") is not None:
                    doc.embedding = self.milvus_server.get_entity_by_id(
                        collection_name=index,
                        ids=[int(doc.meta["vector_id"])]
                    )
            yield doc

    def get_all_documents(
            self,
            index: Optional[str] = None,
            filters: Optional[Dict[str, List[str]]] = None,
            return_embedding: Optional[bool] = None,
            batch_size: int = 10_000,
    ) -> List[Document]:
        index = index or self.index
        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def get_documents_by_id(
            self, ids: List[str], index: Optional[str] = None, batch_size: int = 10_000
    ) -> List[Document]:
        index = index or self.index
        documents = super().get_documents_by_id(ids=ids, index=index)
        if self.return_embedding:
            for doc in documents:
                if doc.meta and doc.meta.get("vector_id") is not None:
                    doc.embedding = self.milvus_server.get_entity_by_id(
                        collection_name=index,
                        ids=[int(doc.meta["vector_id"])]
                    )
        return documents

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
            if not status:
                raise RuntimeError("Unable to delete existing vector ids from Milvus server")

    def get_all_vectors(self, index=None) -> List[np.array]:
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
