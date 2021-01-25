import logging
import time
from typing import Dict, Generator, List, Optional, Union
import numpy as np

from milvus import IndexType, MetricType, Milvus

from haystack import Document
from haystack.document_store.sql import SQLDocumentStore
from haystack.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


class MilvusDocumentStore(SQLDocumentStore):
    """
    Document store for very large scale embedding based dense retrievers like the DPR.
    It implements the Milvus (https://github.com/milvus-io/milvus)
    to perform similarity search on vectors.
    The document text and meta-data(for filtering) is stored using the SQLDocumentStore, while
    the vector embeddings are indexed in a Milvus Index.
    """

    def __init__(
            self,
            sql_url: str = "sqlite:///",
            server_uri: str = "tcp://localhost:19530",
            connection_pool: str = "SingletonThread",
            index: str = "document",
            vector_dim: int = 768,
            index_file_size: int = 2048,
            milvus_metric_type: MetricType = MetricType.IP,
            milvus_index_type: IndexType = IndexType.FLAT,
            update_existing_documents: bool = False,
            return_embedding: bool = False,
            **kwargs,
    ):
        """
        :param sql_url: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
                        deployment, Postgres is recommended.
        :param server_uri: Milvus server uri, it will automatically deduce protocol, host and port from uri.
        :param connection_pool: Connection pool type to connect with Milvus server
        :param index: Index name for text, embedding and metadata.
        :param vector_dim: The embedding vector size.
        :param index_file_size: File size for Milvus server embedding vector store.
        :param milvus_metric_type: Embedding vector search metrics by default it use L2
        :param milvus_index_type: default it use FLAT
        :param update_existing_documents: Whether to update any existing documents with the same ID when adding
                                          documents. When set as True, any document with an existing ID gets updated.
                                          If set to False, an error is raised if the document ID of the document being
                                          added already exists.
        :param base_document_store: Base document store to store text and metadata. Either SQL or ES store can be used.
        """
        self.milvus_server = Milvus(uri=server_uri, pool=connection_pool)
        self.vector_dim = vector_dim
        self.index_file_size = index_file_size
        self.milvus_metric_type = milvus_metric_type
        self.milvus_index_type = milvus_index_type
        self.index = index
        self._create_collection_and_index_if_not_exist(self.index)
        self.return_embedding = return_embedding

        super().__init__(
            url=sql_url,
            update_existing_documents=update_existing_documents,
            index=index
        )

    def __del__(self):
        return self.milvus_server.close()

    def _create_collection_and_index_if_not_exist(self, index: Optional[str] = None):
        index = index or self.index
        status, ok = self.milvus_server.has_collection(collection_name=index)
        if not ok:
            param = {
                'collection_name': index,
                'dimension': self.vector_dim,
                'index_file_size': self.index_file_size,
                'milvus_metric_type': self.milvus_metric_type
            }

            self.milvus_server.create_collection(param)

            index_param = {
                'M': 48,
                'efConstruction': 500
            }
            self.milvus_server.create_index(index, self.milvus_index_type, index_param)

    def _create_document_field_map(self) -> Dict:
        return {
            self.index: "embedding",
        }

    def write_documents(
            self, documents: Union[List[dict], List[Document]], index: Optional[str] = None, batch_size: int = 10_000
    ):
        index = index or self.index
        self._create_collection_and_index_if_not_exist(index)
        document_objects = [Document.from_dict(d) if isinstance(d, dict) else d for d in documents]

        add_vectors = False if document_objects[0].embedding is None else True

        for i in range(0, len(document_objects), self.index_file_size):
            vector_ids = []
            if add_vectors:
                embeddings = [doc.embedding for doc in document_objects[i: i + self.index_file_size]]
                vectors = [emb.tolist() for emb in embeddings]
                status, vector_ids = self.milvus_server.insert(collection_name=index, records=vectors)

            docs_to_write_in_sql = []
            for vector_id, doc in enumerate(document_objects[i: i + self.index_file_size]):
                meta = doc.meta
                if add_vectors:
                    meta["vector_id"] = str(vector_id) if len(vector_ids) == 0 else str(vector_ids[vector_id])
                docs_to_write_in_sql.append(doc)

            super().write_documents(docs_to_write_in_sql, index=index)

        self.milvus_server.flush([index])

    def update_embeddings(self, retriever: BaseRetriever, index: Optional[str] = None, batch_size: int = 10_000):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).
        :param retriever: Retriever to use to get embeddings for text
        :param index: Index name to update
        :return: None
        """
        index = index or self.index
        self.milvus_server.drop_collection(collection_name=index)
        self.milvus_server.flush([index])
        time.sleep(10)
        self._create_collection_and_index_if_not_exist(index)
        time.sleep(10)

        documents = self.get_all_documents(index=index)
        logger.info(f"Updating embeddings for {len(documents)} docs ...")
        embeddings = retriever.embed_passages(documents)  # type: ignore
        assert len(documents) == len(embeddings)
        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]

        vector_ids = []
        for i in range(0, len(documents), self.index_file_size):
            embeddings = [doc.embedding for doc in documents[i: i + self.index_file_size]]
            vectors = [emb.tolist() for emb in embeddings]
            vector_ids = [vid for vid in range(len(embeddings))]
            self.milvus_server.insert(collection_name=index, records=vectors, ids=vector_ids)

        doc_meta_to_update = []
        for vector_id, doc in enumerate(documents[i: i + self.index_file_size]):
            meta = doc.meta or {}
            if not doc.meta:
                doc.meta = meta
            meta["vector_id"] = str(vector_id) if len(vector_ids) == 0 else str(vector_ids[vector_id])
            doc_meta_to_update.append((doc.id, meta))

            for doc_id, meta in doc_meta_to_update:
                super().update_document_meta(id=doc_id, meta=meta)

        self.milvus_server.flush([index])
        self.milvus_server.compact(collection_name=index)

    def query_by_embedding(self,
                           query_emb: np.array,
                           filters: Optional[dict] = None,
                           top_k: int = 10,
                           index: Optional[str] = None,
                           return_embedding: Optional[bool] = None) -> List[Document]:
        if filters:
            raise Exception("Query filters are not implemented for the MilvusDocumentStore.")

        index = index or self.index
        status, ok = self.milvus_server.has_collection(collection_name=index)
        if not ok:
            raise Exception("No index exists. Use 'update_embeddings()` to create an index.")

        query_emb = query_emb.reshape(1, -1)
        search_param = {'ef': 4096}
        status, vector_id_matrix = self.milvus_server.search(
            collection_name=index,
            query_records=query_emb,
            top_k=top_k,
            params=search_param
        )
        vector_ids_for_query = []
        if len(vector_id_matrix) > 0:
            vector_ids_for_query = [str(vector_id.id) for vector_id in vector_id_matrix[0]]

        if len(vector_ids_for_query) > 0:
            documents = self.get_all_documents(filters={"vector_id": vector_ids_for_query}, index=index)
            # sort the documents as per query results
            documents = sorted(documents,
                               key=lambda doc: vector_ids_for_query.index(doc.meta["vector_id"]))  # type: ignore
        else:
            documents = []

        return documents

    def delete_all_documents(self, index=None, filters: Optional[Dict[str, List[str]]] = None):
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
        documents = super().get_all_documents_generator(
            index=index, filters=filters, batch_size=batch_size
        )
        if return_embedding is None:
            return_embedding = self.return_embedding

        for doc in documents:
            if return_embedding:
                if doc.meta and doc.meta.get("vector_id") is not None:
                    doc.embedding = self.faiss_index.reconstruct(int(doc.meta["vector_id"]))
            yield doc

    def get_all_documents(
            self,
            index: Optional[str] = None,
            filters: Optional[Dict[str, List[str]]] = None,
            return_embedding: Optional[bool] = None,
            batch_size: int = 10_000,
    ) -> List[Document]:
        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def get_documents_by_id(
            self, ids: List[str], index: Optional[str] = None, batch_size: int = 10_000
    ) -> List[Document]:
        documents = super().get_documents_by_id(ids=ids, index=index)
        if self.return_embedding:
            for doc in documents:
                if doc.meta and doc.meta.get("vector_id") is not None:
                    doc.embedding = self.faiss_index.reconstruct(int(doc.meta["vector_id"]))
        return documents

    def get_all_vectors(self, index=None) -> List[np.array]:
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
