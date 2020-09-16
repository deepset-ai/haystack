import logging
from pathlib import Path
from typing import Union, List, Optional, Dict

import faiss
import numpy as np
from faiss import Index

from haystack import Document
from haystack.document_store.sql import SQLDocumentStore
from haystack.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


class FaissIndexStore:
    def __init__(
            self,
            faiss_index: Optional[Index] = None,
            # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
            # https://github.com/facebookresearch/faiss/wiki/The-index-factory
            **kwargs
    ):
        self.dimension = kwargs.get('dimension', 768)
        self.allow_training = kwargs.get('allow_training', False)
        self.convert_l2_to_ip = kwargs.get('convert_l2_to_ip', True)
        self.index_factory = kwargs.get('index_factory', 'HNSW4')
        metric_type = kwargs.get('metric_type', None)

        if faiss_index:
            self.faiss_index = faiss_index
        else:
            new_dimension = self.dimension
            if self.convert_l2_to_ip:
                new_dimension = self.dimension + 1

            if metric_type is not None:
                self.faiss_index = faiss.index_factory(new_dimension, self.index_factory, metric_type)
            else:
                self.faiss_index = faiss.index_factory(new_dimension, self.index_factory)

    @staticmethod
    def _get_phi(embeddings: List[np.array]) -> int:
        phi = 0
        for embedding in embeddings:
            norms = (embedding ** 2).sum()  # type: ignore
            phi = max(phi, norms)
        return phi

    @staticmethod
    def _get_hnsw_vectors(embeddings: List[np.array], phi: int) -> np.array:
        """
        HNSW indices in FAISS only support L2 distance. This transformation adds an additional dimension to obtain
        corresponding inner products.

        You can read ore details here:
        https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-do-max-inner-product-search-on-indexes-that-support-only-l2
        """
        vectors = [np.reshape(emb, (1, -1)) for emb in embeddings]
        norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
        aux_dims = [np.sqrt(phi - norm) for norm in norms]
        hnsw_vectors = [np.hstack((doc_vector, aux_dims[i].reshape(-1, 1))) for i, doc_vector in enumerate(vectors)]
        hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)
        return hnsw_vectors

    def add_vectors(self, embeddings: List[np.array]):
        if self.convert_l2_to_ip:
            vectors_to_add = self._get_hnsw_vectors(embeddings, self._get_phi(embeddings))
        else:
            vectors_to_add = np.ascontiguousarray([emb.tolist() for emb in embeddings], dtype=np.float32)

        # Refer https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
        if self.faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(vectors_to_add)

        if not self.faiss_index.is_trained and self.allow_training:
            self.faiss_index.train(vectors_to_add)

        return self.faiss_index.add(vectors_to_add)

    def search_vectors(self, embeddings: List[np.array], top_k: int):
        vectors = embeddings
        if self.convert_l2_to_ip:
            aux_dim = np.zeros(len(embeddings), dtype="float32")
            vectors = np.hstack((embeddings, aux_dim.reshape(-1, 1)))

        return self.faiss_index.search(vectors, top_k)

    def size(self):
        return self.faiss_index.ntotal

    def reset(self):
        return self.faiss_index.reset()


class FAISSDocumentStore(SQLDocumentStore):
    """
    Document store for very large scale embedding based dense retrievers like the DPR.

    It implements the FAISS library(https://github.com/facebookresearch/faiss)
    to perform similarity search on vectors.

    The document text and meta-data(for filtering) is stored using the SQLDocumentStore, while
    the vector embeddings are indexed in a FAISS Index.

    """

    def __init__(
            self,
            sql_url: str = "sqlite:///",
            index_buffer_size: int = 10_000,
            vector_size: int = 768,
            faiss_index: Optional[Index] = None,
            index_factory: str = "HNSW4",
            index: str = "document"
    ):
        """
        :param sql_url: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
                        deployment, Postgres is recommended.
        :param index_buffer_size: When working with large datasets, the ingestion process(FAISS + SQL) can be buffered in
                                  smaller chunks to reduce memory footprint.
        :param vector_size: the embedding vector size.
        :param faiss_index: load an existing FAISS Index.
        """
        self.index = index
        self.vector_size = vector_size
        self.index_factory = index_factory
        self.faiss_indexes = {}
        self.index_buffer_size = index_buffer_size

        if faiss_index:
            self.get_or_create_fiass_index(index_name=index, faiss_index=faiss_index)

        super().__init__(url=sql_url)

    def get_or_create_fiass_index(self, index: Optional[str] = None, faiss_index: Optional[Index] = None, **kwargs):
        index = index or self.index
        if not self.faiss_indexes or not self.faiss_indexes[index]:
            if kwargs and kwargs.get('index_factory') is None:
                kwargs['index_factory'] = self.index_factory

            if kwargs and kwargs.get('dimension') is None:
                kwargs['dimension'] = self.vector_size

            self.faiss_indexes[index] = FaissIndexStore(faiss_index=faiss_index, **kwargs)
        return self.faiss_indexes[index]

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None, **kwargs):

        index = index or self.index
        document_objects = [Document.from_dict(d) if isinstance(d, dict) else d for d in documents]
        add_vectors = False if document_objects[0].embedding is None else True

        vector_id = 0
        if add_vectors:
            faiss_index = self.get_or_create_fiass_index(index=index, **kwargs)
            vector_id = faiss_index.size()
            embeddings = [doc.embedding for doc in document_objects]
            faiss_index.add_vectors(embeddings=embeddings)

        for i in range(0, len(document_objects), self.index_buffer_size):
            docs_to_write_in_sql = []
            for doc in document_objects[i: i + self.index_buffer_size]:
                meta = doc.meta
                if add_vectors:
                    meta["vector_id"] = vector_id
                    vector_id += 1
                docs_to_write_in_sql.append(doc)

            super(FAISSDocumentStore, self).write_documents(docs_to_write_in_sql, index=index)

    def update_embeddings(self, retriever: BaseRetriever, index: Optional[str] = None, **kwargs):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to get embeddings for text
        :param index: Index name to update
        :return: None
        """
        index = index or self.index
        faiss_index = self.get_or_create_fiass_index(index=index, **kwargs)
        # Some FAISS indexes(like the default HNSWx) do not support removing vectors, so a new index is created.
        faiss_index.reset()
        vector_id = faiss_index.size()

        documents = self.get_all_documents(index=index)
        logger.info(f"Updating embeddings for {len(documents)} docs ...")
        embeddings = retriever.embed_passages(documents)  # type: ignore
        assert len(documents) == len(embeddings)

        vector_id_map = {}
        for i in range(0, len(documents), self.index_buffer_size):
            embeddings_batch = [embedding for embedding in embeddings[i: i + self.index_buffer_size]]
            faiss_index.add_vectors(embeddings=embeddings_batch)

            for doc in documents[i: i + self.index_buffer_size]:
                vector_id_map[doc.id] = vector_id
                vector_id += 1

        self.update_vector_ids(vector_id_map, index=index)

    def query_by_embedding(
            self, query_emb: np.array, filters: Optional[dict] = None, top_k: int = 10,
            index: Optional[str] = None
    ) -> List[Document]:

        index = index or self.index
        if not self.faiss_indexes[index]:
            raise Exception("No index exists. Use 'update_embeddings()` to create an index.")

        if filters:
            raise Exception("Query filters are not implemented for the FAISSDocumentStore.")

        faiss_index = self.faiss_indexes[index]
        query_emb = query_emb.reshape(1, -1)

        score_matrix, vector_id_matrix = faiss_index.search_vectors(embeddings=query_emb, top_k=top_k)
        vector_ids_for_query = [str(vector_id) for vector_id in vector_id_matrix[0] if vector_id != -1]

        documents = self.get_documents_by_vector_ids(vector_ids_for_query, index=index)

        # assign query score to each document
        scores_for_vector_ids: Dict[str, float] = {str(v_id): s for v_id, s in
                                                   zip(vector_id_matrix[0], score_matrix[0])}
        for doc in documents:
            doc.score = scores_for_vector_ids[doc.meta["vector_id"]]  # type: ignore
            doc.probability = (doc.score + 1) / 2

        return documents

    def delete_all_documents(self, index=None):
        index = index or self.index
        super(FAISSDocumentStore, self).delete_all_documents(index=index)
        if self.faiss_indexes and self.faiss_indexes[index] is not None:
            self.faiss_indexes[index].reset()

    def save(self, file_path: Union[str, Path]):
        """
        Save FAISS Index to the specified file.
        """
        if self.index:
            faise_index_store = self.faiss_indexes.get(self.index)
            faiss.write_index(faise_index_store.faiss_index, str(file_path))

    @classmethod
    def load(
            cls,
            faiss_file_path: Union[str, Path],
            sql_url: str,
            index_buffer_size: int = 10_000,
            vector_size: int = 768,
            index: str = "document",
            index_factory: str = "HNSW4"
    ):
        """
        Load a saved FAISS index from a file and connect to the SQL database.
        """
        faiss_index = faiss.read_index(str(faiss_file_path))
        return cls(
            index=index,
            index_factory=index_factory,
            faiss_index=faiss_index,
            sql_url=sql_url,
            index_buffer_size=index_buffer_size,
            vector_size=vector_size
        )
