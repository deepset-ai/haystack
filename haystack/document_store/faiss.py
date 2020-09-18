import logging
from pathlib import Path
from typing import Union, List, Optional, Dict

import faiss
import numpy as np
from faiss.swigfaiss import IndexHNSWFlat

from haystack import Document
from haystack.document_store.sql import SQLDocumentStore
from haystack.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


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
        faiss_index: Optional[IndexHNSWFlat] = None,
    ):
        """
        :param sql_url: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
                        deployment, Postgres is recommended.
        :param index_buffer_size: When working with large datasets, the ingestion process(FAISS + SQL) can be buffered in
                                  smaller chunks to reduce memory footprint.
        :param vector_size: the embedding vector size.
        :param faiss_index: load an existing FAISS Index.
        """
        self.vector_size = vector_size
        self.faiss_index = faiss_index
        self.index_buffer_size = index_buffer_size
        super().__init__(url=sql_url)

    def _create_new_index(self, vector_size: int, index_factory: str = "HNSW4"):
        index = faiss.index_factory(vector_size + 1, index_factory)
        return index

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None):
        if self.faiss_index is not None:
            raise Exception("Addition of more data in an existing index is not supported.")

        faiss_index = self._create_new_index(vector_size=self.vector_size)
        index = index or self.index
        document_objects = [Document.from_dict(d) if isinstance(d, dict) else d for d in documents]

        add_vectors = False if document_objects[0].embedding is None else True

        if add_vectors:
            phi = self._get_phi(document_objects)

        for i in range(0, len(document_objects), self.index_buffer_size):
            vector_id = faiss_index.ntotal
            if add_vectors:
                embeddings = [doc.embedding for doc in document_objects[i: i + self.index_buffer_size]]
                hnsw_vectors = self._get_hnsw_vectors(embeddings=embeddings, phi=phi)
                hnsw_vectors = hnsw_vectors.astype(np.float32)
                faiss_index.add(hnsw_vectors)

            docs_to_write_in_sql = []
            for doc in document_objects[i : i + self.index_buffer_size]:
                meta = doc.meta
                if add_vectors:
                    meta["vector_id"] = vector_id
                    vector_id += 1
                docs_to_write_in_sql.append(doc)

            super(FAISSDocumentStore, self).write_documents(docs_to_write_in_sql, index=index)

        self.faiss_index = faiss_index

    def _get_hnsw_vectors(self, embeddings: List[np.array], phi: int) -> np.array:
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

    def _get_phi(self, documents: List[Document]) -> int:
        phi = 0
        for doc in documents:
            norms = (doc.embedding ** 2).sum()  # type: ignore
            phi = max(phi, norms)
        return phi

    def update_embeddings(self, retriever: BaseRetriever, index: Optional[str] = None):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to get embeddings for text
        :param index: Index name to update
        :return: None
        """
        # Some FAISS indexes(like the default HNSWx) do not support removing vectors, so a new index is created.
        faiss_index = self._create_new_index(vector_size=self.vector_size)
        index = index or self.index

        documents = self.get_all_documents(index=index)
        logger.info(f"Updating embeddings for {len(documents)} docs ...")
        embeddings = retriever.embed_passages(documents)  # type: ignore
        assert len(documents) == len(embeddings)
        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]

        phi = self._get_phi(documents)
        doc_meta_to_update = []

        for i in range(0, len(documents), self.index_buffer_size):
            vector_id = faiss_index.ntotal
            embeddings = [doc.embedding for doc in documents[i: i + self.index_buffer_size]]
            hnsw_vectors = self._get_hnsw_vectors(embeddings=embeddings, phi=phi)
            hnsw_vectors = hnsw_vectors.astype(np.float32)
            faiss_index.add(hnsw_vectors)

            for doc in documents[i: i + self.index_buffer_size]:
                meta = doc.meta or {}
                meta["vector_id"] = vector_id
                vector_id += 1
                doc_meta_to_update.append((doc.id, meta))

        for doc_id, meta in doc_meta_to_update:
            super(FAISSDocumentStore, self).update_document_meta(id=doc_id, meta=meta)

        self.faiss_index = faiss_index

    def query_by_embedding(
        self, query_emb: np.array, filters: Optional[dict] = None, top_k: int = 10, index: Optional[str] = None
    ) -> List[Document]:
        if filters:
            raise Exception("Query filters are not implemented for the FAISSDocumentStore.")
        if not self.faiss_index:
            raise Exception("No index exists. Use 'update_embeddings()` to create an index.")
        query_emb = query_emb.reshape(1, -1).astype(np.float32)

        aux_dim = np.zeros(len(query_emb), dtype="float32")
        hnsw_vectors = np.hstack((query_emb, aux_dim.reshape(-1, 1)))
        score_matrix, vector_id_matrix = self.faiss_index.search(hnsw_vectors, top_k)
        vector_ids_for_query = [str(vector_id) for vector_id in vector_id_matrix[0] if vector_id != -1]

        documents = self.get_all_documents(filters={"vector_id": vector_ids_for_query}, index=index)
        # sort the documents as per query results
        documents = sorted(documents, key=lambda doc: vector_ids_for_query.index(doc.meta["vector_id"]))  # type: ignore

        # assign query score to each document
        scores_for_vector_ids: Dict[str, float] = {str(v_id): s for v_id, s in zip(vector_id_matrix[0], score_matrix[0])}
        for doc in documents:
            doc.score = scores_for_vector_ids[doc.meta["vector_id"]]  # type: ignore
            doc.probability = (doc.score + 1) / 2

        return documents

    def save(self, file_path: Union[str, Path]):
        """
        Save FAISS Index to the specified file.
        """
        faiss.write_index(self.faiss_index, str(file_path))

    @classmethod
    def load(
            cls,
            faiss_file_path: Union[str, Path],
            sql_url: str,
            index_buffer_size: int = 10_000,
            vector_size: int = 768
    ):
        """
        Load a saved FAISS index from a file and connect to the SQL database.
        """
        faiss_index = faiss.read_index(str(faiss_file_path))
        return cls(
            faiss_index=faiss_index,
            sql_url=sql_url,
            index_buffer_size=index_buffer_size,
            vector_size=vector_size
        )

