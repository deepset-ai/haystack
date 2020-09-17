import logging
from pathlib import Path
from typing import Union, List, Optional, Dict

import faiss
import numpy as np
from faiss import Index, IndexIDMap

from haystack import Document
from haystack.document_store.sql import SQLDocumentStore
from haystack.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


class FaissIndexStore:
    """
    Index Store to keep all meta information about FAISS index. It abstract out vector transformation capability from
    FAISSDocumentStore. It perform L2 to IP transformation, L2 normalization for IP and perform index training if
    required.

    For details about indexes refer: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    Guideline to choose index refer: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    """
    def __init__(
            self,
            faiss_index: Optional[Index] = None,
            allow_training: bool = False,
            convert_l2_to_ip: bool = True,
            index_factory: Optional[str] = "HNSW4",
            dimension: Optional[int] = 768,
            metric_type: Optional[int] = None
    ):
        """
        :param faiss_index: Customized FAISS Index, useful if user want to customize index params like `nlists`,
                            `nbits`, `efSearch` etc instead of using default params.
        :param allow_training: Some IVF vector require training on GPU and some not as training is time consuming so
                               This flag determine whether to allow training even index's `is_trained` flag is `False`.
        :param convert_l2_to_ip: Mostly indexes on FAISS only support L2 distance hence this flag allow transformation
                                 to adds an additional dimension to obtain corresponding inner products.
                                 For more details refer:
        https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-do-max-inner-product-search-on-indexes-that-support-only-l2
        :param index_factory: String name to produce composite FAISS index. For more details refer:
        https://github.com/facebookresearch/faiss/wiki/The-index-factory
        :param dimension: The embedding vector size.
        :param metric_type: FAISS support Lx (L1, L2, Lp, LInf) and Inner Product metric type for indexes.
        """
        self.allow_training = allow_training
        self.convert_l2_to_ip = convert_l2_to_ip

        if faiss_index:
            self.faiss_index = faiss_index
            self.dimension = None
            self.index_factory = None
        else:
            self.index_factory = index_factory
            assert self.index_factory is not None

            metric_type = metric_type

            self.dimension = dimension
            assert self.dimension is not None

            # To adds an additional dimension to obtain corresponding inner products
            if self.convert_l2_to_ip:
                self.dimension += 1

            if metric_type is not None:
                self.faiss_index = faiss.index_factory(self.dimension, self.index_factory, metric_type)
            else:
                self.faiss_index = faiss.index_factory(self.dimension, self.index_factory)

    @staticmethod
    def _get_phi(embeddings: List[np.array]) -> int:
        """
        This will generate phi for vectors

        :param embeddings: List of vectors
        :return: phi value
        """
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

    def add_vectors(self, embeddings: List[np.array]) -> np.ndarray:
        """
        This add embeddings to FAISS index. Before adding embeddings to the index it perform following task -
         - L2 to IP transformation if `convert_l2_to_ip` flag is enabled.
         - Create contiguous array form embeddings.
         - Normalize vector via `normalize_L2` if metric of index is inner product. For more details refer:
         https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
         - Train index if index need training and `allow_training` is set
         - Generate vector ids to return them to caller and also to pass to index if index is IndexIDMap type

        :param embeddings: List of vectors
        :return: Vector ids
        """
        if self.convert_l2_to_ip:
            vectors_to_add = self._get_hnsw_vectors(embeddings, self._get_phi(embeddings))
        else:
            vectors_to_add = np.ascontiguousarray([emb.tolist() for emb in embeddings], dtype=np.float32)

        if self.faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(vectors_to_add)

        if not self.faiss_index.is_trained and self.allow_training:
            self.faiss_index.train(vectors_to_add)

        total_indices = self.size()
        ids = np.arange(total_indices, total_indices + len(embeddings), 1, dtype=np.int64)

        if isinstance(self.faiss_index, IndexIDMap):
            self.faiss_index.add_with_ids(vectors_to_add, ids)
        else:
            self.faiss_index.add(vectors_to_add)
        return ids

    def search_vectors(self, embeddings: List[np.array], top_k: int):
        """
        This perform similarity search of given vectors and return vector_ids and corresponding score. If
        `convert_l2_to_ip` flag is set then it perform transformation add an dimension.

        :param embeddings: List of search vectors
        :param top_k: How many search result required
        :return: vector_ids and corresponding score
        """
        vectors = embeddings
        if self.convert_l2_to_ip:
            aux_dim = np.zeros(len(embeddings), dtype="float32")
            vectors = np.hstack((embeddings, aux_dim.reshape(-1, 1)))

        return self.faiss_index.search(vectors, top_k)

    def size(self):
        """
        :return: Number of embeddings stored in the index
        """
        return self.faiss_index.ntotal

    def reset(self):
        """
        This delete the all stored embeddings from the index
        :return: None
        """
        self.faiss_index.reset()


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
            custom_index_store: Optional[FaissIndexStore] = None,
            index_factory: str = "HNSW4",
            index: str = "document"
    ):
        """
        Constructor for FAISSDocumentStore

        :param sql_url: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
                        deployment, Postgres is recommended.
        :param index_buffer_size: When working with large datasets, the ingestion process(FAISS + SQL) can be buffered
                                  in smaller chunks to reduce memory footprint.
        :param vector_size: The embedding vector size.
        :param custom_index_store: Customized FaissIndexStore, useful if user want to customize index params like
                                   `nlists`, `nbits`, `efSearch` etc instead of using default params.
        :param index_factory: String name to produce composite FAISS index. For more details refer:
        https://github.com/facebookresearch/faiss/wiki/The-index-factory
        :param index: Index name.
        """
        self.index = index
        self.vector_size = vector_size
        self.index_factory = index_factory
        self.index_dict: Dict[str, FaissIndexStore] = {}
        self.index_buffer_size = index_buffer_size

        if custom_index_store:
            self.get_or_create_index_store(index=self.index, index_store=custom_index_store)

        super().__init__(url=sql_url, index=index)

    def get_or_create_index_store(self, index: Optional[str] = None, index_store: Optional[FaissIndexStore] = None,
                                  **kwargs) -> FaissIndexStore:
        """
        This return FaissIndexStore associated with index, and if not exist then create it from provided parameters.
        Externally this function can be used to get existing index store to configure various FAISS index params like
        `nlists`, `nbits`, `efSearch` etc before performing write, update or query operation.

        :param index: Index name.
        :param index_store: If passed it will replace existing index store corresponding to index.
        :param kwargs: To pass parameters to FaissIndexStore for index creation.
        :return: Index store.
        """
        index = index or self.index
        assert index is not None

        if index_store:
            if self.index_dict and index in self.index_dict:
                # Clear old index before re-assignment
                self.index_dict[index].reset()
            self.index_dict[index] = index_store
        elif not self.index_dict or index not in self.index_dict:
            # Add important params like `index_factory` and `dimension` if not exist in kwargs.
            if kwargs and kwargs.get('index_factory') is None:
                kwargs['index_factory'] = self.index_factory

            if kwargs and kwargs.get('dimension') is None:
                kwargs['dimension'] = self.vector_size

            self.index_dict[index] = FaissIndexStore(**kwargs)
        return self.index_dict[index]

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None, **kwargs):

        index = index or self.index
        document_objects = [Document.from_dict(d) if isinstance(d, dict) else d for d in documents]
        add_vectors = False if document_objects[0].embedding is None else True
        faiss_index = self.get_or_create_index_store(index=index, **kwargs)

        for i in range(0, len(document_objects), self.index_buffer_size):
            vector_ids = []
            if add_vectors:
                embeddings_batch = [doc.embedding for doc in document_objects[i: i + self.index_buffer_size]]
                vector_ids = faiss_index.add_vectors(embeddings=embeddings_batch)

            docs_to_write_in_sql = []
            for idx, doc in enumerate(document_objects[i: i + self.index_buffer_size]):
                meta = doc.meta
                if add_vectors and len(vector_ids) > 0:
                    meta["vector_id"] = str(vector_ids[idx])
                docs_to_write_in_sql.append(doc)

            super(FAISSDocumentStore, self).write_documents(docs_to_write_in_sql, index=index)

    def update_embeddings(self, retriever: BaseRetriever, index: Optional[str] = None, **kwargs):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever
        config).

        :param retriever: Retriever to use to get embeddings for text.
        :param index: Index name to update.
        :return: None
        """
        index = index or self.index
        faiss_index = self.get_or_create_index_store(index=index, **kwargs)
        # Some FAISS indexes(like the default HNSWx) do not support removing vectors, so a new index is created.
        faiss_index.reset()

        documents = self.get_all_documents(index=index)
        logger.info(f"Updating embeddings for {len(documents)} docs ...")
        embeddings = retriever.embed_passages(documents)  # type: ignore
        assert len(documents) == len(embeddings)

        vector_id_map = {}
        for i in range(0, len(documents), self.index_buffer_size):
            embeddings_batch = [embedding for embedding in embeddings[i: i + self.index_buffer_size]]
            vector_ids = faiss_index.add_vectors(embeddings=embeddings_batch)

            for idx, doc in enumerate(documents[i: i + self.index_buffer_size]):
                vector_id_map[doc.id] = str(vector_ids[idx])
                vector_id += 1

        self.update_vector_ids(vector_id_map, index=index)

    def query_by_embedding(
            self, query_emb: np.array, filters: Optional[dict] = None, top_k: int = 10,
            index: Optional[str] = None
    ) -> List[Document]:

        index = index or self.index
        if index not in self.index_dict:
            raise Exception("No index exists. Use 'update_embeddings()` to create an index.")

        if filters:
            raise Exception("Query filters are not implemented for the FAISSDocumentStore.")

        faiss_index = self.index_dict[index]
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

    def delete_all_documents(self, index: Optional[str] = None):
        """
        This delete all documents and FAISS index for the corresponding index
        :param index: Index name.
        :return: None
        """
        index = index or self.index
        super(FAISSDocumentStore, self).delete_all_documents(index=index)
        if self.index_dict and index in self.index_dict:
            self.index_dict[index].reset()

    def save(self, file_path: Union[str, Path]):
        """
        Save FAISS Index to the specified file.
        """
        if self.index:
            faise_index_store = self.index_dict[self.index]
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
            custom_index_store=FaissIndexStore(faiss_index=faiss_index),
            sql_url=sql_url,
            index_buffer_size=index_buffer_size,
            vector_size=vector_size
        )
