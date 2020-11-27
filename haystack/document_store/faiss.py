import logging
from sys import platform
from pathlib import Path
from typing import Union, List, Optional, Dict
from tqdm import tqdm
import numpy as np

from haystack import Document
from haystack.document_store.sql import SQLDocumentStore
from haystack.retriever.base import BaseRetriever

if platform != 'win32' and platform != 'cygwin':
    import faiss
else:
    raise ModuleNotFoundError("FAISSDocumentStore on windows platform is not supported")

logger = logging.getLogger(__name__)


class FAISSDocumentStore(SQLDocumentStore):
    """
    Document store for very large scale embedding based dense retrievers like the DPR.

    It implements the FAISS library(https://github.com/facebookresearch/faiss)
    to perform similarity search on vectors.

    The document text and meta-data (for filtering) are stored using the SQLDocumentStore, while
    the vector embeddings are indexed in a FAISS Index.

    """

    def __init__(
        self,
        sql_url: str = "sqlite:///",
        index_buffer_size: int = 10_000,
        vector_dim: int = 768,
        faiss_index_factory_str: str = "Flat",
        faiss_index: Optional[faiss.swigfaiss.Index] = None,
        return_embedding: bool = False,
        update_existing_documents: bool = False,
        index: str = "document",
        **kwargs,
    ):
        """
        :param sql_url: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
                        deployment, Postgres is recommended.
        :param index_buffer_size: When working with large datasets, the ingestion process(FAISS + SQL) can be buffered in
                                  smaller chunks to reduce memory footprint.
        :param vector_dim: the embedding vector size.
        :param faiss_index_factory_str: Create a new FAISS index of the specified type.
                                        The type is determined from the given string following the conventions
                                        of the original FAISS index factory.
                                        Recommended options:
                                        - "Flat" (default): Best accuracy (= exact). Becomes slow and RAM intense for > 1 Mio docs.
                                        - "HNSW": Graph-based heuristic. If not further specified,
                                                  we use a RAM intense, but more accurate config:
                                                  HNSW256, efConstruction=256 and efSearch=256
                                        - "IVFx,Flat": Inverted Index. Replace x with the number of centroids aka nlist.
                                                          Rule of thumb: nlist = 10 * sqrt (num_docs) is a good starting point.
                                        For more details see:
                                        - Overview of indices https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
                                        - Guideline for choosing an index https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
                                        - FAISS Index factory https://github.com/facebookresearch/faiss/wiki/The-index-factory
                                        Benchmarks: XXX
        :param faiss_index: Pass an existing FAISS Index, i.e. an empty one that you configured manually
                            or one with docs that you used in Haystack before and want to load again.
        :param return_embedding: To return document embedding
        :param update_existing_documents: Whether to update any existing documents with the same ID when adding
                                          documents. When set as True, any document with an existing ID gets updated.
                                          If set to False, an error is raised if the document ID of the document being
                                          added already exists.
        :param index: Name of index in document store to use.
        """
        self.vector_dim = vector_dim

        if faiss_index:
            self.faiss_index = faiss_index
        else:
            self.faiss_index = self._create_new_index(vector_dim=self.vector_dim, index_factory=faiss_index_factory_str, **kwargs)
            if "ivf" in faiss_index_factory_str.lower():  # enable reconstruction of vectors for inverted index
                self.faiss_index.set_direct_map_type(faiss.DirectMap.Hashtable)

        self.index_buffer_size = index_buffer_size
        self.return_embedding = return_embedding
        super().__init__(
            url=sql_url,
            update_existing_documents=update_existing_documents,
            index=index
        )

    def _create_new_index(self, vector_dim: int, index_factory: str = "Flat", metric_type=faiss.METRIC_INNER_PRODUCT, **kwargs):
        if index_factory == "HNSW" and metric_type == faiss.METRIC_INNER_PRODUCT:
            # faiss index factory doesn't give the same results for HNSW IP, therefore direct init.
            # defaults here are similar to DPR codebase (good accuracy, but very high RAM consumption)
            n_links = kwargs.get("n_links", 128)
            index = faiss.IndexHNSWFlat(vector_dim, n_links, metric_type)
            index.hnsw.efSearch = kwargs.get("efSearch", 20)#20
            index.hnsw.efConstruction = kwargs.get("efConstruction", 80)#80
            logger.info(f"HNSW params: n_links: {n_links}, efSearch: {index.hnsw.efSearch}, efConstruction: {index.hnsw.efConstruction}")
        else:
            index = faiss.index_factory(vector_dim, index_factory, metric_type)
        return index

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None):
        """
        Add new documents to the DocumentStore.

        :param documents: List of `Dicts` or List of `Documents`. If they already contain the embeddings, we'll index
                          them right away in FAISS. If not, you can later call update_embeddings() to create & index them.
        :param index: (SQL) index name for storing the docs and metadata
        :return:
        """
        # vector index
        if not self.faiss_index:
            raise ValueError("Couldn't find a FAISS index. Try to init the FAISSDocumentStore() again ...")

        # doc + metadata index
        index = index or self.index
        document_objects = [Document.from_dict(d) if isinstance(d, dict) else d for d in documents]

        add_vectors = False if document_objects[0].embedding is None else True

        if self.update_existing_documents and add_vectors:
            logger.warning("You have enabled `update_existing_documents` feature and "
                           "`FAISSDocumentStore` does not support update in existing `faiss_index`.\n"
                           "Please call `update_embeddings` method to repopulate `faiss_index`")

        for i in range(0, len(document_objects), self.index_buffer_size):
            vector_id = self.faiss_index.ntotal
            if add_vectors:
                embeddings = [doc.embedding for doc in document_objects[i: i + self.index_buffer_size]]
                embeddings = np.array(embeddings, dtype="float32")
                self.faiss_index.add(embeddings)

            docs_to_write_in_sql = []
            for doc in document_objects[i : i + self.index_buffer_size]:
                meta = doc.meta
                if add_vectors:
                    meta["vector_id"] = vector_id
                    vector_id += 1
                docs_to_write_in_sql.append(doc)

            super(FAISSDocumentStore, self).write_documents(docs_to_write_in_sql, index=index)

    def update_embeddings(self, retriever: BaseRetriever, index: Optional[str] = None):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to get embeddings for text
        :param index: (SQL) index name for storing the docs and metadata
        :return: None
        """
        if not self.faiss_index:
            raise ValueError("Couldn't find a FAISS index. Try to init the FAISSDocumentStore() again ...")

        # Faiss does not support update in existing index data so clear all existing data in it
        self.faiss_index.reset()

        index = index or self.index
        documents = self.get_all_documents(index=index)

        if len(documents) == 0:
            logger.warning("Calling DocumentStore.update_embeddings() on an empty index")
            return

        # To clear out the FAISS index contents and frees all memory immediately that is in use by the index
        self.faiss_index.reset()

        logger.info(f"Updating embeddings for {len(documents)} docs...")
        embeddings = retriever.embed_passages(documents)  # type: ignore
        assert len(documents) == len(embeddings)
        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]

        logger.info("Indexing embeddings and updating vectors_ids...")
        for i in tqdm(range(0, len(documents), self.index_buffer_size)):
            vector_id_map = {}
            vector_id = self.faiss_index.ntotal
            embeddings = [doc.embedding for doc in documents[i: i + self.index_buffer_size]]
            embeddings = np.array(embeddings, dtype="float32")
            self.faiss_index.add(embeddings)

            for doc in documents[i: i + self.index_buffer_size]:
                vector_id_map[doc.id] = vector_id
                vector_id += 1
            self.update_vector_ids(vector_id_map, index=index)

    def get_all_documents(
            self,
            index: Optional[str] = None,
            filters: Optional[Dict[str, List[str]]] = None,
            return_embedding: Optional[bool] = None
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        """
        documents = super(FAISSDocumentStore, self).get_all_documents(index=index, filters=filters)
        if return_embedding is None:
            return_embedding = self.return_embedding
        if return_embedding:
            for doc in documents:
                if doc.meta and doc.meta.get("vector_id") is not None:
                    doc.embedding = self.faiss_index.reconstruct(int(doc.meta["vector_id"]))
        return documents

    def train_index(self, documents: Optional[Union[List[dict], List[Document]]], embeddings: Optional[np.array] = None):
        """
        Some FAISS indices (e.g. IVF) require initial "training" on a sample of vectors before you can add your final vectors.
        The train vectors should come from the same distribution as your final ones.
        You can pass either documents (incl. embeddings) or just the plain embeddings that the index shall be trained on.

        :param documents: Documents (incl. the embeddings)
        :param embeddings: Plain embeddings
        :return: None
        """

        if embeddings and documents:
            raise ValueError("Either pass `documents` or `embeddings`. You passed both.")
        if documents:
            document_objects = [Document.from_dict(d) if isinstance(d, dict) else d for d in documents]
            embeddings = [doc.embedding for doc in document_objects]
            embeddings = np.array(embeddings, dtype="float32")
        self.faiss_index.train(embeddings)

    def delete_all_documents(self, index=None):
        """
        Delete all documents from the document store.
        """
        index = index or self.index
        self.faiss_index.reset()
        super().delete_all_documents(index=index)

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
            raise Exception("Query filters are not implemented for the FAISSDocumentStore.")
        if not self.faiss_index:
            raise Exception("No index exists. Use 'update_embeddings()` to create an index.")

        if return_embedding is None:
            return_embedding = self.return_embedding
        index = index or self.index

        query_emb = query_emb.reshape(1, -1).astype(np.float32)
        score_matrix, vector_id_matrix = self.faiss_index.search(query_emb, top_k)
        vector_ids_for_query = [str(vector_id) for vector_id in vector_id_matrix[0] if vector_id != -1]

        documents = self.get_documents_by_vector_ids(vector_ids_for_query, index=index)

        #assign query score to each document
        scores_for_vector_ids: Dict[str, float] = {str(v_id): s for v_id, s in zip(vector_id_matrix[0], score_matrix[0])}
        for doc in documents:
            doc.score = scores_for_vector_ids[doc.meta["vector_id"]]  # type: ignore
            doc.probability = (doc.score + 1) / 2
            if return_embedding is True:
                doc.embedding = self.faiss_index.reconstruct(int(doc.meta["vector_id"]))

        return documents

    def save(self, file_path: Union[str, Path]):
        """
        Save FAISS Index to the specified file.

        :param file_path: Path to save to.
        :return: None
        """
        faiss.write_index(self.faiss_index, str(file_path))

    @classmethod
    def load(
            cls,
            faiss_file_path: Union[str, Path],
            sql_url: str,
            index_buffer_size: int = 10_000,
    ):
        """
        Load a saved FAISS index from a file and connect to the SQL database.
        Note: In order to have a correct mapping from FAISS to SQL,
              make sure to use the same SQL DB that you used when calling `save()`.

        :param faiss_file_path: Stored FAISS index file. Can be created via calling `save()`
        :param sql_url: Connection string to the SQL database that contains your docs and metadata.
        :param index_buffer_size: When working with large datasets, the ingestion process(FAISS + SQL) can be buffered in
                                  smaller chunks to reduce memory footprint.
        :return:
        """
        """
        """
        faiss_index = faiss.read_index(str(faiss_file_path))
        return cls(
            faiss_index=faiss_index,
            sql_url=sql_url,
            index_buffer_size=index_buffer_size,
            vector_dim=faiss_index.d
        )

