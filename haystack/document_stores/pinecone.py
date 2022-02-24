from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from haystack.nodes.retriever import BaseRetriever

import json
import logging
from pathlib import Path
from typing import Union, List, Optional, Dict, Generator
from tqdm.auto import tqdm

import pinecone
import faiss
import numpy as np

from haystack.schema import Document
from haystack.document_stores.sql import SQLDocumentStore
from haystack.document_stores.base import get_batches_from_generator
from inspect import Signature, signature

logger = logging.getLogger(__name__)


class PineconeDocumentStore(SQLDocumentStore):
    """
    Document store for very large scale embedding based dense retrievers like the DPR.

    It implements the Pinecone vector database (https://www.pinecone.io)
    to perform similarity search on vectors.

    The document text is stored using the SQLDocumentStore, while
    the vector embeddings and metadata (for filtering) are indexed in a Pinecone Index.
    """

    top_k_limit = 10_000
    top_k_limit_vectors = 1_000

    def __init__(
        self,
        api_key: str,
        environment: str = "us-west1-gcp",
        sql_url: str = "sqlite:///pinecone_document_store.db",
        pinecone_index: Optional["pinecone.Index"] = None,
        vector_dim: int = 768,
        return_embedding: bool = False,
        index: str = "document",
        similarity: str = "cosine",
        replicas: int = 1,
        shards: int = 1,
        embedding_field: str = "embedding",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        **kwargs,
    ):
        """
        :param api_key: Pinecone vector database API key (https://app.pinecone.io)
        :param environment: Pinecone cloud environment uses "us-west1-gcp" by default. Other GCP and AWS regions are supported,
                            contact Pinecone if required.
        :param sql_url: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
                        deployment, Postgres is recommended.
        :param pinecone_index: pinecone-client Index object, an index will be initialized or loaded if not specified.
        :param vector_dim: the embedding vector size.
        :param return_embedding: To return document embedding
        :param index: Name of index in document store to use.
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default since it is
                   more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence-Transformer model.
                   In both cases, the returned values in Document.score are normalized to be in range [0,1]:
                   For `dot_product`: expit(np.asarray(raw_score / 100))
                   For `cosine`: (raw_score + 1) / 2
        :param replicas: The number of replicas. Replicas duplicate your index. They provide higher availability and
                         throughput.
        :param shards: The number of shards to be used in the index. We recommend you use 1 shard per 1GB of data.
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
        # Connect to Pinecone server using python client binding
        pinecone.init(api_key=api_key, environment=environment)

        # formal similarity string
        if similarity in ("dot_product", "cosine"):
            self.metric_type = similarity
        elif similarity in ("l2", "euclidean"):
            self.metric_type = "euclidean"
        else:
            raise ValueError(
                "The Pinecone document store can currently only support dot_product, cosine and euclidean metrics. "
                "Please set similarity to one of the above."
            )

        self.index = index
        self.vector_dim = vector_dim
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents

        # Pinecone index params
        self.replicas = replicas
        self.shards = shards

        # initialize dictionary of index connections
        self.pinecone_indexes: Dict[str, pinecone.Index] = {}
        clean_index = self._sanitize_index_name(index)
        if pinecone_index:
            self.pinecone_indexes[clean_index] = pinecone_index
        else:
            self.pinecone_indexes[clean_index] = self._create_index_if_not_exist(
                vector_dim=self.vector_dim,
                index=clean_index,
                metric_type=self.metric_type,
                replicas=self.replicas,
                shards=self.shards,
            )

        self.return_embedding = return_embedding
        self.embedding_field = embedding_field

        self.progress_bar = progress_bar

        super().__init__(
            url=sql_url, index=index, duplicate_documents=duplicate_documents  # no sanitation for SQL index name
        )

        self._validate_index_sync()

    def _sanitize_index_name(self, index: Optional[str]) -> Optional[str]:
        if index is None:
            return None
        elif "_" in index:
            return index.replace("_", "-").lower()
        else:
            return index.lower()

    def _create_index_if_not_exist(
        self,
        vector_dim: int,
        index: Optional[str] = None,
        metric_type: Optional[str] = "cosine",
        replicas: Optional[int] = 1,
        shards: Optional[int] = 1,
    ):
        """
        Create a new index for storing documents in case if an
        index with the name doesn't exist already.
        """
        index = index or self.index
        index = self._sanitize_index_name(index)

        # if index already loaded can skip
        if index in self.pinecone_indexes.keys():
            index_conn = self.pinecone_indexes[index]
        else:
            # search pinecone hosted indexes and create if it does not exist
            if index not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index, dimension=vector_dim, metric=metric_type, replicas=replicas, shards=shards
                )
            index_conn = pinecone.Index(index)

        # get index statistics
        stats = index_conn.describe_index_stats()
        dims = stats["dimension"]
        count = stats["namespaces"][""]["vector_count"] if stats["namespaces"].get("") else 0
        logger.info(f"Index statistics: name: {index}, embedding dimensions: {dims}, record count: {count}")
        # return index connection
        return index_conn

    def _convert_pinecone_result_to_document(self, result: dict, return_embedding: bool) -> Document:
        """
        Convert Pinecone result dict into haystack document object. This is more involved because
        weaviate search result dict varies between get and query interfaces.
        Weaviate get methods return the data items in properties key, whereas the query doesn't.
        """
        score = None
        content = ""

        id = result.get("id")
        score = result.get("score")
        embedding = result.get("values")
        meta = result.get("metadata")

        content_type = None
        if meta.get("contenttype") is not None:
            content_type = str(meta.pop("contenttype"))

        if return_embedding and embedding:
            embedding = np.asarray(embedding, dtype=np.float32)

        document = Document.from_dict(
            {
                "id": id,
                "content": content,
                "content_type": content_type,
                "meta": meta,
                "score": score,
                "embedding": embedding,
            }
        )
        return document

    def _validate_params_load_from_disk(self, sig: Signature, locals: dict, kwargs: dict):
        # TODO probably not needed
        raise NotImplementedError("_validate_params_load_from_disk not implemented for PineconeDocumentStore")
        allowed_params = ["faiss_index_path", "faiss_config_path", "self", "kwargs"]
        invalid_param_set = False

        for param in sig.parameters.values():
            if param.name not in allowed_params and param.default != locals[param.name]:
                invalid_param_set = True
                break

        if invalid_param_set or len(kwargs) > 0:
            raise ValueError("if faiss_index_path is passed no other params besides faiss_config_path are allowed.")

    def _validate_index_sync(self):
        # This check ensures the correct document database was loaded.
        # If it fails, make sure you provided the path to the database
        # used when creating the original Pinecone index
        if not self.get_document_count() == self.get_embedding_count():
            raise ValueError(
                "The number of documents present in the SQL database does not "
                "match the number of embeddings in Pinecone. Make sure your Pinecone "
                "index aligns to the same database that was used when creating the "
                "original index."
            )

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Add new documents to the DocumentStore.

        :param documents: List of `Dicts` or List of `Documents`. If they already contain the embeddings, we'll index
                          them right away in Pinecone. If not, you can later call update_embeddings() to create & index them.
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
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        index = index or self.index
        index = self._sanitize_index_name(index)
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert (
            duplicate_documents in self.duplicate_documents_options
        ), f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        if not self.pinecone_indexes.get(index):
            self.pinecone_indexes[index] = self._create_index_if_not_exist(
                vector_dim=self.vector_dim,
                index=index,
                metric_type=self.metric,
                replicas=self.replicas,
                shards=self.shards,
            )

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(
            documents=document_objects, index=index, duplicate_documents=duplicate_documents
        )
        if len(document_objects) > 0:
            add_vectors = False if document_objects[0].embedding is None else True
            # I don't think below is required
            """
            if self.duplicate_documents == "overwrite" and add_vectors:
                logger.warning("You have to provide `duplicate_documents = 'overwrite'` arg and "
                               "`FAISSDocumentStore` does not support update in existing `faiss_index`.\n"
                               "Please call `update_embeddings` method to repopulate `faiss_index`")
            """

            with tqdm(
                total=len(document_objects), disable=not self.progress_bar, position=0, desc="Writing Documents"
            ) as progress_bar:
                for i in range(0, len(document_objects), batch_size):
                    ids = [doc.id for doc in document_objects[i : i + batch_size]]
                    # TODO find way to identify long metadata fields and split these to be stored in SQL
                    metadata = [doc.meta for doc in document_objects[i : i + batch_size]]
                    if add_vectors:
                        embeddings = [doc.embedding for doc in document_objects[i : i + batch_size]]
                        embeddings_to_index = np.array(embeddings, dtype="float32")

                        if self.similarity == "cosine":
                            self.normalize_embedding(embeddings_to_index)
                        # TODO not sure if required to convert to list objects (maybe already are)
                        embeddings = [embed.tolist() for embed in embeddings]
                        vectors = zip(ids, embeddings, metadata)
                        self.pinecone_indexes[index].upsert(vectors=vectors)

                    docs_to_write_in_sql = []
                    for doc in document_objects[i : i + batch_size]:
                        # TODO I think this is not necessary as we have doc.id, before was required
                        # to map from doc.id to the integer 'vector_id' values used by faiss - but
                        # we do need to use vector_id as this is used by the sql doc store
                        # if add_vectors:
                        doc.meta["vector_id"] = doc.id
                        docs_to_write_in_sql.append(doc)
                    super(PineconeDocumentStore, self).write_documents(
                        docs_to_write_in_sql, index=index, duplicate_documents=duplicate_documents
                    )
                    progress_bar.update(batch_size)
            progress_bar.close()

    def _create_document_field_map(self) -> Dict:
        return {
            self.index: self.embedding_field,
        }

    def update_embeddings(
        self,
        retriever: "BaseRetriever",
        index: Optional[str] = None,
        update_existing_embeddings: bool = True,
        filters: Optional[Dict] = None,
        batch_size: int = 10_000,
    ):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever to use to get embeddings for text
        :param index: Index name for which embeddings are to be updated. If set to None, the default self.index is used.
        :param update_existing_embeddings: Whether to update existing embeddings of the documents. If set to False,
                                           only documents without embeddings are processed. This mode can be used for
                                           incremental updating of embeddings, wherein, only newly indexed documents
                                           get processed.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
                        Example: {"genre": {"$in": ["documentary", "action"]}},
                        more info on filtering syntax here https://www.pinecone.io/docs/metadata-filtering/
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return: None
        """
        if filters:
            raise Exception("update_embeddings does not support filtering.")

        index = index or self.index
        index = self._sanitize_index_name(index)

        if not self.pinecone_indexes.get(index):
            raise ValueError("Couldn't find a Pinecone index. Try to init the PineconeDocumentStore() again ...")

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
            only_documents_without_embedding=not update_existing_embeddings,
        )
        batched_documents = get_batches_from_generator(result, batch_size)
        with tqdm(
            total=document_count, disable=not self.progress_bar, position=0, unit=" docs", desc="Updating Embedding"
        ) as progress_bar:
            for document_batch in batched_documents:
                embeddings = retriever.embed_documents(document_batch)  # type: ignore
                assert len(document_batch) == len(embeddings)

                embeddings_to_index = np.array(embeddings, dtype="float32")

                if self.similarity == "cosine":
                    self.normalize_embedding(embeddings_to_index)

                embeddings = embeddings.tolist()

                metadata = []
                ids = []
                for doc in document_batch:
                    # TODO if vector_id unecessary then rewrite below (maybe it is needed)
                    metadata.append({key: value for key, value in doc.meta.items() if key != "vector_id"})
                    ids.append(doc.id)
                # update existing vectors in pinecone index
                self.pinecone_indexes[index].upsert(vectors=zip(ids, embeddings, metadata))

                progress_bar.set_description_str("Documents Processed")
                progress_bar.update(batch_size)

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")
        if filters:
            raise Exception("get_all_documents does not support filters.")
        self._limit_check(batch_size)

        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict] = None,
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
                        Example: {"genre": {"$in": ["documentary", "action"]}},
                        more info on filtering syntax here https://www.pinecone.io/docs/metadata-filtering/
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")
        if filters:
            raise Exception("get_all_documents_generator does not support filters.")
        self._limit_check(batch_size)

        index = index or self.index
        index = self._sanitize_index_name(index)
        documents = super(PineconeDocumentStore, self).get_all_documents_generator(
            index=index, filters=filters, batch_size=batch_size, return_embedding=False
        )
        if return_embedding is None:
            return_embedding = self.return_embedding

        for doc in documents:
            if return_embedding:
                if doc.meta and doc.meta.get("vector_id") is not None:
                    res = self.pinecone_indexes[index].fetch(ids=[doc.id])
                    if res["vectors"].get(doc.id):
                        doc.embedding = self._convert_pinecone_result_to_document(
                            result=res["vectors"][doc.id], return_embedding=return_embedding
                        ).embedding
            yield doc

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")
        self._limit_check(batch_size)

        index = index or self.index
        index = self._sanitize_index_name(index)
        # TODO could put this repetative chunk in a _method?
        if not self.pinecone_indexes.get(index):
            self.pinecone_indexes[index] = self._create_index_if_not_exist(
                vector_dim=self.vector_dim,
                index=index,
                metric_type=self.metric,
                replicas=self.replicas,
                shards=self.shards,
            )
        # check there are vectors
        count = self.get_embedding_count(index)
        if count == 0:
            raise Exception("No documents exist, try creating documents with write_embeddings first.")
        res = self.pinecone_indexes[index].fetch(ids=ids)
        # convert Pinecone responses to documents
        documents = []
        for id_val in ids:
            # check exists
            if res["vectors"].get(id_val):
                documents.append(
                    self._convert_pinecone_result_to_document(
                        result=res["vectors"][id_val], return_embedding=self.return_embedding
                    )
                )
        # get content from SQL
        content = super().get_documents_by_id([doc.id for doc in documents])
        for i, doc in enumerate(documents):
            doc.content = content[i].content
        return documents

    def get_embedding_count(self, index: Optional[str] = None, filters: Optional[Dict] = None) -> int:
        """
        Return the count of embeddings in the document store.
        """
        if filters:
            raise Exception("Filters are not supported for get_embedding_count in PineconeDocumentStore")
        index = index or self.index
        index = self._sanitize_index_name(index)
        if not self.pinecone_indexes.get(index):
            self.pinecone_indexes[index] = self._create_index_if_not_exist(
                vector_dim=self.vector_dim,
                index=self.index,
                metric_type=self.metric,
                replicas=self.replicas,
                shards=self.shards,
            )

        stats = self.pinecone_indexes[index].describe_index_stats()
        # if no namespace return zero
        count = stats["namespaces"][""]["vector_count"] if stats["namespaces"].get("") else 0
        return count

    def train_index(
        self,
        documents: Optional[Union[List[dict], List[Document]]],
        embeddings: Optional[np.ndarray] = None,
        index: Optional[str] = None,
    ):
        """
        Not applicable to PineconeDocumentStore.
        """
        raise NotImplementedError("PineconeDocumentStore does not require training")

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete documents from the document store.

        :param index: Index name to delete the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param ids: Optional list of IDs to narrow down the documents to be deleted.
        :param filters: Optional filters to narrow down the documents to be deleted (not supported by PineconeDocumentStore).
                        Example: {"genre": {"$in": ["documentary", "action"]}},
                        more info on filtering syntax here https://www.pinecone.io/docs/metadata-filtering/
        :return: None
        """
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")
        if filters:
            raise NotImplementedError("PineconeDocumentStore does not support filtering during document deletion.")

        index = index or self.index
        index = self._sanitize_index_name(index)
        if not self.pinecone_indexes.get(index):
            self.pinecone_indexes[index] = self._create_index_if_not_exist(
                vector_dim=self.vector_dim,
                index=self.index,
                metric_type=self.metric,
                replicas=self.replicas,
                shards=self.shards,
            )
        _ = self.pinecone_indexes[index].delete(ids=ids)
        # delete from SQL
        super().delete_documents(index=index, ids=ids, filters=filters)

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[Dict] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param filters: Optional filters to narrow down the search space.
                        Example: {"genre": {"$in": ["documentary", "action"]}},
                        more info on filtering syntax here https://www.pinecone.io/docs/metadata-filtering/
        :param top_k: How many documents to return
        :param index: Index name to query the document from.
        :param return_embedding: To return document embedding
        :return:
        """
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")
        self._limit_check(top_k, include_values=return_embedding)

        index = index or self.index
        index = self._sanitize_index_name(index)

        if not self.pinecone_indexes.get(index):
            raise Exception(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running 'update_embeddings()' to create and populate an index."
            )

        if return_embedding is None:
            return_embedding = self.return_embedding

        query_emb = query_emb.reshape(1, -1).astype(np.float32)

        if self.similarity == "cosine":
            self.normalize_embedding(query_emb)

        res = self.pinecone_indexes[index].query(query_emb.tolist(), top_k=top_k, include_values=True, filter=filters)

        score_matrix = []
        vector_id_matrix = []
        for match in res["results"][0]["matches"]:
            score_matrix.append(match["score"])
            vector_id_matrix.append(match["id"])

        documents = self.get_documents_by_vector_ids(vector_id_matrix, index=index)

        # assign query score to each document
        scores_for_vector_ids: Dict[str, float] = {str(v_id): s for v_id, s in zip(vector_id_matrix, score_matrix)}
        for i, doc in enumerate(documents):
            raw_score = scores_for_vector_ids[doc.id]
            doc.score = self.finalize_raw_score(raw_score, self.similarity)

            if return_embedding is True:
                # get embedding from Pinecone response
                doc.embedding = self.pinecone_indexes[index].reconstruct(int(doc.id))

        return documents

    def save(self):
        """
        Save index to the specified file, not implemented for PineconeDocumentStore.
        """
        raise NotImplementedError("save method not implemented for PineconeDocumentStore")

    def _load_init_params_from_config(
        self, index_path: Optional[Union[str, Path]] = None, config_path: Optional[Union[str, Path]] = None
    ):
        raise NotImplementedError("Load init params from config not implemented for Pinecone")

    def _limit_check(self, top_k: str, include_values: Optional[bool] = None):
        """
        Confirms the top_k value does not exceed Pinecone vector database limits.
        """
        if include_values:
            if top_k > self.top_k_limit_vectors:
                raise Exception(
                    f"PineconeDocumentStore allows requests of no more than {self.top_k_limit_vectors} records ",
                    f"when returning embedding values. This request is attempting to return {top_k} records.",
                )
        else:
            if top_k > self.top_k_limit:
                raise Exception(
                    f"PineconeDocumentStore allows requests of no more than {self.top_k_limit} records. ",
                    f"This request is attempting to return {top_k} records.",
                )

    @classmethod
    def load():
        """
        Default class method used for loading indexes. Not applicable to the PineconeDocumentStore.
        """
        raise NotImplementedError("load method not supported for PineconeDocumentStore")
