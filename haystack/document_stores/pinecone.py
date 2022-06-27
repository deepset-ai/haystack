from argparse import Namespace
from typing import TYPE_CHECKING, Union, List, Optional, Dict, Generator

import logging

import pinecone
import numpy as np
from tqdm.auto import tqdm

from haystack.schema import Document
from haystack.document_stores.base import BaseDocumentStore
#from haystack.document_stores.base import get_batches_from_generator
from haystack.document_stores.filter_utils import LogicalFilterClause
from haystack.errors import DocumentStoreError

if TYPE_CHECKING:
    from haystack.nodes.retriever import BaseRetriever


logger = logging.getLogger(__name__)


class PineconeDocumentStore(BaseDocumentStore):
    """
    Document store for very large scale embedding based dense retrievers like the DPR. This is a hosted document store,
    this means that your vectors will not be stored locally but in the cloud. This means that the similarity
    search will be run on the cloud as well.

    It implements the Pinecone vector database ([https://www.pinecone.io](https://www.pinecone.io))
    to perform similarity search on vectors. In order to use this document store, you need an API key that you can
    obtain by creating an account on the [Pinecone website](https://www.pinecone.io).

    The document text is stored using the SQLDocumentStore, while
    the vector embeddings and metadata (for filtering) are indexed in a Pinecone Index.
    """

    top_k_limit = 10_000
    top_k_limit_vectors = 1_000

    def __init__(
        self,
        api_key: str,
        environment: str = "us-west1-gcp",
        #sql_url: str = "sqlite:///pinecone_document_store.db",
        pinecone_index: Optional[pinecone.Index] = None,
        embedding_dim: int = 768,
        return_embedding: bool = False,
        index: str = "document",
        similarity: str = "cosine",
        replicas: int = 1,
        shards: int = 1,
        embedding_field: str = "embedding",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        recreate_index: bool = False,
        metadata_config: dict = {"indexed": []},
    ):
        """
        :param api_key: Pinecone vector database API key ([https://app.pinecone.io](https://app.pinecone.io)).
        :param environment: Pinecone cloud environment uses `"us-west1-gcp"` by default. Other GCP and AWS regions are
            supported, contact Pinecone [here](https://www.pinecone.io/contact/) if required.
        :param sql_url: SQL connection URL for database. It defaults to local file based SQLite DB. For large scale
            deployment, Postgres is recommended.
        :param pinecone_index: pinecone-client Index object, an index will be initialized or loaded if not specified.
        :param embedding_dim: The embedding vector size.
        :param return_embedding: Whether to return document embeddings.
        :param index: Name of index in document store to use.
        :param similarity: The similarity function used to compare document vectors. `"cosine"` is the default
            and is recommended if you are using a Sentence-Transformer model. `"dot_product"` is more performant
            with DPR embeddings.
            In both cases, the returned values in Document.score are normalized to be in range [0,1]:
                - For `"dot_product"`: `expit(np.asarray(raw_score / 100))`
                - For `"cosine"`: `(raw_score + 1) / 2`
        :param replicas: The number of replicas. Replicas duplicate the index. They provide higher availability and
            throughput.
        :param shards: The number of shards to be used in the index. We recommend to use 1 shard per 1GB of data.
        :param embedding_field: Name of field containing an embedding vector.
        :param progress_bar: Whether to show a tqdm progress bar or not.
            Can be helpful to disable in production deployments to keep the logs clean.
        :param duplicate_documents: Handle duplicate documents based on parameter options.\

            Parameter options:
                - `"skip"`: Ignore the duplicate documents.
                - `"overwrite"`: Update any existing documents with the same ID when adding documents.
                - `"fail"`: An error is raised if the document ID of the document being added already exists.
        :param recreate_index: If set to True, an existing Pinecone index will be deleted and a new one will be
            created using the config you are using for initialization. Be aware that all data in the old index will be
            lost if you choose to recreate the index. Be aware that both the document_index and the label_index will
            be recreated.
        :param metadata_config: Which metadata fields should be indexed. Should be in the format
            `{"indexed": ["metadata-field-1", "metadata-field-2", "metadata-field-n"]}`.
        """
        # Connect to Pinecone server using python client binding
        pinecone.init(api_key=api_key, environment=environment)
        self._api_key = api_key

        # Formal similarity string
        if similarity == "cosine":
            self.metric_type = similarity
        elif similarity == "dot_product":
            self.metric_type = "dotproduct"
        elif similarity in ("l2", "euclidean"):
            self.metric_type = "euclidean"
        else:
            raise ValueError(
                "The Pinecone document store can currently only support dot_product, cosine and euclidean metrics. "
                "Please set similarity to one of the above."
            )

        self.similarity = similarity
        self.index = index
        self.embedding_dim = embedding_dim
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents

        # Pinecone index params
        self.replicas = replicas
        self.shards = shards
        self.metadata_config = metadata_config

        # Initialize dictionary of index connections
        self.pinecone_indexes: Dict[str, pinecone.Index] = {}
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field

        # Initialize temporary set of document IDs
        self.all_ids = set()

        self.progress_bar = progress_bar

        clean_index = self._sanitize_index_name(index)
        #TODO REMOVE super().__init__(url=sql_url, index=clean_index, duplicate_documents=duplicate_documents)

        if pinecone_index:
            self.pinecone_indexes[clean_index] = pinecone_index
        else:
            self.pinecone_indexes[clean_index] = self._create_index(
                embedding_dim=self.embedding_dim,
                index=clean_index,
                metric_type=self.metric_type,
                replicas=self.replicas,
                shards=self.shards,
                recreate_index=recreate_index,
                metadata_config=self.metadata_config,
            )

    def _sanitize_index_name(self, index: str) -> str:
        return index.replace("_", "-").lower()

    def _create_index(
        self,
        embedding_dim: int,
        index: Optional[str] = None,
        metric_type: Optional[str] = "cosine",
        replicas: Optional[int] = 1,
        shards: Optional[int] = 1,
        recreate_index: bool = False,
        metadata_config: dict = {"indexed": []},
    ):
        """
        Create a new index for storing documents in case an
        index with the name doesn't exist already.
        """
        index = index or self.index
        index = self._sanitize_index_name(index)

        if recreate_index:
            self.delete_index(index)
            # TODO REMOVE super().delete_labels()

        # Skip if already exists
        if index in self.pinecone_indexes.keys():
            index_connection = self.pinecone_indexes[index]
        else:
            # Search pinecone hosted indexes and create an index if it does not exist
            if index not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index,
                    dimension=embedding_dim,
                    metric=metric_type,
                    replicas=replicas,
                    shards=shards,
                    metadata_config=metadata_config,
                )
            index_connection = pinecone.Index(index)

        # Get index statistics
        stats = index_connection.describe_index_stats()
        dims = stats["dimension"]
        count = stats["namespaces"][""]["vector_count"] if stats["namespaces"].get("") else 0
        logger.info(f"Index statistics: name: {index}, embedding dimensions: {dims}, record count: {count}")
        # return index connection
        return index_connection

    def get_document_count(
        self, index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None
    ) -> int:
        """
        Return the count of embeddings in the document store.
        """
        if filters:
            raise NotImplementedError("Filters are not supported for get_embedding_count in PineconeDocumentStore")

        index = index or self.index
        index = self._sanitize_index_name(index)
        if not self.pinecone_indexes.get(index, False):
            raise ValueError(f"No index named {index} found in Pinecone.")

        stats = self.pinecone_indexes[index].describe_index_stats()
        # Document count is total number of vectors across all namespaces (no-vectors + vectors)
        count = 0
        for namespace in stats["namespaces"].keys():
            count += stats["namespaces"][namespace]["vector_count"]
        return count

    def _validate_index_sync(self):
        """
        This check ensures the correct document database was loaded. If it fails, make sure you provided the same path
        to the SQL database as when you created the original Pinecone index.
        """
        if not self.get_document_count() == self.get_embedding_count():
            # TODO update error message below
            raise DocumentStoreError(
                "The number of documents present in the SQL database does not "
                "match the number of embeddings in Pinecone. Make sure your Pinecone "
                "index aligns to the same database that was used when creating the "
                "original index."
            )

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 32,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Add new documents to the DocumentStore.

        :param documents: List of `Dicts` or list of `Documents`. If they already contain embeddings, we'll index them
            right away in Pinecone. If not, you can later call `update_embeddings()` to create & index them.
        :param index: Index name for storing the docs and metadata.
        :param batch_size: Number of documents to process at a time. When working with large number of documents,
            batching can help to reduce the memory footprint.
        :param duplicate_documents: handle duplicate documents based on parameter options.

            Parameter options:
                - `"skip"`: Ignore the duplicate documents.
                - `"overwrite"`: Update any existing documents with the same ID when adding documents.
                - `"fail"`: An error is raised if the document ID of the document being added already exists.
        :param headers: PineconeDocumentStore does not support headers.
        :raises DuplicateDocumentError: Exception trigger on duplicate document.
        """
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        index = index or self.index
        index = self._sanitize_index_name(index)
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert (
            duplicate_documents in self.duplicate_documents_options
        ), f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        if index not in self.pinecone_indexes:
            self.pinecone_indexes[index] = self._create_index(
                embedding_dim=self.embedding_dim,
                index=index,
                metric_type=self.metric_type,
                replicas=self.replicas,
                shards=self.shards,
                recreate_index=False,
            )

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(
            documents=document_objects, index=index, duplicate_documents=duplicate_documents
        )
        if len(document_objects) > 0:
            add_vectors = False if document_objects[0].embedding is None else True
            # If not adding vectors we use no-vectors namespace
            namespace = "vectors" if add_vectors else "no-vectors"
            # To store info in Pinecone, we use zero embeddings (to be replaced with real embeddings later)
            if not add_vectors:
                embeddings = np.zeros((batch_size, self.embedding_dim), dtype="float32")
                # Convert embeddings to list objects
                embeddings = [embed.tolist() if embed is not None else None for embed in embeddings]
            with tqdm(
                total=len(document_objects), disable=not self.progress_bar, position=0, desc="Writing Documents"
            ) as progress_bar:
                for i in range(0, len(document_objects), batch_size):
                    ids = [doc.id for doc in document_objects[i : i + batch_size]]
                    metadata = [
                        {**doc.meta, **{"content": doc.content}} for doc in document_objects[i : i + batch_size]
                    ]
                    if add_vectors:
                        embeddings = [doc.embedding for doc in document_objects[i : i + batch_size]]
                        embeddings = np.array(embeddings, dtype="float32")
                        if self.similarity == "cosine":
                            # Normalize embeddings inplace
                            self.normalize_embedding(embeddings)
                        # Convert embeddings to list objects
                        embeddings = [embed.tolist() if embed is not None else None for embed in embeddings]
                    data_to_write_to_pinecone = zip(ids, embeddings, metadata)
                    # Metadata fields and embeddings are stored in Pinecone
                    self.pinecone_indexes[index].upsert(vectors=data_to_write_to_pinecone, namespace=namespace)
                    # TODO remove the below
                    #docs_to_write_to_sql = document_objects[i : i + batch_size]
                    #super(PineconeDocumentStore, self).write_documents(
                    #    docs_to_write_to_sql, index=index, duplicate_documents=duplicate_documents
                    #)
                    # Add IDs to ID list
                    self.all_ids = self.all_ids.union(set(ids))
                    progress_bar.update(batch_size)
            progress_bar.close()

    def _create_document_field_map(self) -> Dict:
        return {self.embedding_field: "embedding"}

    def update_embeddings(
        self,
        retriever: "BaseRetriever",
        index: Optional[str] = None,
        update_existing_embeddings: bool = True,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        batch_size: int = 32,
    ):
        """
        Updates the embeddings in the document store using the encoding model specified in the retriever.
        This can be useful if you want to add or change the embeddings for your documents (e.g. after changing the
        retriever config).

        :param retriever: Retriever to use to get embeddings for text.
        :param index: Index name for which embeddings are to be updated. If set to `None`, the default `self.index` is
            used.
        :param update_existing_embeddings: Whether to update existing embeddings of the documents. If set to `False`,
            only documents without embeddings are processed. This mode can be used for incremental updating of
            embeddings, wherein, only newly indexed documents get processed.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
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
                ```
        :param batch_size: Number of documents to process at a time. When working with large number of documents,
            batching can help reduce memory footprint.
        """
        index = index or self.index
        index = self._sanitize_index_name(index)

        if index not in self.pinecone_indexes:
            raise ValueError(
                f"Couldn't find a the index '{index}' in Pinecone. Try to init the "
                f"PineconeDocumentStore() again ..."
            )

        document_count = self.get_document_count(index=index, filters=filters)
        if document_count == 0:
            logger.warning("Calling DocumentStore.update_embeddings() on an empty index")
            return

        logger.info(f"Updating embeddings for {document_count} docs...")

        # TODO: make below dynamic to deal with namespace "vector" -> "vector updates"
        batched_documents = self.get_all_documents_generator(
            index=index, namespace="no-vectors", filters=filters, return_embedding=False, batch_size=batch_size
        )

        with tqdm(
            total=document_count, disable=not self.progress_bar, position=0, unit=" docs", desc="Updating Embedding"
        ) as progress_bar:
            for document_batch in batched_documents:
                embeddings = retriever.embed_documents(document_batch)  # type: ignore
                assert len(document_batch) == len(embeddings)

                embeddings_to_index = np.array(embeddings, dtype="float32")
                if self.similarity == "cosine":
                    self.normalize_embedding(embeddings_to_index)
                embeddings = embeddings_to_index.tolist()

                metadata = []
                ids = []
                for doc in document_batch:
                    metadata.append({**doc.meta, **{"content": doc.content}})
                    ids.append(doc.id)
                # update existing vectors in pinecone index
                self.pinecone_indexes[index].upsert(vectors=zip(ids, embeddings, metadata), namespace="vectors")
                # delete existing vectors from "no-vectors" namespace if they exist there
                self.delete_documents(index=index, ids=ids, namespace="no-vectors")

                progress_bar.set_description_str("Documents Processed")
                progress_bar.update(batch_size)

    def get_all_documents(
        self,
        index: Optional[str] = None,
        namespace: Optional[str] = "vectors",
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 32,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:

        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        result = self.get_all_documents_generator(
            index=index, namespace=namespace, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        namespace: Optional[str] = "vectors",
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 32,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        Get all documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
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
                ```
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param headers: PineconeDocumentStore does not support headers.
        """
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")
        if return_embedding is None:
            return_embedding = self.return_embedding

        index = index or self.index
        index = self._sanitize_index_name(index)

        ids = self._get_all_document_ids(
            namespace=namespace,
            filters=filters,
            batch_size=batch_size
        )
        for i in range(0, len(ids), batch_size):
            i_end = min(len(ids), i+batch_size)
            documents = self.get_documents_by_id(
                ids=ids[i:i_end],
                namespace=namespace,
                batch_size=batch_size
            )
            for doc in documents:
                if return_embedding:
                    self._attach_embedding_to_document(document=doc, index=index, namespace=namespace)
            yield documents
    
    def _get_all_document_ids(
        self,
        index: Optional[str] = None,
        namespace: Optional[str] = "vectors",
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        batch_size: int = 32,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        index = index or self.index
        index = self._sanitize_index_name(index)
        if index not in self.pinecone_indexes:
            raise DocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        document_count = self.get_document_count()
        if len(self.all_ids) == document_count:
            # We have all of the IDs and don't need to extract from Pinecone
            return list(self.all_ids)
        else:
            dummy_query = [0.0] * self.embedding_dim
            target_namespace = f"{namespace}-copy"
            with tqdm(
                total=document_count, disable=not self.progress_bar, position=0, unit=" ids", desc="Retrieving IDs"
            ) as progress_bar:
                # Retrieve embeddings from Pinecone
                res = self.pinecone_indexes[index].query(
                    dummy_query,
                    namespace="vectors",
                    top_k=batch_size,
                    include_values=False,
                    include_metadata=False,
                    filter=filters,
                )
                vector_id_matrix = []
                for match in res["matches"]:
                    vector_id_matrix.append(match["id"])
                # Save IDs
                self.all_ids = self.all_ids.union(set(vector_id_matrix))
                # Move these IDs to new namespace
                self._move_documents_by_id_namespace(
                    ids=vector_id_matrix,
                    source_namespace=namespace,
                    target_namespace=target_namespace,
                    batch_size=batch_size
                )
                progress_bar.set_description_str("Retrieved IDs")
                progress_bar.update(batch_size)
            # Now move all documents back to source namespace
            self._move_documents_by_id_namespace(
                ids=list(self.all_ids),
                source_namespace=target_namespace,
                target_namespace=namespace,
                batch_size=batch_size
            )
            return list(self.all_ids)
    
    def _move_documents_by_id_namespace(
        self,
        ids: List[str],
        index: Optional[str] = None,
        source_namespace: Optional[str] = "vectors",
        target_namespace: Optional[str] = "copy",
        batch_size: int = 32,
    ):
        with tqdm(
            total=len(ids), disable=not self.progress_bar, position=0, unit=" docs", desc="Moving Embeddings"
        ) as progress_bar:
            for i in range(0, len(ids), batch_size):
                i_end = min(len(ids), i+batch_size)
                document_batch = self.get_documents_by_id(
                    ids=ids[i:i_end],
                    namespace=source_namespace,
                    index=index,
                    return_embedding=True
                )
                metadata = [
                    {**doc.meta, **{"content": doc.content}} for doc in document_batch
                ]
                embeddings = [doc.embedding for doc in document_batch]
                data_to_write_to_pinecone = zip(ids, embeddings, metadata)
                # Metadata fields and embeddings are stored in Pinecone
                self.pinecone_indexes[index].upsert(
                    vectors=data_to_write_to_pinecone,
                    namespace=target_namespace
                )
                # Delete vectors from source_namespace
                self.delete_documents(index=index, ids=ids, namespace=source_namespace)

                progress_bar.set_description_str("Documents Moved")
                progress_bar.update(batch_size)

    def get_documents_by_id(
        self,
        ids: List[str],
        namespace: str = "vectors",
        index: Optional[str] = None,
        batch_size: int = 32,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:

        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        if return_embedding is None:
            return_embedding = self.return_embedding

        index = index or self.index
        index = self._sanitize_index_name(index)
        result = self.pinecone_indexes[index].fetch(ids=ids, namespace=namespace)
        vector_id_matrix = []
        meta_matrix = []
        for _id in result["vectors"].keys():
            vector_id_matrix.append(_id)
            meta_matrix.append(result["vectors"][_id]["metadata"])
        documents = self._get_documents_by_meta(
            vector_id_matrix, meta_matrix, index=index, return_embedding=return_embedding
        )
        # TODO remove duplication of multiple getting embeddings above and below
        if return_embedding:
            for doc in documents:
                self._attach_embedding_to_document(document=doc, index=index, namespace=namespace)

        return documents
    
    def get_document_by_id(self,
        id: str,
        namespace: str = "vectors",
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None,
    ) -> Document:
        """
        Returns a single Document retrieved using an ID.
        """
        documents = self.get_documents_by_id(
            ids=[id],
            namespace=namespace,
            index=index,
            headers=headers,
            return_embedding=return_embedding
        )
        return documents[0]

    def get_embedding_count(
        self, index: Optional[str] = None, filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None
    ) -> int:
        """
        Return the count of embeddings in the document store.
        """
        if filters:
            raise NotImplementedError("Filters are not supported for get_embedding_count in PineconeDocumentStore")

        index = index or self.index
        index = self._sanitize_index_name(index)
        if not self.pinecone_indexes.get(index, False):
            raise ValueError(f"No index named {index} found in Pinecone.")

        stats = self.pinecone_indexes[index].describe_index_stats()
        # if no "vectors" namespace return zero
        count = stats["namespaces"]["vectors"]["vector_count"] if "vectors" in stats["namespaces"] else 0
        return count

    def update_document_meta(self, id: str, meta: Dict[str, str], namespace: str = "vectors", index: str = None):
        """
        Update the metadata dictionary of a document by specifying its string id
        """
        index = index or self.index
        index = self._sanitize_index_name(index)
        if index in self.pinecone_indexes:
            doc = self.get_documents_by_id(ids=[id], index=index, return_embedding=True)[0]
            if doc.embedding is not None:
                meta = {**meta, **{"content": doc.content}}
                self.pinecone_indexes[index].upsert(
                    vectors=([id], [doc.embedding.tolist()], [meta]), namespace=namespace
                )
        # TODO confirm this function works without below
        # super().update_document_meta(id=id, meta=meta, index=index)

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = "vectors",
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete documents from the document store.

        :param index: Index name to delete the documents from. If `None`, the DocumentStore's default index
            (`self.index`) will be used.
        :param ids: Optional list of IDs to narrow down the documents to be deleted.
        :param namespace: Optional namespace str, by default should be "no-vectors" or "vectors"
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
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
                ```
        :param headers: PineconeDocumentStore does not support headers.
        """
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        index = index or self.index
        index = self._sanitize_index_name(index)
        if index in self.pinecone_indexes:
            if ids is None and filters is None:
                # TODO is deleting all a good idea?
                self.pinecone_indexes[index].delete(delete_all=True, namespace=namespace)
            else:
                """TODO is affected docs allowing us to delete with filters applied?
                affected_docs = self.get_all_documents(
                    filters=filters,
                    namespace=namespace,
                    return_embedding=False
                )
                if ids:
                    affected_docs = [doc for doc in affected_docs if doc.id in ids]

                doc_ids = [doc.id for doc in affected_docs]
                """
                # TODO now the deletion is going ahead without filter
                self.pinecone_indexes[index].delete(ids=ids, namespace=namespace)

        #TODO remove below
        # #super().delete_documents(index=index, ids=ids, filters=filters)

    def delete_index(self, index: str):
        """
        Delete an existing index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        index = self._sanitize_index_name(index)
        if index in pinecone.list_indexes():
            pinecone.delete_index(index)
            logger.info(f"Index '{index}' deleted.")
        if index in self.pinecone_indexes:
            del self.pinecone_indexes[index]
        # TODO REMOVE super().delete_index(index)

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[Dict[str, Union[Dict, List, str, int, float, bool]]] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
    ) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query (e.g. gathered from DPR).
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
        :param top_k: How many documents to return.
        :param index: The name of the index from which to retrieve documents.
        :param return_embedding: Whether to return document embedding.
        :param headers: PineconeDocumentStore does not support headers.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        if return_embedding is None:
            return_embedding = self.return_embedding
        self._limit_check(top_k, include_values=return_embedding)

        if filters:
            filters = LogicalFilterClause.parse(filters).convert_to_pinecone()

        index = index or self.index
        index = self._sanitize_index_name(index)

        if index not in self.pinecone_indexes:
            raise DocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )
        query_emb = query_emb.astype(np.float32)

        if self.similarity == "cosine":
            self.normalize_embedding(query_emb)

        res = self.pinecone_indexes[index].query(
            query_emb.tolist(),
            namespace="vectors",
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            filter=filters,
        )

        score_matrix = []
        vector_id_matrix = []
        meta_matrix = []
        for match in res["matches"]:
            score_matrix.append(match["score"])
            vector_id_matrix.append(match["id"])
            meta_matrix.append(match["metadata"])
        documents = self._get_documents_by_meta(
            vector_id_matrix, meta_matrix, index=index, return_embedding=return_embedding
        )

        # assign query score to each document
        scores_for_vector_ids: Dict[str, float] = {str(v_id): s for v_id, s in zip(vector_id_matrix, score_matrix)}
        for doc in documents:
            score = scores_for_vector_ids[doc.id]
            if scale_score:
                score = self.scale_to_unit_interval(score, self.similarity)
            doc.score = score

        return documents

    def _get_documents_by_meta(
        self,
        ids: List[str],
        metadata: List[dict],
        namespace: Optional[str] = "vectors",
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:

        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        if return_embedding is None:
            return_embedding = self.return_embedding

        index = index or self.index
        index = self._sanitize_index_name(index)

        # extract ID, content, and metadata to create Documents
        documents = []
        for _id, meta in zip(ids, metadata):
            content = meta["content"]
            del meta["content"]
            doc = Document(id=_id, content=content, meta=meta)
            documents.append(doc)
        if return_embedding:
            for doc in documents:
                self._attach_embedding_to_document(document=doc, index=index, namespace=namespace)

        return documents

    def _attach_embedding_to_document(self, document: Document, index: str, namespace: str):
        """
        Fetches the Document's embedding from the specified Pinecone index and attaches it to the Document's
        embedding field.
        """
        result = self.pinecone_indexes[index].fetch(ids=[document.id], namespace=namespace)
        if result["vectors"].get(document.id, False):
            embedding = result["vectors"][document.id].get("values", None)
            document.embedding = np.asarray(embedding, dtype=np.float32)

    def _limit_check(self, top_k: int, include_values: Optional[bool] = None):
        """
        Confirms the top_k value does not exceed Pinecone vector database limits.
        """
        if include_values:
            if top_k > self.top_k_limit_vectors:
                raise DocumentStoreError(
                    f"PineconeDocumentStore allows requests of no more than {self.top_k_limit_vectors} records "
                    f"when returning embedding values. This request is attempting to return {top_k} records."
                )
        else:
            if top_k > self.top_k_limit:
                raise DocumentStoreError(
                    f"PineconeDocumentStore allows requests of no more than {self.top_k_limit} records. "
                    f"This request is attempting to return {top_k} records."
                )

    @classmethod
    def load(cls):
        """
        Default class method used for loading indexes. Not applicable to the PineconeDocumentStore.
        """
        raise NotImplementedError("load method not supported for PineconeDocumentStore")

    def delete_labels(self):
        """
        Default class method used for deleting labels. Not support by the PineconeDocumentStore
        """
        raise NotImplementedError("Labels are not support by the PineconeDocumentStore")

    def get_all_labels(self):
        """
        Default class method used for getting all labels. Not support by the PineconeDocumentStore
        """
        raise NotImplementedError("Labels are not support by the PineconeDocumentStore")

    def get_label_count(self):
        """
        Default class method used for counting labels. Not support by the PineconeDocumentStore
        """
        raise NotImplementedError("Labels are not support by the PineconeDocumentStore")

    def write_labels(self):
        """
        Default class method used for writing labels. Not support by the PineconeDocumentStore
        """
        raise NotImplementedError("Labels are not support by the PineconeDocumentStore")
