from typing import Set, Union, List, Optional, Dict, Generator, Any

import logging
from itertools import islice
from functools import reduce
import operator

import pinecone
import numpy as np
from tqdm.auto import tqdm

from haystack.schema import Document, FilterType, Label, Answer, Span
from haystack.document_stores import BaseDocumentStore

from haystack.document_stores.filter_utils import LogicalFilterClause
from haystack.errors import PineconeDocumentStoreError, DuplicateDocumentError
from haystack.nodes.retriever import DenseRetriever


logger = logging.getLogger(__name__)


def _sanitize_index_name(index: Optional[str]) -> Optional[str]:
    if index:
        return index.replace("_", "-").lower()
    return None


def _get_by_path(root, items):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, items, root)


def _set_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    _get_by_path(root, items[:-1])[items[-1]] = value


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
        metadata_config: Optional[Dict] = None,
        validate_index_sync: bool = True,
    ):
        """
        :param api_key: Pinecone vector database API key ([https://app.pinecone.io](https://app.pinecone.io)).
        :param environment: Pinecone cloud environment uses `"us-west1-gcp"` by default. Other GCP and AWS regions are
            supported, contact Pinecone [here](https://www.pinecone.io/contact/) if required.
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
        :param metadata_config: Which metadata fields should be indexed, part of the
            [selective metadata filtering](https://www.pinecone.io/docs/manage-indexes/#selective-metadata-indexing) feature.
            Should be in the format `{"indexed": ["metadata-field-1", "metadata-field-2", "metadata-field-n"]}`. By default,
            no fields are indexed.
        """
        if metadata_config is None:
            metadata_config = {"indexed": []}
        # Connect to Pinecone server using python client binding
        if not api_key:
            raise PineconeDocumentStoreError(
                "Pinecone requires an API key, please provide one. https://app.pinecone.io"
            )

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
        self.index: str = self._index_name(index)
        self.embedding_dim = embedding_dim
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents
        self.document_namespace = "no-vectors"
        self.embedding_namespace = "vectors"

        # Pinecone index params
        self.replicas = replicas
        self.shards = shards

        # Add necessary metadata fields to metadata_config
        fields = ["label-id", "query"]
        metadata_config["indexed"] += fields
        self.metadata_config = metadata_config

        # Initialize dictionary of index connections
        self.pinecone_indexes: Dict[str, pinecone.Index] = {}
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field

        # Initialize dictionary to store temporary set of document IDs
        self.all_ids: dict = {}
        # Dummy query to be used during searches
        self.dummy_query = [0.0] * self.embedding_dim

        self.progress_bar = progress_bar

        if pinecone_index:
            if not isinstance(pinecone_index, pinecone.Index):
                raise PineconeDocumentStoreError(
                    f"The parameter `pinecone_index` needs to be a "
                    f"`pinecone.Index` object. You provided an object of "
                    f"type `{type(pinecone_index)}`."
                )
            self.pinecone_indexes[self.index] = pinecone_index
        else:
            self.pinecone_indexes[self.index] = self._create_index(
                embedding_dim=self.embedding_dim,
                index=self.index,
                metric_type=self.metric_type,
                replicas=self.replicas,
                shards=self.shards,
                recreate_index=recreate_index,
                metadata_config=self.metadata_config,
            )

        super().__init__()

    def _add_local_ids(self, index: str, ids: list):
        """
        Add all document IDs to the set of all IDs.
        """
        if index not in self.all_ids:
            self.all_ids[index] = set()
        self.all_ids[index] = self.all_ids[index].union(set(ids))

    def _index_name(self, index) -> str:
        index = _sanitize_index_name(index) or self.index
        # self.index = index  # TODO maybe not needed
        return index

    def _create_index(
        self,
        embedding_dim: int,
        index: Optional[str] = None,
        metric_type: Optional[str] = "cosine",
        replicas: Optional[int] = 1,
        shards: Optional[int] = 1,
        recreate_index: bool = False,
        metadata_config: Optional[Dict] = None,
    ):
        """
        Create a new index for storing documents in case an
        index with the name doesn't exist already.
        """
        if metadata_config is None:
            metadata_config = {"indexed": []}
        index = self._index_name(index)

        if recreate_index:
            self.delete_index(index)

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
        logger.info("Index statistics: name: %s embedding dimensions: %s, record count: %s", index, dims, count)
        # return index connection
        return index_connection

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Return the count of embeddings in the document store.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
            Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
            operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
            `"$gte"`, `"$lt"`, `"$lte"`), or a metadata field name.
            Logical operator keys take a dictionary of metadata field names or logical operators as
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
        :param index: Optional index to use for the query. If not provided, the default index is used.
        :param only_documents_without_embedding: If set to `True`, only documents without embeddings are counted.
        :param headers: PineconeDocumentStore does not support headers.
        """
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        index = self._index_name(index)
        if index not in self.pinecone_indexes:
            raise PineconeDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        pinecone_syntax_filter = LogicalFilterClause.parse(filters).convert_to_pinecone() if filters else None

        stats = self.pinecone_indexes[index].describe_index_stats(filter=pinecone_syntax_filter)
        # Document count is total number of vectors across all namespaces (no-vectors + vectors)
        count = 0
        for namespace in stats["namespaces"].keys():
            if not (only_documents_without_embedding and "no-vectors" not in namespace):
                count += stats["namespaces"][namespace]["vector_count"]
        return count

    def _validate_index_sync(self, index: Optional[str] = None):
        """
        This check ensures the correct number of documents and embeddings are found in the
        Pinecone database.
        """
        if self.get_document_count(index=index) != self.get_embedding_count(index=index):
            raise PineconeDocumentStoreError(
                f"The number of documents present in Pinecone ({self.get_document_count(index=index)}) "
                "does not match the number of embeddings in Pinecone "
                f" ({self.get_embedding_count(index=index)}). This can happen if a document store "
                "instance is deleted during write operations. Call "
                "the `update_documents` method to fix it."
            )

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 32,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        labels: Optional[bool] = False,
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
        :param labels: Tells us whether these records are labels or not. Defaults to False.
        :raises DuplicateDocumentError: Exception trigger on duplicate document.
        """
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        index = self._index_name(index)
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
                metadata_config=self.metadata_config,
            )

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(
            documents=document_objects, index=index, duplicate_documents=duplicate_documents
        )
        if len(document_objects) > 0:
            add_vectors = False if document_objects[0].embedding is None else True
            # If these are not labels, we need to find the correct namespace
            if not labels:
                # If not adding vectors we use document namespace
                namespace = self.embedding_namespace if add_vectors else self.document_namespace
            else:
                namespace = "labels"
            if not add_vectors:
                # To store documents in Pinecone, we use dummy embeddings (to be replaced with real embeddings later)
                embeddings_to_index = np.zeros((batch_size, self.embedding_dim), dtype="float32")
                # Convert embeddings to list objects
                embeddings = [embed.tolist() if embed is not None else None for embed in embeddings_to_index]
            with tqdm(
                total=len(document_objects), disable=not self.progress_bar, position=0, desc="Writing Documents"
            ) as progress_bar:
                for i in range(0, len(document_objects), batch_size):
                    document_batch = document_objects[i : i + batch_size]
                    ids = [doc.id for doc in document_batch]
                    # If duplicate_documents set to skip or fail, we need to check for existing documents
                    if duplicate_documents in ["skip", "fail"]:
                        existing_documents = self.get_documents_by_id(ids=ids, index=index, namespace=namespace)
                        # First check for documents in current batch that exist in the index
                        if len(existing_documents) > 0:
                            if duplicate_documents == "skip":
                                # If we should skip existing documents, we drop the ids that already exist
                                skip_ids = [doc.id for doc in existing_documents]
                                # We need to drop the affected document objects from the batch
                                document_batch = [doc for doc in document_batch if doc.id not in skip_ids]
                                # Now rebuild the ID list
                                ids = [doc.id for doc in document_batch]
                                progress_bar.update(len(skip_ids))
                            elif duplicate_documents == "fail":
                                # Otherwise, we raise an error
                                raise DuplicateDocumentError(
                                    f"Document ID {existing_documents[0].id} already exists in index {index}"
                                )
                        # Now check for duplicate documents within the batch itself
                        if len(ids) != len(set(ids)):
                            if duplicate_documents in "skip":
                                # We just keep the first instance of each duplicate document
                                ids = []
                                temp_document_batch = []
                                for doc in document_batch:
                                    if doc.id not in ids:
                                        ids.append(doc.id)
                                        temp_document_batch.append(doc)
                                document_batch = temp_document_batch
                            elif duplicate_documents == "fail":
                                # Otherwise, we raise an error
                                raise DuplicateDocumentError(f"Duplicate document IDs found in batch: {ids}")
                    metadata = [
                        self._meta_for_pinecone({"content": doc.content, "content_type": doc.content_type, **doc.meta})
                        for doc in document_objects[i : i + batch_size]
                    ]
                    if add_vectors:
                        embeddings = [doc.embedding for doc in document_objects[i : i + batch_size]]
                        embeddings_to_index = np.array(embeddings, dtype="float32")
                        if self.similarity == "cosine":
                            # Normalize embeddings inplace
                            self.normalize_embedding(embeddings_to_index)
                        # Convert embeddings to list objects
                        embeddings = [embed.tolist() if embed is not None else None for embed in embeddings_to_index]
                    data_to_write_to_pinecone = zip(ids, embeddings, metadata)
                    # Metadata fields and embeddings are stored in Pinecone
                    self.pinecone_indexes[index].upsert(vectors=data_to_write_to_pinecone, namespace=namespace)
                    # Add IDs to ID list
                    self._add_local_ids(index, ids)
                    progress_bar.update(batch_size)
            progress_bar.close()

    def _create_document_field_map(self) -> Dict:
        return {self.embedding_field: "embedding"}

    def update_embeddings(
        self,
        retriever: DenseRetriever,
        index: Optional[str] = None,
        update_existing_embeddings: bool = True,
        filters: Optional[FilterType] = None,
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
        index = self._index_name(index)
        if index not in self.pinecone_indexes:
            raise ValueError(
                f"Couldn't find a the index '{index}' in Pinecone. Try to init the "
                f"PineconeDocumentStore() again ..."
            )
        document_count = self.get_document_count(index=index, filters=filters)
        if document_count == 0:
            logger.warning("Calling DocumentStore.update_embeddings() on an empty index")
            return

        logger.info("Updating embeddings for %s docs...", document_count)

        # If the embedding namespace is empty or the user does not want to update existing embeddings, we use document namespace
        if self.get_embedding_count(index=index) == 0 or not update_existing_embeddings:
            namespace = self.document_namespace
        else:
            # Else, we use the embedding namespace as this is the primary namespace for embeddings
            namespace = self.embedding_namespace

        documents = self.get_all_documents_generator(
            index=index, namespace=namespace, filters=filters, return_embedding=False, batch_size=batch_size
        )

        with tqdm(
            total=document_count, disable=not self.progress_bar, position=0, unit=" docs", desc="Updating Embedding"
        ) as progress_bar:
            for _ in range(0, document_count, batch_size):
                document_batch = list(islice(documents, batch_size))
                embeddings = retriever.embed_documents(document_batch)
                if embeddings.size == 0:
                    # Skip batch if there are no embeddings. Otherwise, incorrect embedding shape will be inferred and
                    # Pinecone APi will return a "No vectors provided" Bad Request Error
                    progress_bar.set_description_str("Documents Processed")
                    progress_bar.update(batch_size)
                    continue
                self._validate_embeddings_shape(
                    embeddings=embeddings, num_documents=len(document_batch), embedding_dim=self.embedding_dim
                )

                if self.similarity == "cosine":
                    self.normalize_embedding(embeddings)

                metadata = []
                ids = []
                for doc in document_batch:
                    metadata.append(
                        self._meta_for_pinecone({"content": doc.content, "content_type": doc.content_type, **doc.meta})
                    )
                    ids.append(doc.id)
                # Update existing vectors in pinecone index
                self.pinecone_indexes[index].upsert(
                    vectors=zip(ids, embeddings.tolist(), metadata), namespace=self.embedding_namespace
                )
                # Delete existing vectors from document namespace if they exist there
                self.delete_documents(index=index, ids=ids, namespace=self.document_namespace)
                # Add these vector IDs to local store
                self._add_local_ids(index, ids)
                progress_bar.set_description_str("Documents Processed")
                progress_bar.update(batch_size)

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 32,
        headers: Optional[Dict[str, str]] = None,
        namespace: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieves all documents in the index.

        :param index: Optional index name to retrieve all documents from.
        :param filters: Optional filters to narrow down the documents that will be retrieved.
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
        :param return_embedding: Optional flag to return the embedding of the document.
        :param batch_size: Number of documents to process at a time. When working with large number of documents,
            batching can help reduce memory footprint.
        :param headers: Pinecone does not support headers.
        :param namespace: Optional namespace to retrieve documents from.
        """
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        if namespace is None:
            if self.get_embedding_count(index=index) > 0:
                namespace = self.embedding_namespace
            else:
                namespace = self.document_namespace

        result = self.get_all_documents_generator(
            index=index, namespace=namespace, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents: List[Document] = list(result)
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 32,
        headers: Optional[Dict[str, str]] = None,
        namespace: Optional[str] = None,
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
        :param namespace: Optional namespace to retrieve documents from.
        """
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        if return_embedding is None:
            return_embedding = self.return_embedding

        index = self._index_name(index)
        if index not in self.pinecone_indexes:
            raise PineconeDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        if namespace is None:
            if self.get_embedding_count(index=index) > 0:
                namespace = self.embedding_namespace
            else:
                namespace = self.document_namespace

        ids = self._get_all_document_ids(index=index, namespace=namespace, filters=filters, batch_size=batch_size)

        if filters is not None and len(ids) == 0:
            logger.warning(
                "This query might have been done without metadata indexed and thus no DOCUMENTS were retrieved. "
                "Make sure the desired metadata you want to filter with is indexed."
            )

        for i in range(0, len(ids), batch_size):
            i_end = min(len(ids), i + batch_size)
            documents = self.get_documents_by_id(
                ids=ids[i:i_end],
                index=index,
                namespace=namespace,
                batch_size=batch_size,
                return_embedding=return_embedding,
            )
            for doc in documents:
                yield doc

    def _get_all_document_ids(
        self,
        index: Optional[str] = None,
        namespace: Optional[str] = None,
        filters: Optional[FilterType] = None,
        batch_size: int = 32,
    ) -> List[str]:
        index = self._index_name(index)
        if index not in self.pinecone_indexes:
            raise PineconeDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        if namespace is None:
            if self.get_embedding_count(index=index) > 0:
                namespace = self.embedding_namespace
            else:
                namespace = self.document_namespace

        document_count = self.get_document_count(index=index)

        if index not in self.all_ids:
            self.all_ids[index] = set()
        if len(self.all_ids[index]) == document_count and filters is None:
            # We have all of the IDs and don't need to extract from Pinecone
            return list(self.all_ids[index])
        else:
            # Otherwise we must query and extract IDs from the original namespace, then move the retrieved embeddings
            # to a temporary namespace and query again for new items. We repeat this process until all embeddings
            # have been retrieved.
            target_namespace = f"{namespace}-copy"
            all_ids: Set[str] = set()
            vector_id_matrix = ["dummy-id"]
            with tqdm(
                total=document_count, disable=not self.progress_bar, position=0, unit=" ids", desc="Retrieving IDs"
            ) as progress_bar:
                while len(vector_id_matrix) != 0:
                    # Retrieve IDs from Pinecone
                    vector_id_matrix = self._get_ids(
                        index=index, namespace=namespace, batch_size=batch_size, filters=filters
                    )
                    # Save IDs
                    all_ids = all_ids.union(set(vector_id_matrix))
                    # Move these IDs to new namespace
                    self._move_documents_by_id_namespace(
                        ids=vector_id_matrix,
                        index=index,
                        source_namespace=namespace,
                        target_namespace=target_namespace,
                        batch_size=batch_size,
                    )
                    progress_bar.set_description_str("Retrieved IDs")
                    progress_bar.update(len(set(vector_id_matrix)))
            # Now move all documents back to source namespace
            self._namespace_cleanup(index)
            self._add_local_ids(index, list(all_ids))
            return list(all_ids)

    def _move_documents_by_id_namespace(
        self,
        ids: List[str],
        index: Optional[str] = None,
        source_namespace: Optional[str] = "vectors",
        target_namespace: Optional[str] = "copy",
        batch_size: int = 32,
    ):
        index = self._index_name(index)
        if index not in self.pinecone_indexes:
            raise PineconeDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        if source_namespace == target_namespace:
            raise PineconeDocumentStoreError(
                f"Source namespace '{source_namespace}' cannot be the same as target namespace '{target_namespace}'."
            )
        with tqdm(
            total=len(ids), disable=not self.progress_bar, position=0, unit=" docs", desc="Moving Documents"
        ) as progress_bar:
            for i in range(0, len(ids), batch_size):
                i_end = min(len(ids), i + batch_size)
                # TODO if i == i_end:
                #    break
                id_batch = ids[i:i_end]
                # Retrieve documents from source_namespace
                result = self.pinecone_indexes[index].fetch(ids=id_batch, namespace=source_namespace)
                vector_id_matrix = result["vectors"].keys()
                meta_matrix = [result["vectors"][_id]["metadata"] for _id in vector_id_matrix]
                embedding_matrix = [result["vectors"][_id]["values"] for _id in vector_id_matrix]
                data_to_write_to_pinecone = list(zip(vector_id_matrix, embedding_matrix, meta_matrix))
                # Store metadata nd embeddings in new target_namespace
                self.pinecone_indexes[index].upsert(vectors=data_to_write_to_pinecone, namespace=target_namespace)
                # Delete vectors from source_namespace
                self.delete_documents(index=index, ids=ids[i:i_end], namespace=source_namespace, drop_ids=False)
                progress_bar.set_description_str("Documents Moved")
                progress_bar.update(len(id_batch))

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = 32,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None,
        namespace: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieves all documents in the index using their IDs.

        :param ids: List of IDs to retrieve.
        :param index: Optional index name to retrieve all documents from.
        :param batch_size: Number of documents to retrieve at a time. When working with large number of documents,
            batching can help reduce memory footprint.
        :param headers: Pinecone does not support headers.
        :param return_embedding: Optional flag to return the embedding of the document.
        :param namespace: Optional namespace to retrieve documents from.
        """

        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        if return_embedding is None:
            return_embedding = self.return_embedding

        if namespace is None:
            if self.get_embedding_count(index=index) > 0:
                namespace = self.embedding_namespace
            else:
                namespace = self.document_namespace

        index = self._index_name(index)
        if index not in self.pinecone_indexes:
            raise PineconeDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        documents = []
        for i in range(0, len(ids), batch_size):
            i_end = min(len(ids), i + batch_size)
            id_batch = ids[i:i_end]
            result = self.pinecone_indexes[index].fetch(ids=id_batch, namespace=namespace)

            vector_id_matrix = []
            meta_matrix = []
            embedding_matrix = []
            for _id in result["vectors"].keys():
                vector_id_matrix.append(_id)
                meta_matrix.append(self._pinecone_meta_format(result["vectors"][_id]["metadata"]))
                if return_embedding:
                    embedding_matrix.append(result["vectors"][_id]["values"])
            if return_embedding:
                values = embedding_matrix
            else:
                values = None
            document_batch = self._get_documents_by_meta(
                vector_id_matrix, meta_matrix, values=values, index=index, return_embedding=return_embedding
            )
            documents.extend(document_batch)

        return documents

    def get_document_by_id(
        self,
        id: str,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None,
        namespace: Optional[str] = None,
    ) -> Document:
        """
        Returns a single Document retrieved using an ID.

        :param id: ID string to retrieve.
        :param index: Optional index name to retrieve all documents from.
        :param headers: Pinecone does not support headers.
        :param return_embedding: Optional flag to return the embedding of the document.
        :param namespace: Optional namespace to retrieve documents from.
        """
        documents = self.get_documents_by_id(
            ids=[id], namespace=namespace, index=index, headers=headers, return_embedding=return_embedding
        )
        return documents[0]

    def get_embedding_count(self, index: Optional[str] = None, filters: Optional[FilterType] = None) -> int:
        """
        Return the count of embeddings in the document store.

        :param index: Optional index name to retrieve all documents from.
        :param filters: Filters are not supported for `get_embedding_count` in Pinecone.
        """
        if filters:
            raise NotImplementedError("Filters are not supported for get_embedding_count in PineconeDocumentStore")

        index = self._index_name(index)
        if index not in self.pinecone_indexes:
            raise PineconeDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        stats = self.pinecone_indexes[index].describe_index_stats()
        # if no embeddings namespace return zero
        if self.embedding_namespace in stats["namespaces"]:
            count = stats["namespaces"][self.embedding_namespace]["vector_count"]
        else:
            count = 0
        return count

    def update_document_meta(self, id: str, meta: Dict[str, str], namespace: Optional[str] = None, index: Optional[str] = None):  # type: ignore
        """
        Update the metadata dictionary of a document by specifying its string ID.

        :param id: ID of the Document to update.
        :param meta: Dictionary of new metadata.
        :param namespace: Optional namespace to update documents from. If not specified, defaults to the embedding
            namespace (vectors) if it exists, otherwise the document namespace (no-vectors).
        :param index: Optional index name to update documents from.
        """

        index = self._index_name(index)
        if index not in self.pinecone_indexes:
            raise PineconeDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        if namespace is None:
            if self.get_embedding_count(index=index) > 0:
                namespace = self.embedding_namespace
            else:
                namespace = self.document_namespace

        doc = self.get_documents_by_id(ids=[id], index=index, return_embedding=True)[0]
        if doc.embedding is not None:
            meta = {"content": doc.content, "content_type": doc.content_type, **meta}
            self.pinecone_indexes[index].upsert(vectors=[(id, doc.embedding.tolist(), meta)], namespace=namespace)

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
        drop_ids: Optional[bool] = True,
        namespace: Optional[str] = None,
    ):
        """
        Delete documents from the document store.

        :param index: Index name to delete the documents from. If `None`, the DocumentStore's default index
            (`self.index`) will be used.
        :param ids: Optional list of IDs to narrow down the documents to be deleted.
        :param namespace: Optional namespace string. By default, it deletes vectors from the embeddings namespace
            unless the namespace is empty, in which case it deletes from the documents namespace.
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
        :param drop_ids: Specifies if the locally stored IDs should be deleted. The default
            is True.
        :param namespace: Optional namespace to delete documents from. If not specified, defaults to the embedding
            namespace (vectors) if it exists, otherwise the document namespace (no-vectors).
        :return None:
        """
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        if namespace is None:
            if self.get_embedding_count(index=index) > 0:
                namespace = self.embedding_namespace
            else:
                namespace = self.document_namespace

        index = self._index_name(index)
        if index not in self.pinecone_indexes:
            raise PineconeDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        pinecone_syntax_filter = LogicalFilterClause.parse(filters).convert_to_pinecone() if filters else None

        if ids is None and pinecone_syntax_filter is None:
            # If no filters or IDs we delete everything
            self.pinecone_indexes[index].delete(delete_all=True, namespace=namespace)
            id_values = list(self.all_ids[index])
        else:
            if ids is None:
                # In this case we identify all IDs that satisfy the filter condition
                id_values = self._get_all_document_ids(index=index, namespace=namespace, filters=pinecone_syntax_filter)
            else:
                id_values = ids
            if pinecone_syntax_filter:
                # We must first identify the IDs that satisfy the filter condition
                docs = self.get_all_documents(index=index, namespace=namespace, filters=pinecone_syntax_filter)
                filter_ids = [doc.id for doc in docs]
                # Find the intersect
                id_values = list(set(id_values).intersection(set(filter_ids)))
            if len(id_values) > 0:
                # Now we delete
                self.pinecone_indexes[index].delete(ids=id_values, namespace=namespace)
        if drop_ids:
            self.all_ids[index] = self.all_ids[index].difference(set(id_values))

    def delete_index(self, index: str):
        """
        Delete an existing index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        index = self._index_name(index)
        if index in pinecone.list_indexes():
            pinecone.delete_index(index)
            logger.info("Index '%s' deleted.", index)
        if index in self.pinecone_indexes:
            del self.pinecone_indexes[index]
        if index in self.all_ids:
            self.all_ids[index] = set()

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
        namespace: Optional[str] = None,
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

        pinecone_syntax_filter = LogicalFilterClause.parse(filters).convert_to_pinecone() if filters else None

        index = self._index_name(index)
        if index not in self.pinecone_indexes:
            raise PineconeDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )
        query_emb = query_emb.astype(np.float32)

        if self.similarity == "cosine":
            self.normalize_embedding(query_emb)

        if namespace is None:
            namespace = self.embedding_namespace

        res = self.pinecone_indexes[index].query(
            query_emb.tolist(),
            namespace=namespace,
            top_k=top_k,
            include_values=return_embedding,
            include_metadata=True,
            filter=pinecone_syntax_filter,
        )

        score_matrix = []
        vector_id_matrix = []
        meta_matrix = []
        embedding_matrix = []
        for match in res["matches"]:
            score_matrix.append(match["score"])
            vector_id_matrix.append(match["id"])
            meta_matrix.append(match["metadata"])
            if return_embedding:
                embedding_matrix.append(match["values"])
        if return_embedding:
            values = embedding_matrix
        else:
            values = None
        documents = self._get_documents_by_meta(
            vector_id_matrix, meta_matrix, values=values, index=index, return_embedding=return_embedding
        )

        if filters is not None and len(documents) == 0:
            logger.warning(
                "This query might have been done without metadata indexed and thus no results were retrieved. "
                "Make sure the desired metadata you want to filter with is indexed."
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
        values: Optional[List[List[float]]] = None,
        namespace: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        if return_embedding is None:
            return_embedding = self.return_embedding

        if namespace is None:
            if self.get_embedding_count(index=index) > 0:
                namespace = self.embedding_namespace
            else:
                namespace = self.document_namespace

        index = self._index_name(index)

        # extract ID, content, and metadata to create Documents
        documents = []
        for _id, meta in zip(ids, metadata):
            content = meta.pop("content")
            content_type = meta.pop("content_type")
            doc = Document(id=_id, content=content, content_type=content_type, meta=meta)
            documents.append(doc)
        if return_embedding:
            if values is None:
                # If no embedding values are provided, we must request the embeddings from Pinecone
                for doc in documents:
                    self._attach_embedding_to_document(document=doc, index=index, namespace=namespace)
            else:
                # If embedding values are given, we just add
                for doc, embedding in zip(documents, values):
                    doc.embedding = np.asarray(embedding, dtype=np.float32)

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
                raise PineconeDocumentStoreError(
                    f"PineconeDocumentStore allows requests of no more than {self.top_k_limit_vectors} records "
                    f"when returning embedding values. This request is attempting to return {top_k} records."
                )
        else:
            if top_k > self.top_k_limit:
                raise PineconeDocumentStoreError(
                    f"PineconeDocumentStore allows requests of no more than {self.top_k_limit} records. "
                    f"This request is attempting to return {top_k} records."
                )

    def _list_namespaces(self, index: str) -> List[str]:
        """
        Returns a list of namespaces.
        """
        res = self.pinecone_indexes[index].describe_index_stats()
        namespaces = res["namespaces"].keys()
        return namespaces

    def _check_exists(self, id: str, index: str, namespace: str) -> bool:
        """
        Checks if the specified ID exists in the specified index and namespace.
        """
        res = self.pinecone_indexes[index].fetch(ids=[id], namespace=namespace)
        return bool(res["vectors"].get(id, False))

    def _namespace_cleanup(self, index: str, batch_size: int = 32):
        """
        Searches for any "-copy" namespaces and shifts vectors back to the original namespace.
        """
        namespaces = self._list_namespaces(index)
        namespaces = [name for name in namespaces if name[-5:] == "-copy"]

        with tqdm(
            total=len(namespaces),
            disable=not self.progress_bar,
            position=0,
            unit=" namespaces",
            desc="Cleaning Namespace",
        ) as progress_bar:
            for namespace in namespaces:
                target_namespace = namespace[:-5]
                while True:
                    # Retrieve IDs from Pinecone
                    vector_id_matrix = self._get_ids(index=index, namespace=namespace, batch_size=batch_size)
                    # Once we reach final item, we break
                    if len(vector_id_matrix) == 0:
                        break
                    # Move these IDs to new namespace
                    self._move_documents_by_id_namespace(
                        ids=vector_id_matrix,
                        index=index,
                        source_namespace=namespace,
                        target_namespace=target_namespace,
                        batch_size=batch_size,
                    )
                progress_bar.set_description_str("Cleaned Namespace")
                progress_bar.update(1)

    def _get_ids(
        self, index: str, namespace: str, batch_size: int = 32, filters: Optional[FilterType] = None
    ) -> List[str]:
        """
        Retrieves a list of IDs that satisfy a particular filter condition (or any) using
        a dummy query embedding.
        """
        pinecone_syntax_filter = LogicalFilterClause.parse(filters).convert_to_pinecone() if filters else None

        # Retrieve embeddings from Pinecone
        try:
            res = self.pinecone_indexes[index].query(
                self.dummy_query,
                namespace=namespace,
                top_k=batch_size,
                include_values=False,
                include_metadata=False,
                filter=pinecone_syntax_filter,
            )
        except pinecone.ApiException as e:
            raise PineconeDocumentStoreError(
                f"The API returned an exception.\nReason: {e.reason}\nHeaders: {e.headers}\nBody: {e.body}"
            ) from e

        ids = []
        for match in res["matches"]:
            ids.append(match["id"])
        return ids

    @classmethod
    def load(cls):
        """
        Default class method used for loading indexes. Not applicable to PineconeDocumentStore.
        """
        raise NotImplementedError("load method not supported for PineconeDocumentStore")

    def _meta_for_pinecone(self, meta: Dict[str, Any], parent_key: str = "", labels: bool = False) -> Dict[str, Any]:
        """
        Converts the meta dictionary to a format that can be stored in Pinecone.
        :param meta: Metadata dictionary to be converted.
        :param parent_key: Optional, used for recursive calls to keep track of parent keys, for example:
            ```
            {"parent1": {"parent2": {"child": "value"}}}
            ```
            On the second recursive call, parent_key would be "parent1", and the final key would be "parent1.parent2.child".
        :param labels: Optional, used to indicate whether the metadata is being stored as a label or not. If True the
            the flattening of dictionaries is not required.
        """
        items: list = []
        if labels:
            # Replace any None values with empty strings
            for key, value in meta.items():
                if value is None:
                    meta[key] = ""
        else:
            # Explode dict of dicts into single flattened dict
            for key, value in meta.items():
                # Replace any None values with empty strings
                if value is None:
                    value = ""
                # format key
                new_key = f"{parent_key}.{key}" if parent_key else key
                # if value is dict, expand
                if isinstance(value, dict):
                    items.extend(self._meta_for_pinecone(value, parent_key=new_key).items())
                else:
                    items.append((new_key, value))
            # Create new flattened dictionary
            meta = dict(items)
        return meta

    def _pinecone_meta_format(self, meta: Dict[str, Any], labels: bool = False) -> Dict[str, Any]:
        """
        Converts the meta extracted from Pinecone into a better format for Python.
        :param meta: Metadata dictionary to be converted.
        :param labels: Optional, used to indicate whether the metadata is being stored as a label or not. If True the
            the flattening of dictionaries is not required.
        """
        new_meta: Dict[str, Any] = {}

        if labels:
            # Replace any empty strings with None values
            for key, value in meta.items():
                if value == "":
                    meta[key] = None
            return meta
        else:
            for key, value in meta.items():
                # Replace any empty strings with None values
                if value == "":
                    value = None
                if "." in key:
                    # We must split into nested dictionary
                    keys = key.split(".")
                    # Iterate through each dictionary level
                    for i in range(len(keys)):
                        path = keys[: i + 1]
                        # Check if path exists
                        try:
                            _get_by_path(new_meta, path)
                        except KeyError:
                            # Create path
                            if i == len(keys) - 1:
                                _set_by_path(new_meta, path, value)
                            else:
                                _set_by_path(new_meta, path, {})
                else:
                    new_meta[key] = value
            return new_meta

    def _label_to_meta(self, labels: list) -> dict:
        """
        Converts a list of labels to a dictionary of ID: metadata mappings.
        """
        metadata = {}
        for label in labels:
            # Get main labels data
            meta = {
                "label-id": label.id,
                "query": label.query,
                "label-is-correct-answer": label.is_correct_answer,
                "label-is-correct-document": label.is_correct_document,
                "label-document-content": label.document.content,
                "label-document-id": label.document.id,
                "label-no-answer": label.no_answer,
                "label-origin": label.origin,
                "label-created-at": label.created_at,
                "label-updated-at": label.updated_at,
                "label-pipeline-id": label.pipeline_id,
            }
            # Get document metadata
            if label.document.meta is not None:
                for k, v in label.document.meta.items():
                    meta[f"label-document-meta-{k}"] = v
            # Get label metadata
            if label.meta is not None:
                for k, v in label.meta.items():
                    meta[f"label-meta-{k}"] = v
            # Get Answer data
            if label.answer is not None:
                meta.update(
                    {
                        "label-answer-answer": label.answer.answer,
                        "label-answer-type": label.answer.type,
                        "label-answer-score": label.answer.score,
                        "label-answer-context": label.answer.context,
                        "label-answer-document-ids": label.answer.document_ids,
                    }
                )
                # Get offset data
                if label.answer.offsets_in_document:
                    meta["label-answer-offsets-in-document-start"] = label.answer.offsets_in_document[0].start
                    meta["label-answer-offsets-in-document-end"] = label.answer.offsets_in_document[0].end
                else:
                    meta["label-answer-offsets-in-document-start"] = None
                    meta["label-answer-offsets-in-document-end"] = None
                if label.answer.offsets_in_context:
                    meta["label-answer-offsets-in-context-start"] = label.answer.offsets_in_context[0].start
                    meta["label-answer-offsets-in-context-end"] = label.answer.offsets_in_context[0].end
                else:
                    meta["label-answer-offsets-in-context-start"] = None
                    meta["label-answer-offsets-in-context-end"] = None
            metadata[label.id] = meta
        metadata = self._meta_for_pinecone(metadata, labels=True)
        return metadata

    def _meta_to_labels(self, documents: List[Document]) -> List[Label]:
        """
        Converts a list of metadata dictionaries to a list of Labels.
        """
        labels = []
        for d in documents:
            label_meta = {k: v for k, v in d.meta.items() if k[:6] == "label-" or k == "query"}
            other_meta = {k: v for k, v in d.meta.items() if k[:6] != "label-" and k != "query"}
            # Create document
            doc = Document(
                id=label_meta["label-document-id"], content=d.content, meta={}, score=d.score, embedding=d.embedding
            )
            # Extract document metadata
            for k, v in d.meta.items():
                if k.startswith("label-document-meta-"):
                    doc.meta[k[20:]] = v
            # Extract offsets
            offsets: Dict[str, Optional[List[Span]]] = {"document": None, "context": None}
            for mode in offsets.keys():
                if label_meta.get(f"label-answer-offsets-in-{mode}-start") is not None:
                    offsets[mode] = [
                        Span(
                            label_meta[f"label-answer-offsets-in-{mode}-start"],
                            label_meta[f"label-answer-offsets-in-{mode}-end"],
                        )
                    ]
            # Extract Answer
            answer = None
            if label_meta.get("label-answer-answer") is not None:
                # backwards compatibility: if legacy answer object with `document_id` is present, convert to `document_ids
                if "label-answer-document-id" in label_meta:
                    document_id = label_meta["label-answer-document-id"]
                    document_ids = [document_id] if document_id is not None else None
                else:
                    document_ids = label_meta["label-answer-document-ids"]

                answer = Answer(
                    answer=label_meta["label-answer-answer"]
                    or "",  # If we leave as None a schema validation error will be thrown
                    type=label_meta["label-answer-type"],
                    score=label_meta["label-answer-score"],
                    context=label_meta["label-answer-context"],
                    offsets_in_document=offsets["document"],
                    offsets_in_context=offsets["context"],
                    document_ids=document_ids,
                    meta=other_meta,
                )
            # Extract Label metadata
            label_meta_metadata = {}
            for k, v in d.meta.items():
                if k.startswith("label-meta-"):
                    label_meta_metadata[k[11:]] = v
            # Rebuild Label object
            label = Label(
                id=label_meta["label-id"],
                query=label_meta["query"],
                document=doc,
                answer=answer,
                pipeline_id=label_meta["label-pipeline-id"],
                created_at=label_meta["label-created-at"],
                updated_at=label_meta["label-updated-at"],
                is_correct_answer=label_meta["label-is-correct-answer"],
                is_correct_document=label_meta["label-is-correct-document"],
                origin=label_meta["label-origin"],
                meta=label_meta_metadata,
                filters=None,
            )
            labels.append(label)
        return labels

    def delete_labels(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: int = 32,
    ):
        """
        Default class method used for deleting labels. Not supported by PineconeDocumentStore.
        """
        index = self._index_name(index)
        if index not in self.pinecone_indexes:
            raise PineconeDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        pinecone_syntax_filter = LogicalFilterClause.parse(filters).convert_to_pinecone() if filters else None

        i = 0
        dummy_query = np.asarray(self.dummy_query)
        # Set label namespace
        namespace = "labels"

        while True:
            if ids is None:
                # Iteratively upsert new records without the labels metadata
                docs = self.query_by_embedding(
                    dummy_query,
                    filters=pinecone_syntax_filter,
                    top_k=batch_size,
                    index=index,
                    return_embedding=True,
                    namespace=namespace,
                )
                update_ids = [doc.id for doc in docs]
            else:
                i_end = min(i + batch_size, len(ids))
                update_ids = ids[i:i_end]
                if pinecone_syntax_filter:
                    pinecone_syntax_filter["label-id"] = {"$in": update_ids}
                else:
                    pinecone_syntax_filter = {"label-id": {"$in": update_ids}}
                # Retrieve embeddings and metadata for the batch of documents
                docs = self.query_by_embedding(
                    dummy_query,
                    filters=pinecone_syntax_filter,
                    top_k=batch_size,
                    index=index,
                    return_embedding=True,
                    namespace=namespace,
                )
                # Apply filter to update IDs, finding intersection
                update_ids = list(set(update_ids).intersection({doc.id for doc in docs}))
                i = i_end
            if len(update_ids) == 0:
                break
            # Delete the documents
            self.delete_documents(ids=update_ids, index=index, namespace=namespace)

    def get_all_labels(
        self, index=None, filters: Optional[FilterType] = None, headers: Optional[Dict[str, str]] = None
    ):
        """
        Default class method used for getting all labels.
        """
        index = self._index_name(index)
        if index not in self.pinecone_indexes:
            raise PineconeDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        documents = self.get_all_documents(index=index, filters=filters, headers=headers, namespace="labels")
        for doc in documents:
            doc.meta = self._pinecone_meta_format(doc.meta, labels=True)
        labels = self._meta_to_labels(documents)
        return labels

    def get_label_count(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        """
        Default class method used for counting labels. Not supported by PineconeDocumentStore.
        """
        raise NotImplementedError("Labels are not supported by PineconeDocumentStore.")

    def write_labels(self, labels, index=None, headers: Optional[Dict[str, str]] = None):
        """
        Default class method used for writing labels.
        """
        index = self._index_name(index)
        if index not in self.pinecone_indexes:
            self.pinecone_indexes[index] = self._create_index(
                embedding_dim=self.embedding_dim,
                index=index,
                metric_type=self.metric_type,
                replicas=self.replicas,
                shards=self.shards,
                recreate_index=False,
                metadata_config=self.metadata_config,
            )

        # Convert Label objects to dictionary of metadata
        metadata = self._label_to_meta(labels)
        ids = list(metadata.keys())
        # Set label namespace
        namespace = "labels"
        # Check if vectors exist in the namespace
        existing_documents = self.get_documents_by_id(ids=ids, index=index, namespace=namespace, return_embedding=True)
        if len(existing_documents) != 0:
            # If they exist, we loop through and partial update their metadata with the new labels
            existing_ids = [doc.id for doc in existing_documents]
            for _id in existing_ids:
                meta = self._meta_for_pinecone(metadata[_id])
                self.pinecone_indexes[index].update(id=_id, set_metadata=meta, namespace=namespace)
                # After update, we delete the ID from the metadata list
                del metadata[_id]
        # If there are any remaining IDs, we create new documents with the remaining metadata
        if len(metadata) != 0:
            documents = []
            for _id, meta in metadata.items():
                metadata[_id] = self._meta_for_pinecone(meta)
                documents.append(Document(id=_id, content=meta["label-document-content"], meta=meta))
            self.write_documents(documents, index=index, labels=True)
