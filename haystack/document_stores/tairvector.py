import copy
from typing import Set, Union, List, Optional, Dict, Generator, Any

import json
import logging
from itertools import islice
from functools import reduce
import operator

import tair
import numpy as np
from tqdm.auto import tqdm

from haystack.schema import Document, FilterType, Label, Answer, Span
from haystack.document_stores import BaseDocumentStore

from haystack.document_stores.filter_utils import LogicalFilterClause
from haystack.errors import TairDocumentStoreError, DuplicateDocumentError
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


class TairDocumentStore(BaseDocumentStore):
    top_k_limit = 10_000
    top_k_limit_vectors = 1_000

    def __init__(
        self,
        url: str,
        tair_index: Optional["tair.TairVectorIndex"] = None,
        embedding_dim: int = 768,
        return_embedding: bool = False,
        index: str = "document",
        similarity: str = "COSINE",
        index_type: str = "HNSW",
        data_type: str = "FLOAT32",
        embedding_field: str = "embedding",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        recreate_index: bool = False,
        **kwargs: Any,
    ):
        """
        :param url: Tair vector database url corresponding to a instance(https://www.alibabacloud.com/help/en/tair/latest/tairvector).
        :param tair_index: tair-client Index object, an index will be initialized or loaded if not specified.
        :param embedding_dim: The embedding vector size.
        :param return_embedding: Whether to return document embeddings.
        :param index: Name of index in document store to use.
        :param similarity: The similarity function used to compare document vectors. `"COSINE"` is the default
            and is recommended if you are using a Sentence-Transformer model. `"IP"` is more performant
            with DPR embeddings.
            In both cases, the returned values in Document.score are normalized to be in range [0,1]:
                - For `"IP"`: `expit(np.asarray(raw_score / 100))`
                - For `"COSINE"`: `(raw_score + 1) / 2`
        :param index_type: The type of indexing algorithms. Valid values: "HNSW" creates graph-based vector indexes,
            "FLAT" uses the Flat Search algorithm to search for vectors without creating indexes.
        :param data_type: the data type of the vector. Valid values: "FLOAT32", "FLOAT16", "BINARY"
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
        """
        # Connect to Tair server using python client binding
        if not url:
            raise TairDocumentStoreError(
                "Tair requires an url, please provide one. https://www.alibabacloud.com/help/en/tair/latest/tairvector,"
                "The format of url: redis://[[username]:[password]]@localhost:6379/0"
            )

        # Formal similarity string
        if similarity == "IP" or "COSINE":
            self.distance_type = similarity
        elif similarity in ("L2", "Euclidean"):
            self.distance_type = "L2"
        else:
            raise ValueError(
                "The Tair document store can currently only support inner_product and euclidean metrics. "
                "Please set distance_type to one of the above."
            )

        try:
            from tair import Tair as TairClient
        except ImportError:
            raise ValueError(
                "Could not import tair python package. "
                "Please install it with `pip install tair`."
            )
        try:
            # connect to tair from url
            client = TairClient.from_url(url, **kwargs)
        except ValueError as e:
            raise ValueError(f"Tair failed to connect: {e}")

        self.client = client
        self.similarity = similarity
        self.index: str = self._index_name(index)
        self.embedding_dim = embedding_dim
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents

        # Tair index params
        self.index_type = index_type
        self.data_type = data_type

        # Initialize dictionary of index connections
        self.tair_indexes: Dict[str, tair.TairVectorIndex] = {}

        # Initialize dictionary to store temporary set of document IDs
        self.all_ids: dict = {}
        # Dummy query to be used during searches
        self.dummy_query = [0.0] * self.embedding_dim

        if tair_index:
            if not isinstance(tair_index, tair.TairVectorIndex):
                raise TairDocumentStoreError(
                    f"The parameter `tair_index` needs to be a "
                    f"`tair.TairVectorIndex` object. You provided an object of "
                    f"type `{type(tair_index)}`."
                )
            self.tair_indexes[self.index] = tair_index
        else:
            self.tair_indexes[self.index] = self._create_index(
                embedding_dim=self.embedding_dim,
                index=self.index,
                distance_type=self.distance_type,
                index_type=self.index_type,
                data_type=self.data_type,
                recreate_index=recreate_index,
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
        return index

    def _create_index(
        self,
        embedding_dim: int,
        index: Optional[str] = None,
        distance_type: Optional[str] = "IP",
        index_type: Optional[str] = "HNSW",
        data_type: Optional[str] = "FLOAT32",
        recreate_index: bool = False,
        **kwargs,
    ):
        """
        Create a new index for storing documents in case an
        index with the name doesn't exist already.
        """
        index = self._index_name(index)

        if recreate_index:
            self.delete_index(index)

        # Skip if already exists
        index_connection = self.client.tvs_get_index(index)
        if index_connection is not None:
            logger.info("Index already exists")
            return index_connection
        else:
            self.client.tvs_create_index(
                name=index,
                dim=embedding_dim,
                distance_type=distance_type,
                index_type=index_type,
                data_type=data_type,
                **kwargs
            )
        index_connection = self.client.tvs_get_index(index)

        return index_connection

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        duplicate_documents: Optional[str] = 'overwrite',
        headers: Optional[Dict[str, str]] = None,
        labels: Optional[bool] = False,
    ):
        """
        Indexes documents for later queries.

        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: Optional name of index where the documents shall be written to.
                      If None, the DocumentStore's default index (self.index) will be used.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

        :param labels: Tells us whether these records are labels or not. Defaults to False.
        :return: None
        """
        if headers:
            raise NotImplementedError("TairDocumentStore does not support headers.")

        index = self._index_name(index)
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert (
            duplicate_documents in self.duplicate_documents_options
        ), f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        if index not in self.tair_indexes:
            self.tair_indexes[index] = self._create_index(
                embedding_dim=self.embedding_dim,
                index=self.index,
                distance_type=self.distance_type,
                index_type=self.index_type,
                data_type=self.data_type,
                recreate_index=False,
            )

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(
            documents=document_objects, index=index, duplicate_documents=duplicate_documents
        )
        if len(document_objects) > 0:
            add_vectors = False if document_objects[0].embedding is None else True

            if not add_vectors:
                # To store documents, we use dummy embeddings (to be replaced with real embeddings later)
                embeddings_to_index = np.random.rand(batch_size, self.embedding_dim).astype(np.float32)
                # Convert embeddings to list objects
                embeddings = [embed.tolist() if embed is not None else None for embed in embeddings_to_index]
            with tqdm(
                    total=len(document_objects), disable=not self.progress_bar, position=0, desc="Writing Documents"
            ) as progress_bar:
                for i in range(0, len(document_objects), batch_size):
                    document_batch = document_objects[i: i + batch_size]
                    ids = [doc.id for doc in document_batch]
                    # If duplicate_documents set to skip or fail, we need to check for existing documents
                    if duplicate_documents in ["skip", "fail"]:
                        existing_documents = self.get_documents_by_id(ids=ids, index=index)
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
                        self._meta_for_tair({"content": doc.content, "content_type": doc.content_type, **doc.meta})
                        for doc in document_objects[i: i + batch_size]
                    ]
                    if add_vectors:
                        embeddings = [doc.embedding for doc in document_objects[i: i + batch_size]]
                        embeddings_to_index = np.array(embeddings, dtype="float32")
                        if self.similarity == "COSINE":
                            # Normalize embeddings inplace
                            self.normalize_embedding(embeddings_to_index)
                        # Convert embeddings to list objects
                        embeddings = [embed.tolist() if embed is not None else None for embed in embeddings_to_index]

                    # Insert documents into index
                    for j in range(len(ids)):
                        for meta_key in metadata[j]:
                            if isinstance(metadata[j][meta_key], bool):
                                metadata[j][meta_key] = str(metadata[j][meta_key])
                            if isinstance(metadata[j][meta_key], list):
                                metadata[j][meta_key] = ','.join(metadata[j][meta_key])
                        self.client.tvs_hset(
                            index=index,
                            key=ids[j],
                            vector=embeddings[j],
                            is_binary=False,
                            **{
                                "meta": json.dumps(metadata[j]),
                            },
                            **(metadata[j])
                        )
                    # Add IDs to ID list
                    self._add_local_ids(index, ids)
                    progress_bar.update(batch_size)
            progress_bar.close()

    def _meta_for_tair(self, meta: Dict[str, Any], parent_key: str = "", labels: bool = False) -> Dict[str, Any]:
        """
        Converts the meta dictionary to a format that can be stored in Tair.
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
                if key == "_split_overlap":
                    value = json.dumps(value)
                # format key
                new_key = f"{parent_key}.{key}" if parent_key else key
                # if value is dict, expand
                if isinstance(value, dict):
                    items.extend(self._meta_for_tair(value, parent_key=new_key).items())
                else:
                    items.append((new_key, value))
            # Create new flattened dictionary
            meta = dict(items)
        return meta

    def _create_document_field_map(self) -> Dict:
        return {self.embedding_field: "embedding"}

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = 32,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:
        """
        Retrieves all documents in the index using their IDs.

        :param ids: List of IDs to retrieve.
        :param index: Optional index name to retrieve all documents from.
        :param batch_size: Number of documents to retrieve at a time. When working with large number of documents,
            batching can help reduce memory footprint.
        :param headers: Tair does not support headers.
        :param return_embedding: Optional flag to return the embedding of the document.
        """

        if headers:
            raise NotImplementedError("TairDocumentStore does not support headers.")

        if return_embedding is None:
            return_embedding = self.return_embedding

        index = self._index_name(index)
        if index not in self.tair_indexes:
            raise TairDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing TairDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        documents = []
        for i in range(0, len(ids), batch_size):
            i_end = min(len(ids), i + batch_size)
            id_batch = ids[i:i_end]
            result = []
            vector_id_matrix = []
            meta_matrix = []
            embedding_matrix = []
            for id_iter in id_batch:
                result_iter = self.client.tvs_hgetall(index, id_iter)
                if len(result_iter)>0:
                    result.append(result_iter)
                    vector_id_matrix.append(id_iter)
                    embedding_matrix.append(result_iter['VECTOR'])
                    meta_matrix.append(json.loads(result_iter['meta']))

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
    ) -> Optional[Document]:
        """
        Returns a single Document retrieved using an ID.

        :param id: ID string to retrieve.
        :param index: Optional index name to retrieve all documents from.
        :param headers: Tair does not support headers.
        :param return_embedding: Optional flag to return the embedding of the document.
        """
        documents = self.get_documents_by_id(
            ids=[id], index=index, headers=headers, return_embedding=return_embedding
        )
        return documents[0]

    def update_embeddings(
        self,
        retriever: DenseRetriever,
        index: Optional[str] = None,
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
        if index not in self.tair_indexes:
            raise ValueError(
                f"Couldn't find a the index '{index}' in Tair. Try to init the "
                f"TairDocumentStore() again ..."
            )
        document_count = self.get_document_count(index=index, filters=filters)
        if document_count == 0:
            logger.warning("Calling DocumentStore.update_embeddings() on an empty index")
            return

        logger.info("Updating embeddings for %s docs...", document_count)

        documents = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=False, batch_size=batch_size
        )

        with tqdm(
            total=document_count, disable=not self.progress_bar, position=0, unit=" docs", desc="Updating Embedding"
        ) as progress_bar:
            for _ in range(0, document_count, batch_size):
                document_batch = list(islice(documents, batch_size))
                embeddings = retriever.embed_documents(document_batch)
                if embeddings.size == 0:
                    # Skip batch if there are no embeddings. Otherwise, incorrect embedding shape will be inferred and
                    # Tair APi will return a "No vectors provided" Bad Request Error
                    progress_bar.set_description_str("Documents Processed")
                    progress_bar.update(batch_size)
                    continue
                self._validate_embeddings_shape(
                    embeddings=embeddings, num_documents=len(document_batch), embedding_dim=self.embedding_dim
                )

                if self.similarity == "COSINE":
                    self.normalize_embedding(embeddings)

                metadata = []
                ids = []
                for doc in document_batch:
                    metadata.append(
                        self._meta_for_tair({"content": doc.content, "content_type": doc.content_type, **doc.meta})
                    )
                    ids.append(doc.id)
                # Update existing vectors in tair index
                for j in range(len(ids)):
                    self.client.tvs_hset(
                        index=index,
                        key=ids[j],
                        vector=embeddings[j].tolist(),
                        is_binary=False,
                        **{
                            "meta": json.dumps(metadata[j]),
                        },
                    )
                # Delete existing vectors if they exist there
                self.delete_documents(index=index, ids=ids)
                # Add these vector IDs to local store
                self._add_local_ids(index, ids)
                progress_bar.set_description_str("Documents Processed")
                progress_bar.update(batch_size)

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
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
                            ```
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        """
        if headers:
            raise NotImplementedError("TairDocumentStore does not support headers.")

        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents: List[Document] = list(result)
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
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
                        ```

        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        """
        if headers:
            raise NotImplementedError("TairDocumentStore does not support headers.")

        if return_embedding is None:
            return_embedding = self.return_embedding

        index = self._index_name(index)
        if index not in self.tair_indexes:
            raise TairDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing TairDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        ids = self._get_all_document_ids(index=index, filters=filters, batch_size=batch_size)

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
                batch_size=batch_size,
                return_embedding=return_embedding,
            )
            for doc in documents:
                yield doc

    def _get_all_document_ids(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        batch_size: int = 32,
    ) -> List[str]:
        index = self._index_name(index)
        if index not in self.tair_indexes:
            raise TairDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing TairDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        document_count = self.get_document_count(index=index)

        if index not in self.all_ids:
            self.all_ids[index] = set()
        if len(self.all_ids[index]) == int(document_count) and filters is None:
            # We have all of the IDs and don't need to extract from Tair
            return list(self.all_ids[index])
        else:
            all_ids: Set[str] = set()
            vector_id_matrix = ["dummy-id"]
            if len(vector_id_matrix) != 0:
                # Retrieve IDs from Tair
                vector_id_matrix = self._get_ids(
                    index=index, batch_size=batch_size, filters=filters
                )
                # Save IDs
                all_ids = all_ids.union(set(vector_id_matrix))

            self._add_local_ids(index, list(all_ids))
            return list(all_ids)


    def get_all_labels(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Label]:
        """
        Default class method used for getting all labels.
        """
        index = self._index_name(index)
        if index not in self.tair_indexes:
            raise TairDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing TairDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        documents = self.get_all_documents(index=index, filters=filters, headers=headers)
        for doc in documents:
            doc.meta = self._tair_meta_format(doc.meta, labels=True)
        labels = self._meta_to_labels(documents)
        return labels

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """
                Return the count of embeddings in the document store.
        """
        if headers:
            raise NotImplementedError("TairDocumentStore does not support headers.")

        index = self._index_name(index)
        if index not in self.tair_indexes:
            raise TairDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing TairDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        tair_syntax_filter = LogicalFilterClause.parse(filters).convert_to_tair() if filters else None

        stats = self.client.tvs_get_index(index)

        # Document count is total number of vectors(no-vectors + vectors)
        count = int(stats["data_count"])
        if tair_syntax_filter is None:
            return count

        res = self.client.tvs_knnsearch(index=index, k=count,
                                        vector=self.dummy_query, filter_str=tair_syntax_filter)

        return len(res)

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
    ) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.
        """
        if headers:
            raise NotImplementedError("TairDocumentStore does not support headers.")

        if return_embedding is None:
            return_embedding = self.return_embedding
        self._limit_check(top_k, include_values=return_embedding)

        tair_syntax_filter = LogicalFilterClause.parse(filters).convert_to_tair() if filters else None

        index = self._index_name(index)
        if index not in self.tair_indexes:
            raise TairDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing TairDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )
        query_emb = query_emb.astype(np.float32)

        self.normalize_embedding(query_emb)

        res_content = []
        res = self.client.tvs_knnsearch(index=index, k=top_k, vector=query_emb, filter_str=tair_syntax_filter)
        for res_iter in res:
            result_iter = self.client.tvs_hgetall(index, res_iter[0].decode('utf-8'))
            res_content.append(json.loads(result_iter["meta"])["content"])
        print("searching result:", res_content)

        score_matrix = []
        vector_id_matrix = []
        for match in res:
            score_matrix.append(1/match[1]) # score is calculated by inverse of distance
            vector_id = match[0].decode('utf-8')
            vector_id_matrix.append(vector_id)

        documents = self.get_documents_by_id(ids=vector_id_matrix, index=index, batch_size=top_k)

        if filters is not None and len(documents) == 0:
            logger.warning(
                "This query might have been done without metadata indexed and thus no results were retrieved. "
                "Make sure the desired metadata you want to filter with is indexed."
            )

        # assign query score to each document
        scores_for_vector_ids: Dict[str, float] = {str(v_id): s for v_id, s in zip(vector_id_matrix, score_matrix)}
        return_documents = []
        for doc in documents:
            score = scores_for_vector_ids[doc.id]
            if scale_score:
                score = self.scale_to_unit_interval(score, self.similarity)
            doc.score = score
            return_document = copy.copy(doc)
            return_documents.append(return_document)

        return return_documents

    def get_label_count(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int:
        """
        Default class method used for counting labels. Not supported by TairDocumentStore.
        """
        raise NotImplementedError("Counting labels is not supported by TairDocumentStore.")

    def write_labels(
        self,
        labels: Union[List[Label], List[dict]],
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Default class method used for writing labels.
        """
        index = self._index_name(index)
        if index not in self.tair_indexes:
            self.tair_indexes[index] = self._create_index(
                embedding_dim=self.embedding_dim,
                index=index,
                distance_type=self.distance_type,
                index_type=self.index_type,
                data_type=self.data_type,
                recreate_index=False,
            )

        # Convert Label objects to dictionary of metadata
        metadata = self._label_to_meta(labels)
        ids = list(metadata.keys())
        existing_documents = self.get_documents_by_id(ids=ids, index=index, return_embedding=True)
        if len(existing_documents) != 0:
            # If they exist, we loop through and partial update their metadata with the new labels
            existing_ids = [doc.id for doc in existing_documents]
            for _id in existing_ids:
                meta = self._meta_for_tair(metadata[_id])
                self.client.tvs_hset(index=index, key=_id, kwargs=meta)
                # After update, we delete the ID from the metadata list
                del metadata[_id]
        # If there are any remaining IDs, we create new documents with the remaining metadata
        if len(metadata) != 0:
            documents = []
            for _id, meta in metadata.items():
                metadata[_id] = self._meta_for_tair(meta)
                documents.append(Document(id=_id, content=meta["label-document-content"], meta=meta))
            self.write_documents(documents, index=index, labels=True)

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
        drop_ids: Optional[bool] = True,
    ):
        if headers:
            raise NotImplementedError("TairDocumentStore does not support headers.")
        index = self._index_name(index)
        if index not in self.tair_indexes:
            raise TairDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing TairDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        tair_syntax_filter = LogicalFilterClause.parse(filters).convert_to_tair() if filters else None

        if index not in self.all_ids:
            self.all_ids[index] = set()
        if ids is None and tair_syntax_filter is None:
            # If no filters or IDs we delete everything
            id_values = list(self.all_ids[index])
            for id_value in id_values:
                self.client.tvs_del(index, id_value)
        else:
            if ids is None:
                # In this case we identify all IDs that satisfy the filter condition
                id_values = self._get_all_document_ids(index=index, filters=filters)
            else:
                id_values = ids
            if tair_syntax_filter:
                # We must first identify the IDs that satisfy the filter condition
                docs = self.get_all_documents(index=index, filters=filters)
                filter_ids = [doc.id for doc in docs]
                # Find the intersection
                id_values = list(set(id_values).intersection(set(filter_ids)))
            if len(id_values) > 0:
                for id_value in id_values:
                    # Now we delete
                    self.client.tvs_del(index, id_value)
        if drop_ids:
            self.all_ids[index] = self.all_ids[index].difference(set(id_values))

    def delete_labels(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: int = 32,
    ):
        index = self._index_name(index)
        if index not in self.tair_indexes:
            raise TairDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing TairDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        i = 0
        dummy_query = np.asarray(self.dummy_query)

        while True:
            docs = self.query_by_embedding(
                dummy_query,
                filters=filters,
                top_k=batch_size,
                index=index,
                return_embedding=True,
            )
            if ids is None:
                # Iteratively upsert new records without the labels metadata
                update_ids = [doc.id for doc in docs]
            else:
                i_end = min(i + batch_size, len(ids))
                update_ids = ids[i:i_end]
                # Apply filter to update IDs, finding intersection
                update_ids = list(set(update_ids).intersection({doc.id for doc in docs}))
                i = i_end
            if len(update_ids) == 0:
                break
            # Delete the documents
            self.delete_documents(ids=update_ids, index=index)

    def delete_index(self, index: str):
        """
        Delete an existing index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        index = self._index_name(index)
        ret = self.client.tvs_del_index(index)
        if ret == 0:
            # index not exist
            logger.info("Index does not exist")
        else:
            logger.info("Index '%s' deleted.", index)
        if index in self.tair_indexes:
            del self.tair_indexes[index]
        if index in self.all_ids:
            self.all_ids[index] = set()

    def get_embedding_count(self, index: Optional[str] = None, filters: Optional[FilterType] = None) -> int:
        """
        Return the count of embeddings in the document store.

        :param index: Optional index name to retrieve all documents from.
        :param filters: Filters are not supported for `get_embedding_count` in Tair.
        """
        if filters:
            raise NotImplementedError("Filters are not supported for get_embedding_count in TairDocumentStore")

        index = self._index_name(index)
        if index not in self.tair_indexes:
            raise TairDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing TairDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )
        # TODO: distinguish vector and non-vector
        stats = self.client.tvs_get_index(index)
        count = int(stats["current_record_count"]) - int(stats["delete_record_count"])
        return count


    def update_document_meta(self, id: str, meta: Dict[str, Any], index: Optional[str] = None):
        index = self._index_name(index)
        if index not in self.tair_indexes:
            raise TairDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing TairDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )

        doc = self.get_documents_by_id(ids=[id], index=index, return_embedding=True)[0]
        if doc.embedding is not None:
            meta = {"content": doc.content, "content_type": doc.content_type, **meta}
            self.client.tvs_hset(index=index, key=id, vector=doc.embedding,
                                 **{"meta": json.dumps(meta)}, **meta)

    def _tair_meta_format(self, meta: Dict[str, Any], labels: bool = False) -> Dict[str, Any]:
        """
        Converts the meta extracted from Tair into a better format for Python.
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

    def _get_documents_by_meta(
        self,
        ids: List[str],
        metadata: List[dict],
        values: Optional[List[List[float]]] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:
        if headers:
            raise NotImplementedError("TairDocumentStore does not support headers.")

        if return_embedding is None:
            return_embedding = self.return_embedding

        index = self._index_name(index)

        # extract ID, content, and metadata to create Documents
        documents = []
        for _id, meta in zip(ids, metadata):
            content = meta.pop("content")
            content_type = meta.pop("content_type")
            if "_split_overlap" in meta:
                meta["_split_overlap"] = json.loads(meta["_split_overlap"])
            doc = Document(id=_id, content=content, content_type=content_type, meta=meta)
            documents.append(doc)
        if return_embedding:
            if values is None:
                # If no embedding values are provided, we must request the embeddings from Tair
                for doc in documents:
                    self._attach_embedding_to_document(document=doc, index=index)
            else:
                # If embedding values are given, we just add
                for doc, embedding in zip(documents, values):
                    doc.embedding = np.asarray(embedding, dtype=np.float32)

        return documents

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
        metadata = self._meta_for_tair(metadata, labels=True)
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
                if document_ids is not None:
                    document_ids_split = document_ids.split(',')
                else:
                    document_ids_split = None

                answer = Answer(
                    answer=label_meta["label-answer-answer"],  # If we leave as None a schema validation error will be thrown
                    type=label_meta["label-answer-type"],
                    score=label_meta["label-answer-score"],
                    context=label_meta["label-answer-context"],
                    offsets_in_document=offsets["document"],
                    offsets_in_context=offsets["context"],
                    document_ids=document_ids_split,
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

    def _attach_embedding_to_document(self, document: Document, index: str):
        """
        Fetches the Document's embedding from the specified Tair index and attaches it to the Document's
        embedding field.
        """
        result = self.client.tvs_hgetall(index, [document.id])
        if result["vectors"].get(document.id, False):
            embedding = result["vectors"][document.id].get("values", None)
            document.embedding = np.asarray(embedding, dtype=np.float32)

    def _limit_check(self, top_k: int, include_values: Optional[bool] = None):
        """
        Confirms the top_k value does not exceed Tair vector database limits.
        """
        if include_values:
            if top_k > self.top_k_limit_vectors:
                raise TairDocumentStoreError(
                    f"TairDocumentStore allows requests of no more than {self.top_k_limit_vectors} records "
                    f"when returning embedding values. This request is attempting to return {top_k} records."
                )
        else:
            if top_k > self.top_k_limit:
                raise TairDocumentStoreError(
                    f"TairDocumentStore allows requests of no more than {self.top_k_limit} records. "
                    f"This request is attempting to return {top_k} records."
                )

    def _get_ids(
        self, index: str, batch_size: int = 32, filters: Optional[FilterType] = None
    ) -> List[str]:
        """
        Retrieves a list of IDs that satisfy a particular filter condition (or any) using
        a dummy query embedding.
        """
        tair_syntax_filter = LogicalFilterClause.parse(filters).convert_to_tair() if filters else None

        # Retrieve embeddings from Tair
        res = self.client.tvs_knnsearch(index=index, k=batch_size,
                                        vector=self.dummy_query, filter_str=tair_syntax_filter)

        ids = []
        for match in res:
            ids.append(match[0].decode('utf-8'))
        return ids
