from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Generator

import time
import logging
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from haystack.schema import Document, Label
from haystack.errors import DuplicateDocumentError
from haystack.document_stores import BaseDocumentStore
from haystack.document_stores.base import get_batches_from_generator
from haystack.modeling.utils import initialize_device_settings
from haystack.document_stores.filter_utils import LogicalFilterClause

if TYPE_CHECKING:
    from haystack.nodes.retriever import BaseRetriever

logger = logging.getLogger(__name__)


class InMemoryDocumentStore(BaseDocumentStore):
    """
    In-memory document store
    """

    def __init__(
        self,
        index: str = "document",
        label_index: str = "label",
        embedding_field: Optional[str] = "embedding",
        embedding_dim: int = 768,
        return_embedding: bool = False,
        similarity: str = "dot_product",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        use_gpu: bool = True,
        scoring_batch_size: int = 500000,
    ):
        """
        :param index: The documents are scoped to an index attribute that can be used when writing, querying,
                      or deleting documents. This parameter sets the default value for document index.
        :param label_index: The default value of index attribute for the labels.
        :param embedding_field: Name of field containing an embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
        :param embedding_dim: The size of the embedding vector.
        :param return_embedding: To return document embedding
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default sine it is
                   more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence BERT model.
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param use_gpu: Whether to use a GPU or the CPU for calculating embedding similarity.
                        Falls back to CPU if no GPU is available.
        :param scoring_batch_size: Batch size of documents to calculate similarity for. Very small batch sizes are inefficent.
                                   Very large batch sizes can overrun GPU memory. In general you want to make sure
                                   you have at least `embedding_dim`*`scoring_batch_size`*4 bytes available in GPU memory.
                                   Since the data is originally stored in CPU memory there is little risk of overruning memory
                                   when running on CPU.
        """
        super().__init__()

        self.indexes: Dict[str, Dict] = defaultdict(dict)
        self.index: str = index
        self.label_index: str = label_index
        self.embedding_field = embedding_field
        self.embedding_dim = embedding_dim
        self.return_embedding = return_embedding
        self.similarity = similarity
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents
        self.use_gpu = use_gpu
        self.scoring_batch_size = scoring_batch_size

        self.devices, _ = initialize_device_settings(use_cuda=self.use_gpu)
        self.main_device = self.devices[0]

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
         Indexes documents for later queries.


        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                           For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                           Optionally: Include meta data via {"text": "<the-actual-text>",
                           "meta": {"name": "<some-document-name>, "author": "somebody", ...}}
                           It can be used for filtering and is accessible in the responses of the Finder.
         :param index: write documents to a custom namespace. For instance, documents for evaluation can be indexed in a
                       separate index than the documents for search.
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
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        index = index or self.index
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert (
            duplicate_documents in self.duplicate_documents_options
        ), f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        field_map = self._create_document_field_map()
        documents = deepcopy(documents)
        documents_objects = [
            Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents
        ]
        documents_objects = self._drop_duplicate_documents(documents=documents_objects)
        for document in documents_objects:
            if document.id in self.indexes[index]:
                if duplicate_documents == "fail":
                    raise DuplicateDocumentError(
                        f"Document with id '{document.id} already " f"exists in index '{index}'"
                    )
                if duplicate_documents == "skip":
                    logger.warning(
                        f"Duplicate Documents: Document with id '{document.id} already exists in index " f"'{index}'"
                    )
                    continue
            self.indexes[index][document.id] = document

    def _create_document_field_map(self):
        return {self.embedding_field: "embedding"}

    def write_labels(
        self,
        labels: Union[List[dict], List[Label]],
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Write annotation labels into document store.
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        index = index or self.label_index
        label_objects = [Label.from_dict(l) if isinstance(l, dict) else l for l in labels]

        duplicate_ids: list = [label.id for label in self._get_duplicate_labels(label_objects, index=index)]
        if len(duplicate_ids) > 0:
            logger.warning(
                f"Duplicate Label IDs: Inserting a Label whose id already exists in this document store."
                f" This will overwrite the old Label. Please make sure Label.id is a unique identifier of"
                f" the answer annotation and not the question."
                f" Problematic ids: {','.join(duplicate_ids)}"
            )

        for label in label_objects:
            # create timestamps if not available yet
            if not label.created_at:
                label.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
            if not label.updated_at:
                label.updated_at = label.created_at
            self.indexes[index][label.id] = label

    def get_document_by_id(
        self, id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> Optional[Document]:
        """
        Fetch a document by specifying its text id string.
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        index = index or self.index
        documents = self.get_documents_by_id([id], index=index)
        if documents:
            return documents[0]
        else:
            return None

    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None) -> List[Document]:  # type: ignore
        """
        Fetch documents by specifying a list of text id strings.
        """
        index = index or self.index
        documents = [self.indexes[index][id] for id in ids]
        return documents

    def get_scores_torch(self, query_emb: np.ndarray, document_to_search: List[Document]) -> List[float]:
        """
        Calculate similarity scores between query embedding and a list of documents using torch.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param document_to_search: List of documents to compare `query_emb` against.
        """
        query_emb = torch.tensor(query_emb, dtype=torch.float).to(self.main_device)
        if len(query_emb.shape) == 1:
            query_emb = query_emb.unsqueeze(dim=0)

        doc_embeds = np.array([doc.embedding for doc in document_to_search])
        doc_embeds = torch.as_tensor(doc_embeds, dtype=torch.float)
        if len(doc_embeds.shape) == 1 and doc_embeds.shape[0] == 1:
            doc_embeds = doc_embeds.unsqueeze(dim=0)
        elif len(doc_embeds.shape) == 1 and doc_embeds.shape[0] == 0:
            return []

        if self.similarity == "cosine":
            # cosine similarity is just a normed dot product
            query_emb_norm = torch.norm(query_emb, dim=1)
            query_emb = torch.div(query_emb, query_emb_norm)

            doc_embeds_norms = torch.norm(doc_embeds, dim=1)
            doc_embeds = torch.div(doc_embeds.T, doc_embeds_norms).T

        curr_pos = 0
        scores = []
        while curr_pos < len(doc_embeds):
            doc_embeds_slice = doc_embeds[curr_pos : curr_pos + self.scoring_batch_size]
            doc_embeds_slice = doc_embeds_slice.to(self.main_device)
            with torch.no_grad():
                slice_scores = torch.matmul(doc_embeds_slice, query_emb.T).cpu()
                slice_scores = slice_scores.squeeze(dim=1)
                slice_scores = slice_scores.numpy().tolist()

            scores.extend(slice_scores)
            curr_pos += self.scoring_batch_size

        return scores

    def get_scores_numpy(self, query_emb: np.ndarray, document_to_search: List[Document]) -> List[float]:
        """
        Calculate similarity scores between query embedding and a list of documents using numpy.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param document_to_search: List of documents to compare `query_emb` against.
        """
        if len(query_emb.shape) == 1:
            query_emb = np.expand_dims(query_emb, 0)

        doc_embeds = np.array([doc.embedding for doc in document_to_search])
        if len(doc_embeds.shape) == 1 and doc_embeds.shape[0] == 1:
            doc_embeds = doc_embeds.unsqueeze(dim=0)
        elif len(doc_embeds.shape) == 1 and doc_embeds.shape[0] == 0:
            return []

        if self.similarity == "cosine":
            # cosine similarity is just a normed dot product
            query_emb_norm = np.apply_along_axis(np.linalg.norm, 1, query_emb)
            query_emb_norm = np.expand_dims(query_emb_norm, 1)
            query_emb = np.divide(query_emb, query_emb_norm)

            doc_embeds_norms = np.apply_along_axis(np.linalg.norm, 1, doc_embeds)
            doc_embeds_norms = np.expand_dims(doc_embeds_norms, 1)
            doc_embeds = np.divide(doc_embeds, doc_embeds_norms)

        scores = np.dot(query_emb, doc_embeds.T)[0].tolist()

        return scores

    def get_scores(self, query_emb: np.ndarray, document_to_search: List[Document]) -> List[float]:
        if self.main_device.type == "cuda":
            scores = self.get_scores_torch(query_emb, document_to_search)
        else:
            scores = self.get_scores_numpy(query_emb, document_to_search)

        return scores

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in InMemoryDocStore
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
    ) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
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
                        Example:
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
                        Example:
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
        :param top_k: How many documents to return
        :param index: Index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :return:
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        index = index or self.index
        if return_embedding is None:
            return_embedding = self.return_embedding

        if query_emb is None:
            return []

        document_to_search = self.get_all_documents(index=index, filters=filters, return_embedding=True)
        scores = self.get_scores(query_emb, document_to_search)

        candidate_docs = []
        for doc, score in zip(document_to_search, scores):
            curr_meta = deepcopy(doc.meta)
            new_document = Document(id=doc.id, content=doc.content, meta=curr_meta, embedding=doc.embedding)
            new_document.embedding = doc.embedding if return_embedding is True else None

            new_document.embedding = doc.embedding if return_embedding is True else None
            if scale_score:
                score = self.scale_to_unit_interval(score, self.similarity)
            new_document.score = score
            candidate_docs.append(new_document)

        return sorted(candidate_docs, key=lambda x: x.score if x.score is not None else 0.0, reverse=True)[0:top_k]

    def update_embeddings(
        self,
        retriever: "BaseRetriever",
        index: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in InMemoryDocStore
        update_existing_embeddings: bool = True,
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
        :param filters: Narrow down the scope to documents that match the given filters.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.
                        Example:
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
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return: None
        """
        if index is None:
            index = self.index

        if not self.embedding_field:
            raise RuntimeError("Specify the arg embedding_field when initializing InMemoryDocumentStore()")

        # TODO Index embeddings every X batches to avoid OOM for huge document collections
        result = self._query(
            index=index, filters=filters, only_documents_without_embedding=not update_existing_embeddings
        )
        document_count = len(result)
        logger.info(f"Updating embeddings for {document_count} docs ...")
        batched_documents = get_batches_from_generator(result, batch_size)
        with tqdm(
            total=document_count, disable=not self.progress_bar, position=0, unit=" docs", desc="Updating Embedding"
        ) as progress_bar:
            for document_batch in batched_documents:
                embeddings = retriever.embed_documents(document_batch)  # type: ignore
                assert len(document_batch) == len(embeddings)

                if embeddings[0].shape[0] != self.embedding_dim:
                    raise RuntimeError(
                        f"Embedding dim. of model ({embeddings[0].shape[0]})"
                        f" doesn't match embedding dim. in DocumentStore ({self.embedding_dim})."
                        "Specify the arg `embedding_dim` when initializing InMemoryDocumentStore()"
                    )

                for doc, emb in zip(document_batch, embeddings):
                    self.indexes[index][doc.id].embedding = emb
                progress_bar.set_description_str("Documents Processed")
                progress_bar.update(batch_size)

    def get_document_count(
        self,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in InMemoryDocStore
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Return the number of documents in the document store.
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        documents = self._query(
            index=index, filters=filters, only_documents_without_embedding=only_documents_without_embedding
        )
        return len(documents)

    def get_embedding_count(self, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int:
        """
        Return the count of embeddings in the document store.
        """
        documents = self.get_all_documents(filters=filters, index=index)
        embedding_count = sum(doc.embedding is not None for doc in documents)
        return embedding_count

    def get_label_count(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int:
        """
        Return the number of labels in the document store.
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        index = index or self.label_index
        return len(self.indexes[index].items())

    def _query(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in InMemoryDocStore
        return_embedding: Optional[bool] = None,
        only_documents_without_embedding: bool = False,
    ):
        index = index or self.index
        documents = deepcopy(list(self.indexes[index].values()))
        documents = [d for d in documents if isinstance(d, Document)]

        if return_embedding is None:
            return_embedding = self.return_embedding
        if return_embedding is False:
            for doc in documents:
                doc.embedding = None

        if only_documents_without_embedding:
            documents = [doc for doc in documents if doc.embedding is None]
        if filters:
            parsed_filter = LogicalFilterClause.parse(filters)
            filtered_documents = list(filter(lambda doc: parsed_filter.evaluate(doc.meta), documents))
        else:
            filtered_documents = documents

        return filtered_documents

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in InMemoryDocStore
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Get all documents from the document store as a list.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Narrow down the scope to documents that match the given filters.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.
                        Example:
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
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in InMemoryDocStore
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        Get all documents from the document store. The methods returns a Python Generator that yields individual
        documents.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Narrow down the scope to documents that match the given filters.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.
                        Example:
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
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        result = self._query(index=index, filters=filters, return_embedding=return_embedding)
        yield from result

    def get_all_labels(
        self,
        index: str = None,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in InMemoryDocStore
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Label]:
        """
        Return all labels in the document store.
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        index = index or self.label_index

        if filters:
            result = []
            for label in self.indexes[index].values():
                label_dict = label.to_dict()
                is_hit = True
                for key, values in filters.items():
                    if label_dict[key] not in values:
                        is_hit = False
                        break
                if is_hit:
                    result.append(label)
        else:
            result = list(self.indexes[index].values())

        return result

    def delete_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in InMemoryDocStore
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the document from.
        :param filters: Narrow down the scope to documents that match the given filters.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.
                        Example:
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
        :return: None
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        logger.warning(
            """DEPRECATION WARNINGS: 
                1. delete_all_documents() method is deprecated, please use delete_documents method
                For more details, please refer to the issue: https://github.com/deepset-ai/haystack/issues/1045
                """
        )
        self.delete_documents(index, None, filters)

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in InMemoryDocStore
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param ids: Optional list of IDs to narrow down the documents to be deleted.
        :param filters: Narrow down the scope to documents that match the given filters.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.
                        Example:
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
        :return: None
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        index = index or self.index
        if not filters and not ids:
            self.indexes[index] = {}
            return
        docs_to_delete = self.get_all_documents(index=index, filters=filters)
        if ids:
            docs_to_delete = [doc for doc in docs_to_delete if doc.id in ids]
        for doc in docs_to_delete:
            del self.indexes[index][doc.id]

    def delete_index(self, index: str):
        """
        Delete an existing index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        if index in self.indexes:
            del self.indexes[index]
            logger.info(f"Index '{index}' deleted.")

    def delete_labels(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,  # TODO: Adapt type once we allow extended filters in InMemoryDocStore
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete labels in an index. All labels are deleted if no filters are passed.

        :param index: Index name to delete the labels from. If None, the
                      DocumentStore's default label index (self.label_index) will be used.
        :param ids: Optional list of IDs to narrow down the labels to be deleted.
        :param filters: Narrow down the scope to documents that match the given filters.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.
                        Example:
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
        :return: None
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")

        index = index or self.label_index
        if not filters and not ids:
            self.indexes[index] = {}
            return
        labels_to_delete = self.get_all_labels(index=index, filters=filters)
        if ids:
            labels_to_delete = [label for label in labels_to_delete if label.id in ids]
        for label in labels_to_delete:
            del self.indexes[index][label.id]
