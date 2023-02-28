from typing import Any, Dict, List, Optional, Union, Generator

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import time
import logging
from copy import deepcopy
from collections import defaultdict
import re

import numpy as np
import torch
from tqdm.auto import tqdm
import rank_bm25
import pandas as pd

from haystack.schema import Document, FilterType, Label
from haystack.errors import DuplicateDocumentError, DocumentStoreError
from haystack.document_stores import KeywordDocumentStore
from haystack.document_stores.base import get_batches_from_generator
from haystack.modeling.utils import initialize_device_settings
from haystack.document_stores.filter_utils import LogicalFilterClause
from haystack.nodes.retriever import DenseRetriever

logger = logging.getLogger(__name__)


class InMemoryDocumentStore(KeywordDocumentStore):
    # pylint: disable=R0904
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
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_bm25: bool = False,
        bm25_tokenization_regex: str = r"(?u)\b\w\w+\b",
        bm25_algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25Okapi",
        bm25_parameters: Optional[Dict] = None,
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
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        :param use_bm25: Whether to build a sparse representation of documents based on BM25.
                         `use_bm25=True` is required to connect `BM25Retriever` to this Document Store.
        :param bm25_tokenization_regex: The regular expression to use for tokenization of the text.
        :param bm25_algorithm: The specific BM25 implementation to adopt.
                               Parameter options : ( 'BM25Okapi', 'BM25L', 'BM25Plus')
        :param bm25_parameters: Parameters for BM25 implementation in a dictionary format.
                                For example: {'k1':1.5, 'b':0.75, 'epsilon':0.25}
                                You can learn more about these parameters by visiting https://github.com/dorianbrown/rank_bm25
                                By default, no parameters are set.
        """
        if bm25_parameters is None:
            bm25_parameters = {}
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
        self.use_bm25 = use_bm25
        self.bm25_tokenization_regex = bm25_tokenization_regex
        self.bm25_algorithm = bm25_algorithm
        self.bm25_parameters = bm25_parameters
        self.bm25: Dict[str, rank_bm25.BM25] = {}

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=self.use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                self.devices[0],
            )

        self.main_device = self.devices[0]

    @property
    def bm25_tokenization_regex(self):
        return self._tokenizer

    @bm25_tokenization_regex.setter
    def bm25_tokenization_regex(self, regex_string: str):
        self._tokenizer = re.compile(regex_string).findall

    @property
    def bm25_algorithm(self):
        return self._bm25_class

    @bm25_algorithm.setter
    def bm25_algorithm(self, algorithm: str):
        self._bm25_class = getattr(rank_bm25, algorithm)

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
                           For documents as dictionaries, the format is {"content": "<the-actual-text>"}.
                           Optionally: Include meta data via {"content": "<the-actual-text>",
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
        modified_documents = 0
        for document in documents_objects:
            if document.id in self.indexes[index]:
                if duplicate_documents == "fail":
                    raise DuplicateDocumentError(
                        f"Document with id '{document.id} already " f"exists in index '{index}'"
                    )
                if duplicate_documents == "skip":
                    logger.warning(
                        "Duplicate Documents: Document with id '%s' already exists in index '%s'", document.id, index
                    )
                    continue
            self.indexes[index][document.id] = document
            modified_documents += 1

        if self.use_bm25 is True and modified_documents > 0:
            self.update_bm25(index=index)

    def update_bm25(self, index: Optional[str] = None):
        """
        Updates the BM25 sparse representation in the the document store.

        :param index: Index name for which the BM25 representation is to be updated. If set to None, the default self.index is used.
        """
        index = index or self.index

        all_documents = self.get_all_documents(index=index)
        textual_documents = []
        for doc in all_documents:
            if doc.content_type == "text":
                textual_documents.append(doc.content.lower())
            elif doc.content_type == "table":
                if isinstance(doc.content, pd.DataFrame):
                    textual_documents.append(doc.content.astype(str).to_csv(index=False).lower())
                else:
                    raise DocumentStoreError("Documents of type 'table' need to have a pd.DataFrame as content field")
        if len(textual_documents) < len(all_documents):
            logger.warning(
                "Some documents in %s index are non-textual."
                " They will be written to the index, but the corresponding BM25 representations will not be generated.",
                index,
            )

        tokenized_corpus = [
            self.bm25_tokenization_regex(doc)
            for doc in tqdm(textual_documents, unit=" docs", desc="Updating BM25 representation...")
        ]
        self.bm25[index] = self.bm25_algorithm(tokenized_corpus, **self.bm25_parameters)

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
                "Duplicate Label IDs: Inserting a Label whose id already exists in this document store."
                " This will overwrite the old Label. Please make sure Label.id is a unique identifier of"
                " the answer annotation and not the question."
                " Problematic ids: %s",
                ",".join(duplicate_ids),
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

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Fetch documents by specifying a list of text id strings.
        """
        if headers:
            raise NotImplementedError("InMemoryDocumentStore does not support headers.")
        if batch_size:
            logger.warning(
                "InMemoryDocumentStore does not support batching in `get_documents_by_id` method. This parameter is ignored."
            )
        index = index or self.index
        documents = [self.indexes[index][id] for id in ids]
        return documents

    def _get_scores_torch(self, query_emb: np.ndarray, documents_to_search: List[Document]) -> List[float]:
        """
        Calculate similarity scores between query embedding and a list of documents using torch.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param documents_to_search: List of documents to compare `query_emb` against.
        """
        query_emb_tensor = torch.tensor(query_emb, dtype=torch.float).to(self.main_device)
        if query_emb_tensor.ndim == 1:
            query_emb_tensor = query_emb_tensor.unsqueeze(dim=0)

        doc_embeds = np.array([doc.embedding for doc in documents_to_search])
        doc_embeds_tensor = torch.as_tensor(doc_embeds, dtype=torch.float)
        if doc_embeds_tensor.ndim == 1:
            # if there are no embeddings, return an empty list
            if doc_embeds_tensor.shape[0] == 0:
                return []
            doc_embeds_tensor = doc_embeds_tensor.unsqueeze(dim=0)

        if self.similarity == "cosine":
            # cosine similarity is just a normed dot product
            query_emb_norm = torch.norm(query_emb_tensor, dim=1)
            query_emb_tensor = torch.div(query_emb_tensor, query_emb_norm)

            doc_embeds_norms = torch.norm(doc_embeds_tensor, dim=1)
            doc_embeds_tensor = torch.div(doc_embeds_tensor.T, doc_embeds_norms).T

        curr_pos = 0
        scores: List[float] = []
        while curr_pos < len(doc_embeds_tensor):
            doc_embeds_slice = doc_embeds_tensor[curr_pos : curr_pos + self.scoring_batch_size]
            doc_embeds_slice = doc_embeds_slice.to(self.main_device)
            with torch.inference_mode():
                slice_scores = torch.matmul(doc_embeds_slice, query_emb_tensor.T).cpu()
                slice_scores = slice_scores.squeeze(dim=1)
                slice_scores = slice_scores.numpy().tolist()

            scores.extend(slice_scores)
            curr_pos += self.scoring_batch_size

        return scores

    def _get_scores_numpy(self, query_emb: np.ndarray, documents_to_search: List[Document]) -> List[float]:
        """
        Calculate similarity scores between query embedding and a list of documents using numpy.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param documents_to_search: List of documents to compare `query_emb` against.
        """
        if query_emb.ndim == 1:
            query_emb = np.expand_dims(a=query_emb, axis=0)

        doc_embeds = np.array([doc.embedding for doc in documents_to_search])
        if doc_embeds.ndim == 1:
            # if there are no embeddings, return an empty list
            if doc_embeds.shape[0] == 0:
                return []
            doc_embeds = np.expand_dims(a=doc_embeds, axis=0)

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

    def _get_scores(self, query_emb: np.ndarray, documents_to_search: List[Document]) -> List[float]:
        if self.main_device.type == "cuda":
            scores = self._get_scores_torch(query_emb, documents_to_search)
        else:
            scores = self._get_scores_numpy(query_emb, documents_to_search)

        return scores

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

        documents = self.get_all_documents(index=index, filters=filters, return_embedding=True)
        documents_with_embeddings = [doc for doc in documents if doc.embedding is not None]
        if len(documents) != len(documents_with_embeddings):
            logger.warning(
                "Skipping some of your documents that don't have embeddings. "
                "To generate embeddings, run the document store's update_embeddings() method."
            )
        scores = self._get_scores(query_emb, documents_with_embeddings)

        candidate_docs = []
        for doc, score in zip(documents_with_embeddings, scores):
            curr_meta = deepcopy(doc.meta)
            new_document = Document(
                id=doc.id, content=doc.content, content_type=doc.content_type, meta=curr_meta, embedding=doc.embedding
            )
            new_document.embedding = doc.embedding if return_embedding is True else None

            new_document.embedding = doc.embedding if return_embedding is True else None
            if scale_score:
                score = self.scale_to_unit_interval(score, self.similarity)
            new_document.score = score
            candidate_docs.append(new_document)

        return sorted(candidate_docs, key=lambda x: x.score if x.score is not None else 0.0, reverse=True)[0:top_k]

    def update_embeddings(
        self,
        retriever: DenseRetriever,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
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
        logger.info("Updating embeddings for %s docs ...", len(result) if logger.level > logging.DEBUG else 0)
        batched_documents = get_batches_from_generator(result, batch_size)
        with tqdm(
            total=len(result), disable=not self.progress_bar, position=0, unit=" docs", desc="Updating Embedding"
        ) as progress_bar:
            for document_batch in batched_documents:
                embeddings = retriever.embed_documents(document_batch)
                self._validate_embeddings_shape(
                    embeddings=embeddings, num_documents=len(document_batch), embedding_dim=self.embedding_dim
                )

                for doc, emb in zip(document_batch, embeddings):
                    self.indexes[index][doc.id].embedding = emb
                progress_bar.set_description_str("Documents Processed")
                progress_bar.update(batch_size)

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
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

    def update_document_meta(self, id: str, meta: Dict[str, Any], index: Optional[str] = None):
        """
        Update the metadata dictionary of a document by specifying its string id.

        :param id: The ID of the Document whose metadata is being updated.
        :param meta: A dictionary with key-value pairs that should be added / changed for the provided Document ID.
        :param index: Name of the index the Document is located at.
        """
        if index is None:
            index = self.index
        for key, value in meta.items():
            self.indexes[index][id].meta[key] = value

    def get_embedding_count(self, filters: Optional[FilterType] = None, index: Optional[str] = None) -> int:
        """
        Return the count of embeddings in the document store.
        """
        documents = self.get_all_documents_generator(filters=filters, index=index, return_embedding=True)
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
        filters: Optional[FilterType] = None,
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
        filters: Optional[FilterType] = None,
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
        filters: Optional[FilterType] = None,
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
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
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
                for key, value_or_values in filters.items():
                    if isinstance(value_or_values, list):
                        if label_dict[key] not in value_or_values:
                            is_hit = False
                            break
                    else:
                        if label_dict[key] != value_or_values:
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
        filters: Optional[FilterType] = None,
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
        filters: Optional[FilterType] = None,
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
            if index in self.bm25:
                self.bm25[index] = {}
            return
        docs_to_delete = self.get_all_documents(index=index, filters=filters)
        if ids:
            docs_to_delete = [doc for doc in docs_to_delete if doc.id in ids]
        for doc in docs_to_delete:
            del self.indexes[index][doc.id]
        if self.use_bm25 is True and len(docs_to_delete) > 0:
            self.update_bm25(index=index)

    def delete_index(self, index: str):
        """
        Delete an existing index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        if index in self.indexes:
            del self.indexes[index]
            logger.info("Index '%s' deleted.", index)

        if index in self.bm25:
            del self.bm25[index]

    def delete_labels(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
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

    def query(
        self,
        query: Optional[str],
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        all_terms_must_match: bool = False,
        scale_score: bool = False,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query as defined by the BM25 algorithm.
        :param query: The query.
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents.
        """

        if headers:
            logger.warning("InMemoryDocumentStore does not support headers. This parameter is ignored.")
        if custom_query:
            logger.warning("InMemoryDocumentStore does not support custom_query. This parameter is ignored.")
        if all_terms_must_match is True:
            logger.warning("InMemoryDocumentStore does not support all_terms_must_match. This parameter is ignored.")
        if filters:
            logger.warning(
                "InMemoryDocumentStore does not support filters for BM25 retrieval. This parameter is ignored."
            )
        if scale_score is True:
            logger.warning(
                "InMemoryDocumentStore does not support scale_score for BM25 retrieval. This parameter is ignored."
            )

        index = index or self.index
        if index not in self.bm25:
            raise DocumentStoreError(
                f"No BM25 representation found for the index: {index}. The Document store should be initialized with use_bm25=True"
            )

        if query is None:
            return []

        tokenized_query = self.bm25_tokenization_regex(query.lower())
        docs_scores = self.bm25[index].get_scores(tokenized_query)
        top_docs_positions = np.argsort(docs_scores)[::-1][:top_k]

        textual_docs_list = [doc for doc in self.indexes[index].values() if doc.content_type in ["text", "table"]]
        top_docs = []
        for i in top_docs_positions:
            doc = textual_docs_list[i]
            doc.score = docs_scores[i]
            top_docs.append(doc)

        return top_docs

    def query_batch(
        self,
        queries: List[str],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        all_terms_must_match: bool = False,
        scale_score: bool = False,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the provided queries as defined by keyword matching algorithms like BM25.
        This method lets you find relevant documents for list of query strings (output: List of Lists of Documents).
        :param query: The query.
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents.
        """

        if headers:
            logger.warning("InMemoryDocumentStore does not support headers. This parameter is ignored.")
        if custom_query:
            logger.warning("InMemoryDocumentStore does not support custom_query. This parameter is ignored.")
        if all_terms_must_match is True:
            logger.warning("InMemoryDocumentStore does not support all_terms_must_match. This parameter is ignored.")
        if filters:
            logger.warning(
                "InMemoryDocumentStore does not support filters for BM25 retrieval. This parameter is ignored."
            )
        if scale_score is True:
            logger.warning(
                "InMemoryDocumentStore does not support scale_score for BM25 retrieval. This parameter is ignored."
            )

        index = index or self.index
        if index not in self.bm25:
            raise DocumentStoreError(
                f"No BM25 representation found for the index: {index}. The Document store should be initialized with use_bm25=True"
            )

        result_documents = []
        for query in queries:
            result_documents.append(self.query(query=query, top_k=top_k, index=index))

        return result_documents
