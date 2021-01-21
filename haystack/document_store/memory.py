from copy import deepcopy
from typing import Dict, List, Optional, Union, Generator
from uuid import uuid4
from collections import defaultdict

from haystack.document_store.base import BaseDocumentStore
from haystack import Document, Label
from haystack.retriever.base import BaseRetriever

from scipy.spatial.distance import cosine

import logging

logger = logging.getLogger(__name__)


class InMemoryDocumentStore(BaseDocumentStore):
    """
        In-memory document store
    """

    def __init__(self, embedding_field: Optional[str] = "embedding", return_embedding: bool = False, similarity="dot_product"):
        """
        :param embedding_field: Name of field containing an embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
        :param return_embedding: To return document embedding
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default sine it is
                   more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence BERT model.
        """
        self.indexes: Dict[str, Dict] = defaultdict(dict)
        self.index: str = "document"
        self.label_index: str = "label"
        self.embedding_field: str = embedding_field if embedding_field is not None else "embedding"
        self.embedding_dim: int = 768
        self.return_embedding: bool = return_embedding
        self.similarity: str = similarity

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None):
        """
        Indexes documents for later queries.


       :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta": {"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: write documents to a custom namespace. For instance, documents for evaluation can be indexed in a
                      separate index than the documents for search.
        :return: None
        """
        index = index or self.index

        field_map = self._create_document_field_map()
        documents_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]

        for document in documents_objects:
            self.indexes[index][document.id] = document

    def _create_document_field_map(self):
        return {
            self.embedding_field: "embedding",
        }

    def write_labels(self, labels: Union[List[dict], List[Label]], index: Optional[str] = None):
        """Write annotation labels into document store."""
        index = index or self.label_index
        label_objects = [Label.from_dict(l) if isinstance(l, dict) else l for l in labels]

        for label in label_objects:
            label_id = str(uuid4())
            self.indexes[index][label_id] = label

    def get_document_by_id(self, id: str, index: Optional[str] = None) -> Optional[Document]:
        """Fetch a document by specifying its text id string"""
        index = index or self.index
        documents = self.get_documents_by_id([id], index=index)
        if documents:
            return documents[0]
        else:
            return None

    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None) -> List[Document]:
        """Fetch documents by specifying a list of text id strings"""
        index = index or self.index
        documents = [self.indexes[index][id] for id in ids]
        return documents

    def query_by_embedding(self,
                           query_emb: List[float],
                           filters: Optional[Dict[str, List[str]]] = None,
                           top_k: int = 10,
                           index: Optional[str] = None,
                           return_embedding: Optional[bool] = None) -> List[Document]:

        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param filters: Optional filters to narrow down the search space.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param top_k: How many documents to return
        :param index: Index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :return:
        """

        from numpy import dot
        from numpy.linalg import norm

        index = index or self.index
        if return_embedding is None:
            return_embedding = self.return_embedding

        if query_emb is None:
            return []

        document_to_search = self.get_all_documents(index=index, filters=filters, return_embedding=True)
        candidate_docs = []
        for doc in document_to_search:
            curr_meta = deepcopy(doc.meta)
            new_document = Document(
                id=doc.id,
                text=doc.text,
                meta=curr_meta,
                embedding=doc.embedding
            )
            new_document.embedding = doc.embedding if return_embedding is True else None

            if self.similarity == "dot_product":
                score = dot(query_emb, doc.embedding) / (
                    norm(query_emb) * norm(doc.embedding)
                )
            elif self.similarity == "cosine":
                # cosine similarity score = 1 - cosine distance
                score = 1 - cosine(query_emb, doc.embedding)
            new_document.score = score
            new_document.probability = (score + 1) / 2
            candidate_docs.append(new_document)

        return sorted(candidate_docs, key=lambda x: x.score if x.score is not None else 0.0, reverse=True)[0:top_k]

    def update_embeddings(self, retriever: BaseRetriever, index: Optional[str] = None):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).

        :param retriever: Retriever
        :param index: Index name to update
        :return: None
        """
        if index is None:
            index = self.index

        if not self.embedding_field:
            raise RuntimeError("Specify the arg embedding_field when initializing InMemoryDocumentStore()")

        # TODO Index embeddings every X batches to avoid OOM for huge document collections
        docs = self.get_all_documents(index)
        logger.info(f"Updating embeddings for {len(docs)} docs ...")
        embeddings = retriever.embed_passages(docs)  # type: ignore
        assert len(docs) == len(embeddings)

        if embeddings[0].shape[0] != self.embedding_dim:
            raise RuntimeError(f"Embedding dim. of model ({embeddings[0].shape[0]})"
                               f" doesn't match embedding dim. in DocumentStore ({self.embedding_dim})."
                               "Specify the arg `embedding_dim` when initializing InMemoryDocumentStore()")

        for doc, emb in zip(docs, embeddings):
            self.indexes[index][doc.id].embedding = emb

    def get_document_count(self, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int:
        """
        Return the number of documents in the document store.
        """
        documents = self.get_all_documents(index=index, filters=filters)
        return len(documents)

    def get_label_count(self, index: Optional[str] = None) -> int:
        """
        Return the number of labels in the document store
        """
        index = index or self.label_index
        return len(self.indexes[index].items())

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
    ) -> List[Document]:
        result = self.get_all_documents_generator(index=index, filters=filters, return_embedding=return_embedding)
        documents = list(result)
        return documents
      
    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
    ) -> Generator[Document, None, None]:
        """
        Get all documents from the document store. The methods returns a Python Generator that yields individual
        documents.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        """
        index = index or self.index
        documents = deepcopy(list(self.indexes[index].values()))

        filtered_documents = []

        if return_embedding is None:
            return_embedding = self.return_embedding
        if return_embedding is False:
            for doc in documents:
                doc.embedding = None

        if filters:
            for doc in documents:
                is_hit = True
                for key, values in filters.items():
                    if doc.meta.get(key):
                        if doc.meta[key] not in values:
                            is_hit = False
                    else:
                        is_hit = False
                if is_hit:
                    filtered_documents.append(doc)
        else:
            filtered_documents = documents

        yield from filtered_documents

    def get_all_labels(self, index: str = None, filters: Optional[Dict[str, List[str]]] = None) -> List[Label]:
        """
        Return all labels in the document store
        """
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

    def delete_all_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the document from.
        :param filters: Optional filters to narrow down the documents to be deleted.
        :return: None
        """

        if filters:
            raise NotImplementedError("Delete by filters is not implemented for InMemoryDocumentStore.")
        index = index or self.index
        self.indexes[index] = {}
