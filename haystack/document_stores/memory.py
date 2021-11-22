from typing import TYPE_CHECKING, Dict, List, Optional, Union, Generator

if TYPE_CHECKING:
    from haystack.nodes.retriever import BaseRetriever

import time
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from scipy.spatial.distance import cosine
from tqdm import tqdm

from haystack.schema import Document, Label
from haystack.errors import DuplicateDocumentError
from haystack.document_stores import BaseDocumentStore
from haystack.document_stores.base import get_batches_from_generator


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
        duplicate_documents: str = 'overwrite',
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
        """
        # save init parameters to enable export of component config as YAML
        self.set_config(
                index=index, label_index=label_index, embedding_field=embedding_field, embedding_dim=embedding_dim,
                return_embedding=return_embedding, similarity=similarity, progress_bar=progress_bar,
                duplicate_documents=duplicate_documents,
        )

        self.indexes: Dict[str, Dict] = defaultdict(dict)
        self.index: str = index
        self.label_index: str = label_index
        self.embedding_field = embedding_field
        self.embedding_dim = embedding_dim
        self.return_embedding = return_embedding
        self.similarity = similarity
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,  # type: ignore
                        duplicate_documents: Optional[str] = None):
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
        index = index or self.index
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert duplicate_documents in self.duplicate_documents_options, \
            f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        field_map = self._create_document_field_map()
        documents = deepcopy(documents)
        documents_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in
                             documents]
        documents_objects = self._drop_duplicate_documents(documents=documents_objects)
        for document in documents_objects:
            if document.id in self.indexes[index]:
                if duplicate_documents == "fail":
                    raise DuplicateDocumentError(f"Document with id '{document.id} already "
                                                 f"exists in index '{index}'")
                elif duplicate_documents == "skip":
                    logger.warning(f"Duplicate Documents: Document with id '{document.id} already exists in index "
                                   f"'{index}'")
                    continue
            self.indexes[index][document.id] = document

    def _create_document_field_map(self):
        return {
            self.embedding_field: "embedding",
        }

    def write_labels(self, labels: Union[List[dict], List[Label]], index: Optional[str] = None):
        """
        Write annotation labels into document store.
        """
        index = index or self.label_index
        label_objects = [Label.from_dict(l) if isinstance(l, dict) else l for l in labels]

        duplicate_ids: list = [label.id for label in self._get_duplicate_labels(label_objects, index=index)]
        if len(duplicate_ids) > 0:
            logger.warning(f"Duplicate Label IDs: Inserting a Label whose id already exists in this document store."
                           f" This will overwrite the old Label. Please make sure Label.id is a unique identifier of"
                           f" the answer annotation and not the question."
                           f" Problematic ids: {','.join(duplicate_ids)}")

        for label in label_objects:
            # create timestamps if not available yet
            if not label.created_at:
                label.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
            if not label.updated_at:
                label.updated_at = label.created_at
            self.indexes[index][label.id] = label

    def get_document_by_id(self, id: str, index: Optional[str] = None) -> Optional[Document]:
        """
        Fetch a document by specifying its text id string.
        """
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
        
    def query_by_embedding(self,
                           query_emb: np.ndarray,
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
                content=doc.content,
                meta=curr_meta,
                embedding=doc.embedding
            )
            new_document.embedding = doc.embedding if return_embedding is True else None

            if self.similarity == "dot_product":
                score = np.dot(query_emb, doc.embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc.embedding)
                )
            elif self.similarity == "cosine":
                # cosine similarity score = 1 - cosine distance
                score = 1 - cosine(query_emb, doc.embedding)
            new_document.score = (score + 1) / 2
            candidate_docs.append(new_document)

        return sorted(candidate_docs, key=lambda x: x.score if x.score is not None else 0.0, reverse=True)[0:top_k]

    def update_embeddings(
        self,
        retriever: 'BaseRetriever',
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
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
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
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
        with tqdm(total=document_count, disable=not self.progress_bar, position=0, unit=" docs",
                  desc="Updating Embedding") as progress_bar:
            for document_batch in batched_documents:
                embeddings = retriever.embed_documents(document_batch)  # type: ignore
                assert len(document_batch) == len(embeddings)

                if embeddings[0].shape[0] != self.embedding_dim:
                    raise RuntimeError(f"Embedding dim. of model ({embeddings[0].shape[0]})"
                                       f" doesn't match embedding dim. in DocumentStore ({self.embedding_dim})."
                                       "Specify the arg `embedding_dim` when initializing InMemoryDocumentStore()")

                for doc, emb in zip(document_batch, embeddings):
                    self.indexes[index][doc.id].embedding = emb
                progress_bar.set_description_str("Documents Processed")
                progress_bar.update(batch_size)

    def get_document_count(self, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int:
        """
        Return the number of documents in the document store.
        """
        documents = self.get_all_documents(index=index, filters=filters)
        return len(documents)

    def get_embedding_count(self, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int:
        """
        Return the count of embeddings in the document store.
        """
        documents = self.get_all_documents(filters=filters, index=index)
        embedding_count = sum(doc.embedding is not None for doc in documents)
        return embedding_count

    def get_label_count(self, index: Optional[str] = None) -> int:
        """
        Return the number of labels in the document store.
        """
        index = index or self.label_index
        return len(self.indexes[index].items())

    def _query(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
        only_documents_without_embedding: bool = False,
        batch_size: int = 10_000,
    ):
        index = index or self.index
        documents = deepcopy(list(self.indexes[index].values()))
        documents = [d for d in documents if isinstance(d, Document)]

        filtered_documents = []

        if return_embedding is None:
            return_embedding = self.return_embedding
        if return_embedding is False:
            for doc in documents:
                doc.embedding = None

        if only_documents_without_embedding:
            documents = [doc for doc in documents if doc.embedding is None]
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

        return filtered_documents

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
    ) -> List[Document]:
        """
        Get all documents from the document store as a list.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        """
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
        result = self._query(
            index=index,
            filters=filters,
            return_embedding=return_embedding,
            batch_size=batch_size
        )
        yield from result

    def get_all_labels(self, index: str = None, filters: Optional[Dict[str, List[str]]] = None) -> List[Label]:
        """
        Return all labels in the document store.
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
        logger.warning(
                """DEPRECATION WARNINGS: 
                1. delete_all_documents() method is deprecated, please use delete_documents method
                For more details, please refer to the issue: https://github.com/deepset-ai/haystack/issues/1045
                """
        )
        self.delete_documents(index, None, filters)

    def delete_documents(self, index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.

        :param index: Index name to delete the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param ids: Optional list of IDs to narrow down the documents to be deleted.
        :param filters: Optional filters to narrow down the documents to be deleted.
            Example filters: {"name": ["some", "more"], "category": ["only_one"]}.
            If filters are provided along with a list of IDs, this method deletes the
            intersection of the two query results (documents that match the filters and
            have their ID in the list).

        :return: None
        """
        index = index or self.index
        if not filters and not ids:
            self.indexes[index] = {}
            return
        docs_to_delete = self.get_all_documents(index=index, filters=filters)
        if ids:
            docs_to_delete = [doc for doc in docs_to_delete if doc.id in ids]
        for doc in docs_to_delete:
            del self.indexes[index][doc.id]

    def delete_labels(self, index: Optional[str] = None, ids: Optional[List[str]] = None, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete labels in an index. All labels are deleted if no filters are passed.

        :param index: Index name to delete the labels from. If None, the
                      DocumentStore's default label index (self.label_index) will be used.
        :param ids: Optional list of IDs to narrow down the labels to be deleted.
        :param filters: Optional filters to narrow down the labels to be deleted.
                        Example filters: {"id": ["9a196e41-f7b5-45b4-bd19-5feb7501c159", "9a196e41-f7b5-45b4-bd19-5feb7501c159"]} or {"query": ["question2"]}
        :return: None
        """
        index = index or self.label_index
        if not filters and not ids:
            self.indexes[index] = {}
            return
        labels_to_delete = self.get_all_labels(index=index, filters=filters)
        if ids:
            labels_to_delete = [label for label in labels_to_delete if label.id in ids]
        for label in labels_to_delete:
            del self.indexes[index][label.id]

