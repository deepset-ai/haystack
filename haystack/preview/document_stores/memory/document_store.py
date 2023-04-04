from typing import Literal, Any, Dict, List, Optional, Iterable

import logging

from haystack.preview.dataclasses import Document
from haystack.preview.document_stores._utils import DuplicateDocumentError, MissingDocumentError


logger = logging.getLogger(__name__)
DuplicatePolicy = Literal["skip", "overwrite", "fail"]


class MemoryDocumentStore:
    """
    Stores data in-memory. It's ephemeral and cannot be saved to disk.
    """

    def __init__(self):
        """
        Initializes the store.

        :param use_bm25: whether to support bm25 retrieval. It might slow down this document store on high volumes.
        :param bm25_parameters: parameters for rank_bm25.
        :param devices: which devices to use for embedding retrieval. Leave empty to support embedding retrieval on CPU only.
        """
        self.storage = {}

    def count_documents(self) -> int:
        """
        Returns the number of how many documents match the given filters.
        """
        return len(self.storage.keys())

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        :param filters: the filters to apply to the document list.
        """
        # TODO apply filters
        return list(self.storage.values())

    def write_documents(self, documents: List[Document], duplicates: DuplicatePolicy = "fail") -> None:
        """
        Writes (or overwrites) documents into the store.

        :param documents: a list of documents.
        :param duplicates: documents with the same ID count as duplicates. When duplicates are met,
            the store can:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :raises DuplicateError: Exception trigger on duplicate document
        :return: None
        """
        if (
            not isinstance(documents, Iterable)
            or isinstance(documents, str)
            or any(not isinstance(doc, Document) for doc in documents)
        ):
            raise ValueError("Please provide a list of Documents.")

        for document in documents:
            if document.id in self.storage.keys():
                if duplicates == "fail":
                    raise DuplicateDocumentError(f"ID '{document.id}' already exists.")
                if duplicates == "skip":
                    logger.warning("ID '%s' already exists", document.id)
            self.storage[document.id] = document

    def delete_documents(self, object_ids: List[str]) -> None:
        """
        Deletes all object_ids from the given pool.
        Fails with `MissingDocumentError` if no object with this id is present in the store.

        :param object_ids: the object_ids to delete
        """
        for object_id in object_ids:
            if not object_id in self.storage.keys():
                raise MissingDocumentError(f"ID '{object_id}' not found, cannot delete it.")
            del self.storage[object_id]
