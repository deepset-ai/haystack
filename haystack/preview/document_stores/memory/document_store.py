from typing import Literal, Any, Dict, List, Optional, Union, Iterable

import logging

import numpy as np

from haystack.v2.data import Document, Query, TextDocument

from haystack.v2.stores.memory.store import StoreInMemory
from haystack.v2.stores.memory._bm25 import BM25Representation, BM25RepresentationMissing
from haystack.v2.stores.memory._scores import get_scores_numpy, get_scores_torch, scale_to_unit_interval


logger = logging.getLogger(__name__)

try:
    import torch
except ImportError as e:
    logger.debug("torch not found: DocumentStoreInMemory won't be able to search by embedding with a local model.")


class DocumentStoreInMemory:
    def __init__(
        self,
        use_bm25: bool = True,
        bm25_tokenization_regex: str = r"(?u)\b\w\w+\b",
        bm25_algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25Okapi",
        bm25_parameters: dict = {},
        use_gpu: bool = True,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
    ):
        self.document_store = StoreInMemory()

        # For BM25 retrieval
        self.use_bm25 = use_bm25
        self.bm25_algorithm = (bm25_algorithm,)
        self.bm25_parameters = (bm25_parameters,)
        self.bm25_tokenization_regex = bm25_tokenization_regex
        self.bm25 = {}
        if use_bm25:
            self.bm25 = {
                "documents": BM25Representation(
                    bm25_algorithm=bm25_algorithm,
                    bm25_parameters=bm25_parameters,
                    bm25_tokenization_regex=bm25_tokenization_regex,
                )
            }

        # For embedding retrieval
        # self.device = None
        # init_devices, _ = initialize_device_settings(
        #     devices=devices, use_cuda=use_gpu, multi_gpu=False
        # )
        # if init_devices:
        #     if len(init_devices) > 1:
        #         logger.warning(
        #             "Multiple devices are not supported in %s inference, using the first device %s.",
        #             self.__class__.__name__,
        #             init_devices[0],
        #         )
        #     self.device = init_devices[0]

    def has_document(self, id: str) -> bool:
        """
        Checks if this ID exists in the document store.

        :param id: the id to find in the document store.
        """
        return self.document_store.has_item(id=id)

    def get_document(self, id: str) -> Optional[Document]:
        """
        Finds a document by ID in the document store.

        :param id: the id of the document to get.
        """
        return Document.from_dict(self.document_store.get_item(id=id))

    def count_documents(self, filters: Dict[str, Any]):
        """
        Returns the number of how many documents match the given filters.
        Pass filters={} to count all documents

        :param filters: the filters to apply to the documents list.
        """
        return self.document_store.count_items(filters=filters)

    def get_document_ids(self, filters: Dict[str, Any]) -> Iterable[str]:
        """
        Returns only the ID of the documents that match the filters provided.

        :param filters: the filters to apply to the documents list.
        """
        return self.document_store.get_ids(filters=filters)

    def get_documents(self, filters: Dict[str, Any]) -> Iterable[Document]:
        """
        Returns the documents that match the filters provided.

        :param filters: the filters to apply to the documents list.
        """
        for doc in self.document_store.get_items(filters=filters):
            yield Document.from_dict(dictionary=doc)

    def write_documents(
        self, documents: Iterable[Document], duplicates: Literal["skip", "overwrite", "fail"] = "overwrite"
    ) -> None:
        """
        Writes documents into the store.

        :param documents: a list of Haystack Document objects.
        :param duplicates: Documents with the same ID count as duplicates. When duplicates are met,
            Haystack can choose to:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return: None
        """
        self.document_store.write_items(items=(doc.to_dict() for doc in documents), duplicates=duplicates)
        if self.bm25:
            self.bm25.update_bm25(self.get_documents(filters={}))

    def delete_documents(self, ids: List[str], fail_on_missing_item: bool = False) -> None:
        """
        Deletes all documents with the given ids.

        :param ids: the ids to delete
        :param fail_on_missing_item: fail if the id is not found, log ignore otherwise
        """
        self.document_store.delete_items(ids=ids, fail_on_missing_item=fail_on_missing_item)
