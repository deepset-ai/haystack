# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List

from haystack import DeserializationError, Document, component, default_from_dict, default_to_dict
from haystack.core.serialization import import_class_by_name
from haystack.document_stores.types import DocumentStore


@component
class SentenceWindowRetrieval:
    """
    A component that retrieves surrounding documents of a given document from the document store.

    This component is designed to work together with one of the existing retrievers, e.g. BM25Retriever,
    EmbeddingRetriever. One of these retrievers can be used to retrieve documents based on a query and then use this
    component to get the surrounding documents of the retrieved documents.
    """

    def __init__(self, document_store: DocumentStore, window_size: int = 3):
        """
        Creates a new SentenceWindowRetrieval component.

        :param document_store: The document store to use for retrieving the surrounding documents.
        :param window_size: The number of surrounding documents to retrieve.
        """
        if window_size < 1:
            raise ValueError("The window_size parameter must be greater than 0.")

        self.window_size = window_size
        self.document_store = document_store

    @staticmethod
    def merge_documents_text(documents: List[Document]) -> str:
        """
        Merge a list of document text into a single string.

        This functions concatenates the textual content of a list of documents into a single string, eliminating any
        overlapping content.

        :param documents: List of Documents to merge.
        """
        sorted_docs = sorted(documents, key=lambda doc: doc.meta["split_idx_start"])
        merged_text = ""
        last_idx_end = 0
        for doc in sorted_docs:
            start = doc.meta["split_idx_start"]  # start of the current content

            # if the start of the current content is before the end of the last appended content, adjust it
            start = max(start, last_idx_end)

            # append the non-overlapping part to the merged text
            merged_text = merged_text.strip()
            merged_text += doc.content[start - doc.meta["split_idx_start"] :]  # type: ignore

            # update the last end index
            last_idx_end = doc.meta["split_idx_start"] + len(doc.content)  # type: ignore

        return merged_text

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        docstore = self.document_store.to_dict()
        return default_to_dict(self, document_store=docstore, window_size=self.window_size)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceWindowRetrieval":
        """
        Deserializes the component from a dictionary.

        :returns:
            Deserialized component.
        """
        init_params = data.get("init_parameters", {})

        if "document_store" not in init_params:
            raise DeserializationError("Missing 'document_store' in serialization data")
        if "type" not in init_params["document_store"]:
            raise DeserializationError("Missing 'type' in document store's serialization data")

        # deserialize the document store
        doc_store_data = data["init_parameters"]["document_store"]
        try:
            doc_store_class = import_class_by_name(doc_store_data["type"])
        except ImportError as e:
            raise DeserializationError(f"Class '{doc_store_data['type']}' not correctly imported") from e

        data["init_parameters"]["document_store"] = default_from_dict(doc_store_class, doc_store_data)

        # deserialize the component
        return default_from_dict(cls, data)

    @component.output_types(context_windows=List[str])
    def run(self, retrieved_documents: List[Document]):
        """
        Based on the `source_id` and on the `doc.meta['split_id']` get surrounding documents from the document store.

        Implements the logic behind the sentence-window technique, retrieving the surrounding documents of a given
        document from the document store.

        :param retrieved_documents: List of retrieved documents from the previous retriever.
        :type retrieved_documents: List[Document]
        :returns:
            A dictionary with the following keys:
            - `context_windows`:  List of strings representing the context windows of the retrieved documents.
        """

        if not all("split_id" in doc.meta for doc in retrieved_documents):
            raise ValueError("The retrieved documents must have 'split_id' in the metadata.")

        if not all("source_id" in doc.meta for doc in retrieved_documents):
            raise ValueError("The retrieved documents must have 'source_id' in the metadata.")

        context_windows = []
        for doc in retrieved_documents:
            source_id = doc.meta["source_id"]
            split_id = doc.meta["split_id"]
            min_before = min(list(range(split_id - 1, split_id - self.window_size - 1, -1)))
            max_after = max(list(range(split_id + 1, split_id + self.window_size + 1, 1)))
            context_docs = self.document_store.filter_documents(
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "source_id", "operator": "==", "value": source_id},
                        {"field": "split_id", "operator": ">=", "value": min_before},
                        {"field": "split_id", "operator": "<=", "value": max_after},
                    ],
                }
            )
            context_windows.append(self.merge_documents_text(context_docs))

        return {"context_windows": context_windows}
