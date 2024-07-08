from typing import Any, Dict, List

from haystack import Document, component, default_to_dict
from haystack.document_stores.in_memory import InMemoryDocumentStore


@component
class SentenceWindowRetrieval:
    """
    A component that retrieves surrounding documents of a given document from the document store.

    This component is designed to work together with one of the existing retrievers, e.g. BM25Retriever,
    EmbeddingRetriever. One of these retrievers can be used to retrieve documents based on a query and then use this
    component to get the surrounding documents of the retrieved documents.
    """

    def __init__(self, document_store: InMemoryDocumentStore, window_size: int = 3):
        self.window_size = window_size
        self.document_store = document_store

    @staticmethod
    def merge_documents(documents: List[Document]):
        """
        Merge a list of document chunks text into a single string.

        This functions concatenates the textual content of a List of Documents into a single string, eliminating any
        overlapping content.
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

    @component.output_types(context_windows=List[str])
    def run(self, retrieved_documents: List[Document]):
        """
        Based on the `source_id` and on the `doc.meta['split_id']` get surrounding documents from the document store.

        Implements the logic behind the sentence-window technique, retrieving the surrounding documents of a given
        document from the document store.

        :param retrieved_documents: List of retrieved documents from the previous retriever.
        :type retrieved_documents: List[Document]
        :return:
            List of strings representing the context windows of the retrieved documents.
        """
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
            context_windows.append(self.merge_documents(context_docs))

        return {"context_windows": context_windows}
