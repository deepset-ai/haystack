from typing import List

from haystack import Document, component
from haystack.dataclasses.document_hierarchical import HierarchicalDocument
from haystack.document_stores.types import DocumentStore


@component
class AutoMergingRetriever:
    """
    A retriever which returns parent documents of the matched leaf documents, based on a threshold.

    The AutoMergingRetriever assumes you have a hierarchical tree structure of documents, where the leaf nodes
    are indexed in a document store. During retrieval, if the number of matched leaf documents below the same parent is
    above a certain threshold, the retriever will return the parent document instead of the individual leaf documents.

    The rational is, given that a paragraph is split into multiple sentences represented as leaf documents, and if for
    a given query, multiple sentences are matched, the retriever will return the whole paragraph instead of the
    individual sentences, since the whole paragraph might be more informative than the individual sentences alone.

    # https://www.youtube.com/watch?v=oDzWsynpOyI
    # https://pbs.twimg.com/media/F7ONuajWMAAvuWh?format=jpg&name=4096x4096
    """

    def __init__(self, document_store: DocumentStore, threshold: float = 0.9):
        self.document_store = document_store
        self.threshold = threshold

    @component.output_types(documents=List[Document])
    def run(self, matched_leaf_documents: List[HierarchicalDocument]):
        """
        Run the AutoMergingRetriever.
        """

        # find the parent documents for the matched leaf documents
        # parent_ids = [doc.parent_id for doc in matched_leaf_documents]
