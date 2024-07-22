from haystack import component
from haystack.document_stores.types import DocumentStore


@component
class AutoMergingRetriever:
    """
    Auto-merging retrieval aims to combine (or merge) information from multiple sources or segments of text.

    This approach is particularly useful when no single document or segment fully answers the query but rather the
    answer lies in combining information from multiple sources.

    It allows smaller chunks to be merged into bigger parent chunks.


    It does this via the following steps:

    - Define a hierarchy of smaller chunks linked to parent chunks.
    - If the set of smaller chunks linking to a parent chunk exceeds some threshold (say, cosine similarity),
      then “merge” smaller chunks into the bigger parent chunk.

    The method will finally retrieve the parent chunk for better context.
    """

    def __init__(self, document_store: DocumentStore, threshold: float = 0.9):
        self.document_store = document_store
        self.threshold = threshold

    def run(self):
        """
        Run the AutoMergingRetriever.

        :return:
        :rtype:
        """
        pass
