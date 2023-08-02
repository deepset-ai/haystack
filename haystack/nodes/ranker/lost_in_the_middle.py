from typing import Optional, Union, List
import logging

from haystack.schema import Document
from haystack.nodes.ranker.base import BaseRanker

logger = logging.getLogger(__name__)


class LostInTheMiddleRanker(BaseRanker):
    """
    The LostInTheMiddleRanker implements a ranker that reorders documents based on the "lost in the middle" order.
    "Lost in the Middle: How Language Models Use Long Contexts" paper by Liu et al. aims to lay out paragraphs into LLM
    context so that the relevant paragraphs are at the beginning or end of the input context, while the least relevant
    information is in the middle of the context.

    See https://arxiv.org/abs/2307.03172 for more details.
    """

    def __init__(self, word_count_threshold: Optional[int] = None, top_k: Optional[int] = None):
        """
        Creates an instance of LostInTheMiddleRanker.

        If 'word_count_threshold' is specified, this ranker includes all documents up until the point where adding
        another document would exceed the 'word_count_threshold'. The last document that causes the threshold to
        be breached will be included in the resulting list of documents, but all subsequent documents will be
        discarded.

        :param word_count_threshold: The maximum total number of words across all documents selected by the ranker.
        :param top_k: The maximum number of documents to return.
        """
        super().__init__()
        if isinstance(word_count_threshold, int) and word_count_threshold <= 0:
            raise ValueError(
                f"Invalid value for word_count_threshold: {word_count_threshold}. "
                f"word_count_threshold must be a positive integer."
            )
        self.word_count_threshold = word_count_threshold
        self.top_k = top_k

    def reorder_documents(self, documents: List[Document]) -> List[Document]:
        """
        Ranks documents based on the "lost in the middle" order. Assumes that all documents are ordered by relevance.

        :param documents: List of Documents to merge.
        :return: Documents in the "lost in the middle" order.
        """

        # Return empty list if no documents are provided
        if not documents:
            return []

        # If there's only one document, return it as is
        if len(documents) == 1:
            return documents

        # Raise an error if any document is not textual
        if any(not doc.content_type == "text" for doc in documents):
            raise ValueError("Some provided documents are not textual; LostInTheMiddleRanker can process only text.")

        # Initialize word count and indices for the "lost in the middle" order
        word_count = 0
        document_index = list(range(len(documents)))
        lost_in_the_middle_indices = [0]

        # If word count threshold is set, calculate word count for the first document
        if self.word_count_threshold:
            word_count = len(documents[0].content.split())

            # If the first document already meets the word count threshold, return it
            if word_count >= self.word_count_threshold:
                return [documents[0]]

        # Start from the second document and create "lost in the middle" order
        for doc_idx in document_index[1:]:
            # Calculate the index at which the current document should be inserted
            insertion_index = len(lost_in_the_middle_indices) // 2 + len(lost_in_the_middle_indices) % 2

            # Insert the document index at the calculated position
            lost_in_the_middle_indices.insert(insertion_index, doc_idx)

            # If word count threshold is set, calculate the total word count
            if self.word_count_threshold:
                word_count += len(documents[doc_idx].content.split())

                # If the total word count meets the threshold, stop processing further documents
                if word_count >= self.word_count_threshold:
                    break

        # Return the documents in the "lost in the middle" order
        return [documents[idx] for idx in lost_in_the_middle_indices]

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """
        Reranks documents based on the "lost in the middle" order.

        :param query: The query to reorder documents for (ignored).
        :param documents: List of Documents to reorder.
        :param top_k: The number of documents to return.

        :return: The reordered documents.
        """
        top_k = top_k or self.top_k
        documents_to_reorder = documents[:top_k] if top_k else documents
        ranked_docs = self.reorder_documents(documents=documents_to_reorder)
        return ranked_docs

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        """
        Reranks batch of documents based on the "lost in the middle" order.

        :param queries: The queries to reorder documents for (ignored).
        :param documents: List of Documents to reorder.
        :param top_k: The number of documents to return.
        :param batch_size: The number of queries to process in one batch (ignored).

        :return: The reordered documents.
        """
        if len(documents) > 0 and isinstance(documents[0], Document):
            return self.predict(query="", documents=documents, top_k=top_k)  # type: ignore
        else:
            # Docs case 2: list of lists of Documents -> rerank each list of Documents
            results = []
            for cur_docs in documents:
                assert isinstance(cur_docs, list)
                results.append(self.predict(query="", documents=cur_docs, top_k=top_k))
            return results
