from typing import Any, Dict, List, Optional

from haystack import Document, component, default_to_dict


@component
class LostInTheMiddleRanker:
    """
    The LostInTheMiddleRanker implements a ranker that reorders documents based on the "lost in the middle" order.
    "Lost in the Middle: How Language Models Use Long Contexts" paper by Liu et al. aims to lay out paragraphs into LLM
    context so that the relevant paragraphs are at the beginning or end of the input context, while the least relevant
    information is in the middle of the context.

    See https://arxiv.org/abs/2307.03172 for more details.
    """

    def __init__(self, word_count_threshold: Optional[int] = None, top_k: Optional[int] = None):
        """
        If 'word_count_threshold' is specified, this ranker includes all documents up until the point where adding
        another document would exceed the 'word_count_threshold'. The last document that causes the threshold to
        be breached will be included in the resulting list of documents, but all subsequent documents will be
        discarded.

        :param word_count_threshold: The maximum total number of words across all documents selected by the ranker.
        :param top_k: The maximum number of documents to return.
        """
        if isinstance(word_count_threshold, int) and word_count_threshold <= 0:
            raise ValueError(
                f"Invalid value for word_count_threshold: {word_count_threshold}. " f"word_count_threshold must be > 0."
            )
        if isinstance(top_k, int) and top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        self.word_count_threshold = word_count_threshold
        self.top_k = top_k

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize object to a dictionary.
        """
        return default_to_dict(self, word_count_threshold=self.word_count_threshold, top_k=self.top_k)

    def run(
        self, documents: List[Document], top_k: Optional[int] = None, word_count_threshold: Optional[int] = None
    ) -> Dict[str, List[Document]]:
        """
        Reranks documents based on the "lost in the middle" order.
        Returns a list of Documents reordered based on the input query.
        :param documents: List of Documents to reorder.
        :param top_k: The number of documents to return.
        :param word_count_threshold: The maximum total number of words across all documents selected by the ranker.

        :return: The reordered documents.
        """
        if isinstance(word_count_threshold, int) and word_count_threshold <= 0:
            raise ValueError(
                f"Invalid value for word_count_threshold: {word_count_threshold}. " f"word_count_threshold must be > 0."
            )
        if isinstance(top_k, int) and top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k
        word_count_threshold = word_count_threshold or self.word_count_threshold

        documents_to_reorder = documents[:top_k] if top_k else documents

        # If there's only one document, return it as is
        if len(documents_to_reorder) == 1:
            return {"documents": documents_to_reorder}

        # Raise an error if any document is not textual
        if any(not doc.content_type == "text" for doc in documents_to_reorder):
            raise ValueError("Some provided documents are not textual; LostInTheMiddleRanker can process only text.")

        # Initialize word count and indices for the "lost in the middle" order
        word_count = 0
        document_index = list(range(len(documents_to_reorder)))
        lost_in_the_middle_indices = [0]

        # If word count threshold is set and the first document has content, calculate word count for the first document
        if word_count_threshold and documents_to_reorder[0].content:
            word_count = len(documents_to_reorder[0].content.split())

            # If the first document already meets the word count threshold, return it
            if word_count >= word_count_threshold:
                return {"documents": [documents_to_reorder[0]]}

        # Start from the second document and create "lost in the middle" order
        for doc_idx in document_index[1:]:
            # Calculate the index at which the current document should be inserted
            insertion_index = len(lost_in_the_middle_indices) // 2 + len(lost_in_the_middle_indices) % 2

            # Insert the document index at the calculated position
            lost_in_the_middle_indices.insert(insertion_index, doc_idx)

            # If word count threshold is set and the document has content, calculate the total word count
            if word_count_threshold and documents_to_reorder[doc_idx].content:
                word_count += len(documents_to_reorder[doc_idx].content.split())  # type: ignore[union-attr]

                # If the total word count meets the threshold, stop processing further documents
                if word_count >= word_count_threshold:
                    break

        # Documents in the "lost in the middle" order
        ranked_docs = [documents_to_reorder[idx] for idx in lost_in_the_middle_indices]
        return {"documents": ranked_docs}
