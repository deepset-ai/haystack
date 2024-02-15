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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, word_count_threshold=self.word_count_threshold, top_k=self.top_k)

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """
        Reranks documents based on the "lost in the middle" order.
        Returns a list of Documents reordered based on the input query.
        :param query: The query to reorder documents for (ignored).
        :param documents: List of Documents to reorder.
        :param top_k: The number of documents to return.

        :return: The reordered documents.
        """
        if not documents:
            return {"documents": []}

        if top_k is None:
            top_k = self.top_k

        documents_to_reorder = documents[:top_k] if top_k else documents
        ranked_docs = self.reorder_documents(documents=documents_to_reorder)
        return {"documents": ranked_docs}
