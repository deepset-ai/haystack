from typing import Optional, Union, List
import logging

from haystack.schema import Document
from haystack.nodes.ranker.base import BaseRanker

logger = logging.getLogger(__name__)


class LostInTheMiddleRanker(BaseRanker):
    """
    The LostInTheMiddleRanker implements a ranker that reorders documents based on the lost in the middle order.
    "Lost in the Middle: How Language Models Use Long Contexts" by Liu et al. aims to layout paragraphs into LLM
    context so that relevant paragraphs are at the beginning or end of the input context, while the least relevant
    information should be in the middle of a context.

    See https://arxiv.org/abs/2307.03172 for more details.
    """

    def __init__(self, word_count_threshold: Optional[int] = None, truncate_document: Optional[bool] = False):
        """
        Creates an instance of LostInTheMiddleRanker.

        If truncate_document is set to True, you must specify a word_count_threshold as well.

        :param word_count_threshold: The maximum number of words in all ordered documents.
        :param truncate_document: Whether to truncate the last document that overflows the word count threshold.
        """
        super().__init__()
        if truncate_document and not word_count_threshold:
            raise ValueError("If truncate_document is set to True, you must specify a word_count_threshold as well.")
        self.word_count_threshold = word_count_threshold
        self.truncate_document = truncate_document

    def reorder_documents(self, documents: List[Document]) -> List[Document]:
        """
        Orders documents based on the lost in the middle order.

        :param documents: List of Documents to merge.
        :return: Documents in the lost in the middle order.
        """
        if not documents:
            return []
        if len(documents) == 1:
            return documents

        if any(not doc.content_type == "text" for doc in documents):
            raise ValueError("Some provided documents are not textual; LostInTheMiddleRanker can process only text.")

        word_count = 0
        document_index = list(range(len(documents)))
        lost_in_the_middle_indices = [0]
        if self.word_count_threshold:
            word_count = len(documents[0].content.split())
            if word_count >= self.word_count_threshold:
                return [documents[0]]

        for doc_idx in document_index[1:]:
            insertion_index = len(lost_in_the_middle_indices) // 2 + len(lost_in_the_middle_indices) % 2
            lost_in_the_middle_indices.insert(insertion_index, doc_idx)
            if self.word_count_threshold:
                word_count += len(documents[doc_idx].content.split())
                # if threshold is specified, check if we have enough words in all selected documents
                # if yes, we can stop adding documents
                if word_count >= self.word_count_threshold:
                    if self.truncate_document:
                        # truncate the last document that overflows the word count threshold
                        last_docs_length = len(documents[doc_idx].content.split())
                        truncate_last_doc_length = last_docs_length - (word_count - self.word_count_threshold)
                        documents[doc_idx] = self._truncate(documents[doc_idx], truncate_last_doc_length)
                    break

        return [documents[idx] for idx in lost_in_the_middle_indices]

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """
        Reorders documents based on the lost in the middle order.

        :param query: The query to rerank documents for.
        :param documents: List of Documents to reorder.
        :param top_k: The number of documents to return.

        :return: The reranked documents.
        """
        ordered_docs = self.reorder_documents(documents=documents)
        return ordered_docs[:top_k] if top_k else ordered_docs

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        """
        Reorders batch of documents based on the lost in the middle order.

        :param queries: The queries to rerank documents for (ignored).
        :param documents: List of Documents to reorder.
        :param top_k: The number of documents to return.
        :param batch_size: The number of queries to process in one batch.

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

    def _truncate(self, document: Document, word_count_threshold: int) -> Document:
        """
        Shortens a document by cutting off the content after a specified number of words.

        :param document: Document to truncate.
        :param word_count_threshold: integer representing the maximum number of words
                                     allowed in the truncated document.

        :return: Document with truncated content.
        """
        document.content = " ".join(document.content.split()[:word_count_threshold])
        return document
