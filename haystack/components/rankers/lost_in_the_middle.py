# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from typing import Literal

from haystack import Document, component
from haystack.lazy_imports import LazyImport
from haystack.utils.misc import _deduplicate_documents

with LazyImport("Run 'pip install tiktoken'") as tiktoken_imports:
    import tiktoken


@component
class LostInTheMiddleRanker:
    """
    A LostInTheMiddle Ranker.

    Ranks documents based on the 'lost in the middle' order so that the most relevant documents are either at the
    beginning or end, while the least relevant are in the middle.

    LostInTheMiddleRanker assumes that some prior component in the pipeline has already ranked documents by relevance
    and requires no query as input but only documents. It is typically used as the last component before building a
    prompt for an LLM to prepare the input context for the LLM.

    Lost in the Middle ranking lays out document contents into LLM context so that the most relevant contents are at
    the beginning or end of the input context, while the least relevant is in the middle of the context. See the
    paper ["Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172) for more
    details.

    Usage example:
    ```python
    from haystack.components.rankers import LostInTheMiddleRanker
    from haystack import Document

    ranker = LostInTheMiddleRanker()
    docs = [Document(content="Paris"), Document(content="Berlin"), Document(content="Madrid")]
    result = ranker.run(documents=docs)
    for doc in result["documents"]:
        print(doc.content)
    ```
    """

    def __init__(
        self,
        word_count_threshold: int | None = None,
        top_k: int | None = None,
        *,
        count_mode: Literal["word", "char", "token"] = "word",
        tokenizer_encoding: str = "o200k_base",
    ) -> None:
        """
        Initialize the LostInTheMiddleRanker.

        If 'word_count_threshold' is specified, this ranker includes all documents up until the point where adding
        another document would exceed the 'word_count_threshold'. The last document that causes the threshold to
        be breached will be included in the resulting list of documents, but all subsequent documents will be
        discarded.

        :param word_count_threshold: The maximum total count across all documents selected by the ranker. The count is
            measured in the unit configured by `count_mode`.
        :param top_k: The maximum number of documents to return.
        :param count_mode: The unit used for threshold counting. It can be either "word", "char", or "token".
            If "token" is selected, the text is counted using the tiktoken tokenizer.
        :param tokenizer_encoding: The tiktoken encoding to use when `count_mode` is "token".
        """
        if isinstance(word_count_threshold, int) and word_count_threshold <= 0:
            raise ValueError(
                f"Invalid value for word_count_threshold: {word_count_threshold}. word_count_threshold must be > 0."
            )
        if isinstance(top_k, int) and top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        if count_mode not in ["word", "char", "token"]:
            raise ValueError(
                f"Invalid value for count_mode: {count_mode}. count_mode must be one of: 'word', 'char', 'token'."
            )

        self.word_count_threshold = word_count_threshold
        self.top_k = top_k
        self.count_mode = count_mode
        self.tokenizer_encoding = tokenizer_encoding
        self.tiktoken_tokenizer: "tiktoken.Encoding" | None = None

    def warm_up(self) -> None:
        """
        Initialize the tokenizer when `count_mode` is "token".
        """
        if self.count_mode == "token" and self.tiktoken_tokenizer is None:
            tiktoken_imports.check()
            self.tiktoken_tokenizer = tiktoken.get_encoding(self.tokenizer_encoding)

    @component.output_types(documents=list[Document])
    def run(
        self, documents: list[Document], top_k: int | None = None, word_count_threshold: int | None = None
    ) -> dict[str, list[Document]]:
        """
        Reranks documents based on the "lost in the middle" order.

        Before ranking, documents are deduplicated by their id, retaining only the document with the highest score
        if a score is present.

        :param documents: List of Documents to reorder.
        :param top_k: The maximum number of documents to return.
        :param word_count_threshold: The maximum total count across all documents selected by the ranker. The count is
            measured in the unit configured by `count_mode`.
        :returns:
            A dictionary with the following keys:
            - `documents`: Reranked list of Documents

        :raises ValueError:
            If any of the documents is not textual.
        """
        if isinstance(word_count_threshold, int) and word_count_threshold <= 0:
            raise ValueError(
                f"Invalid value for word_count_threshold: {word_count_threshold}. word_count_threshold must be > 0."
            )
        if isinstance(top_k, int) and top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k
        word_count_threshold = self.word_count_threshold if word_count_threshold is None else word_count_threshold

        deduplicated_documents = _deduplicate_documents(documents)
        documents_to_reorder = deduplicated_documents[:top_k] if top_k else deduplicated_documents

        # If there's only one document, return it as is
        if len(documents_to_reorder) == 1:
            return {"documents": documents_to_reorder}

        # Raise an error if any document is not textual
        if any(not doc.content_type == "text" for doc in documents_to_reorder):
            raise ValueError("Some provided documents are not textual; LostInTheMiddleRanker can process only text.")

        # Initialize threshold count and indices for the "lost in the middle" order
        count = 0
        document_index = list(range(len(documents_to_reorder)))
        lost_in_the_middle_indices = [0]

        # If threshold is set and the first document has content, calculate count for the first document.
        first_document_content = documents_to_reorder[0].content
        if word_count_threshold and first_document_content:
            count = self._count_text_units(first_document_content)

            # If the first document already meets the threshold, return it.
            if count >= word_count_threshold:
                return {"documents": [documents_to_reorder[0]]}

        # Start from the second document and create "lost in the middle" order
        for doc_idx in document_index[1:]:
            # Calculate the index at which the current document should be inserted
            insertion_index = len(lost_in_the_middle_indices) // 2 + len(lost_in_the_middle_indices) % 2

            # Insert the document index at the calculated position
            lost_in_the_middle_indices.insert(insertion_index, doc_idx)

            # If threshold is set and the document has content, calculate the total count.
            document_content = documents_to_reorder[doc_idx].content
            if word_count_threshold and document_content:
                count += self._count_text_units(document_content)

                # If the total count meets the threshold, stop processing further documents.
                if count >= word_count_threshold:
                    break

        # Documents in the "lost in the middle" order
        ranked_docs = [documents_to_reorder[idx] for idx in lost_in_the_middle_indices]
        return {"documents": ranked_docs}

    def _count_text_units(self, text: str) -> int:
        """
        Count text according to the configured count mode.
        """
        if self.count_mode == "word":
            return len(text.split())
        if self.count_mode == "char":
            return len(text)

        tokenizer = self.tiktoken_tokenizer
        if tokenizer is None:
            self.warm_up()
            tokenizer = self.tiktoken_tokenizer
        if tokenizer is None:
            raise RuntimeError("Tokenizer was not initialized.")
        return len(tokenizer.encode(text))
