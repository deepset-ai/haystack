# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Literal, Optional

from haystack import Document, component, logging
from haystack.components.preprocessors import DocumentSplitter

logger = logging.getLogger(__name__)


@component
class MarkdownHeaderSplitter:
    """
    Split documents at ATX-style Markdown headers (#), with optional secondary splitting.

    This component processes text documents by:
    - Splitting them into chunks at Markdown headers (e.g., '#', '##', etc.), preserving header hierarchy as metadata.
    - Optionally applying a secondary split (by word, passage, period, or line) to each chunk
      (using haystack's DocumentSplitter).
    - Preserving and propagating metadata such as parent headers, page numbers, and split IDs.
    """

    def __init__(
        self,
        *,
        page_break_character: str = "\f",
        keep_headers: bool = True,
        secondary_split: Optional[Literal["word", "passage", "period", "line"]] = None,
        split_length: int = 200,
        split_overlap: int = 0,
        split_threshold: int = 0,
        skip_empty_documents: bool = True,
    ):
        """
        Initialize the MarkdownHeaderSplitter.

        :param page_break_character: Character used to identify page breaks. Defaults to form feed ("\f").
        :param keep_headers: If True, headers are kept in the content. If False, headers are moved to metadata.
            Defaults to True.
        :param secondary_split: Optional secondary split condition after header splitting.
            Options are None, "word", "passage", "period", "line". Defaults to None.
        :param split_length: The maximum number of units in each split when using secondary splitting. Defaults to 200.
        :param split_overlap: The number of overlapping units for each split when using secondary splitting.
            Defaults to 0.
        :param split_threshold: The minimum number of units per split when using secondary splitting. Defaults to 0.
        :param skip_empty_documents: Choose whether to skip documents with empty content. Default is True.
            Set to False when downstream components in the Pipeline (like LLMDocumentContentExtractor) can extract text
            from non-textual documents.
        """
        self.page_break_character = page_break_character
        self.secondary_split = secondary_split
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_threshold = split_threshold
        self.skip_empty_documents = skip_empty_documents
        self.keep_headers = keep_headers
        self._header_pattern = re.compile(r"(?m)^(#{1,6}) (.+)$")  # ATX-style .md-headers
        self._is_warmed_up = False

        # initialize secondary_splitter only if needed
        if self.secondary_split:
            self.secondary_splitter = DocumentSplitter(
                split_by=self.secondary_split,
                split_length=self.split_length,
                split_overlap=self.split_overlap,
                split_threshold=self.split_threshold,
            )

    def warm_up(self):
        """
        Warm up the MarkdownHeaderSplitter.
        """
        if self.secondary_split and not self._is_warmed_up:
            self.secondary_splitter.warm_up()
            self._is_warmed_up = True

    def _split_text_by_markdown_headers(self, text: str, doc_id: str) -> list[dict]:
        """Split text by ATX-style headers (#) and create chunks with appropriate metadata."""
        logger.debug("Splitting text by markdown headers")

        # find headers
        matches = list(re.finditer(self._header_pattern, text))

        # return unsplit if no headers found
        if not matches:
            logger.info(
                "No headers found in document {doc_id}; returning full document as single chunk.", doc_id=doc_id
            )
            return [{"content": text, "meta": {}}]

        # process headers and build chunks
        chunks: list[dict] = []
        header_stack: list[Optional[str]] = [None] * 6
        active_parents: list[str] = []  # track active parent headers
        pending_headers: list[str] = []  # store empty headers to prepend to next content
        has_content = False  # flag to track if any header has content

        for i, match in enumerate(matches):
            # extract header info
            header_prefix = match.group(1)
            header_text = match.group(2).strip()
            level = len(header_prefix)

            # get content
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end]

            # update header stack to track nesting
            header_stack[level - 1] = header_text
            for j in range(level, 6):
                header_stack[j] = None

            # prepare header_line if keep_headers
            header_line = f"{header_prefix} {header_text}"

            # skip splits w/o content
            if not content.strip():
                # add as parent for subsequent headers
                active_parents = [h for h in header_stack[: level - 1] if h is not None]
                active_parents.append(header_text)
                if self.keep_headers:
                    pending_headers.append(header_line)
                continue

            has_content = True  # at least one header has content
            parent_headers = list(active_parents)

            logger.debug(
                "Creating chunk for header '{header_text}' at level {level}", header_text=header_text, level=level
            )

            if self.keep_headers:
                # add pending & current header to content
                chunk_content = ""
                if pending_headers:
                    chunk_content += "\n".join(pending_headers) + "\n"
                chunk_content += f"{header_line}{content}"
                chunks.append(
                    {
                        "content": chunk_content,
                        "meta": {} if self.keep_headers else {"header": header_text, "parent_headers": parent_headers},
                    }
                )
                pending_headers = []  # reset pending headers
            else:
                chunks.append({"content": content, "meta": {"header": header_text, "parent_headers": parent_headers}})

            # reset active parents
            active_parents = [h for h in header_stack[: level - 1] if h is not None]

        # return doc unchunked if no headers have content
        if not has_content:
            logger.info(
                "Document {doc_id} contains only headers with no content; returning original document.", doc_id=doc_id
            )
            return [{"content": text, "meta": {}}]

        return chunks

    def _apply_secondary_splitting(self, documents: list[Document]) -> list[Document]:
        """
        Apply secondary splitting while preserving header metadata and structure.

        Ensures page counting is maintained across splits.
        """
        result_docs = []

        for doc in documents:
            if doc.content is None:
                result_docs.append(doc)
                continue

            content_for_splitting: str = doc.content

            if not self.keep_headers:  # skip header extraction if keep_headers
                # extract header information
                header_match = re.search(self._header_pattern, doc.content)
                if header_match:
                    content_for_splitting = doc.content[header_match.end() :]

            if not content_for_splitting or not content_for_splitting.strip():  # skip empty content
                result_docs.append(doc)
                continue

            # track page from meta
            current_page = doc.meta.get("page_number", 1)

            secondary_splits = self.secondary_splitter.run(
                documents=[Document(content=content_for_splitting, meta=doc.meta)]
            )["documents"]

            # split processing
            for i, split in enumerate(secondary_splits):
                # calculate page number for this split
                if i > 0 and secondary_splits[i - 1].content:
                    current_page = self._update_page_number_with_breaks(secondary_splits[i - 1].content, current_page)

                # set page number to meta
                split.meta["page_number"] = current_page

                # preserve header metadata if we're not keeping headers in content
                if not self.keep_headers:
                    for key in ["header", "parent_headers"]:
                        if key in doc.meta:
                            split.meta[key] = doc.meta[key]

                result_docs.append(split)

        logger.debug(
            "Secondary splitting complete. Final count: {final_count} documents.", final_count=len(result_docs)
        )
        return result_docs

    def _update_page_number_with_breaks(self, content: str, current_page: int) -> int:
        """
        Update page number based on page breaks in content.

        :param content: Content to check for page breaks
        :param current_page: Current page number
        :return: New current page number
        """
        if not isinstance(content, str):
            return current_page

        page_breaks = content.count(self.page_break_character)
        new_page_number = current_page + page_breaks

        if page_breaks > 0:
            logger.debug(
                "Found {page_breaks} page breaks, page number updated: {old} → {new}",
                page_breaks=page_breaks,
                old=current_page,
                new=new_page_number,
            )

        return new_page_number

    def _split_documents_by_markdown_headers(self, documents: list[Document]) -> list[Document]:
        """Split a list of documents by markdown headers, preserving metadata."""

        result_docs = []
        for doc in documents:
            logger.debug("Splitting document with id={doc_id}", doc_id=doc.id)
            # mypy: doc.content is Optional[str], so we must check for None before passing to splitting method
            if doc.content is None:
                continue
            splits = self._split_text_by_markdown_headers(doc.content, doc.id)
            docs = []

            current_page = doc.meta.get("page_number", 1) if doc.meta else 1
            total_pages = doc.content.count(self.page_break_character) + 1
            logger.debug(
                "Processing page number: {current_page} out of {total_pages}",
                current_page=current_page,
                total_pages=total_pages,
            )
            for split in splits:
                meta = {}
                if doc.meta:
                    meta = doc.meta.copy()
                meta.update({"source_id": doc.id, "page_number": current_page})
                if split.get("meta"):
                    meta.update(split["meta"])
                current_page = self._update_page_number_with_breaks(split["content"], current_page)
                docs.append(Document(content=split["content"], meta=meta))
            logger.debug(
                "Split into {num_docs} documents for id={doc_id}, final page: {current_page}",
                num_docs=len(docs),
                doc_id=doc.id,
                current_page=current_page,
            )
            result_docs.extend(docs)
        return result_docs

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Run the markdown header splitter with optional secondary splitting.

        :param documents: List of documents to split

        :returns: A dictionary with the following key:
            - `documents`: List of documents with the split texts. Each document includes:
                - A metadata field `source_id` to track the original document.
                - A metadata field `page_number` to track the original page number.
                - A metadata field `split_id` to uniquely identify each split chunk.
                - All other metadata copied from the original document.
        """
        # validate input documents
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    (
                        "MarkdownHeaderSplitter only works with text documents but content for document ID"
                        f" {doc.id} is None."
                    )
                )
            if not isinstance(doc.content, str):
                raise ValueError("MarkdownHeaderSplitter only works with text documents (str content).")

        processed_documents = []
        for doc in documents:
            # handle empty documents
            if not doc.content or not doc.content.strip():
                if self.skip_empty_documents:
                    logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                    continue
                # keep empty documents
                processed_documents.append(doc)
                logger.warning(
                    "Document ID {doc_id} has an empty content. Keeping this document as per configuration.",
                    doc_id=doc.id,
                )
                continue

            processed_documents.append(doc)

        if not processed_documents:
            return {"documents": []}

        header_split_docs = self._split_documents_by_markdown_headers(processed_documents)

        # secondary splitting if configured
        final_docs = self._apply_secondary_splitting(header_split_docs) if self.secondary_split else header_split_docs

        # assign split_id to all output documents
        for idx, doc in enumerate(final_docs):
            doc.meta["split_id"] = idx

        return {"documents": final_docs}
