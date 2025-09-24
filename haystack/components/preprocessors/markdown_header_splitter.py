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
    Split documents at ATX-style Markdown headers (#), with optional secondary splitting and header level inference.

    This component processes text documents by:
    - Splitting them into chunks at Markdown headers (e.g., '#', '##', etc.), preserving header hierarchy as metadata.
    - Optionally inferring and rewriting header levels for documents where header structure is ambiguous.
    - Optionally applying a secondary split (by word, passage, period, or line) to each chunk.
      This is done in haystack's DocumentSplitter.
    - Preserving and propagating metadata such as parent headers, page numbers, and split IDs.
    """

    def __init__(
        self,
        *,
        infer_header_levels: bool = False,
        page_break_character: str = "\f",
        secondary_split: Literal["none", "word", "passage", "period", "line"] = "none",
        split_length: int = 200,
        split_overlap: int = 0,
        split_threshold: int = 0,
    ):
        """
        Initialize the MarkdownHeaderSplitter.

        :param infer_header_levels: If True, attempts to infer and rewrite header levels based on content structure.
            Useful for documents where all headers use the same level. Defaults to False.
        :param page_break_character: Character used to identify page breaks. Defaults to form feed ("\f").
        :param secondary_split: Optional secondary split condition after header splitting.
            Options are "none", "word", "passage", "period", "line". Defaults to "none".
        :param split_length: The maximum number of units in each split when using secondary splitting. Defaults to 200.
        :param split_overlap: The number of overlapping units for each split when using secondary splitting.
            Defaults to 0.
        :param split_threshold: The minimum number of units per split when using secondary splitting. Defaults to 0.
        """
        self.infer_header_levels = infer_header_levels
        self.page_break_character = page_break_character
        self.secondary_split = secondary_split
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_threshold = split_threshold

        # initialize secondary_splitter only if needed
        if self.secondary_split != "none":
            self.secondary_splitter = DocumentSplitter(
                split_by=self.secondary_split,
                split_length=self.split_length,
                split_overlap=self.split_overlap,
                split_threshold=self.split_threshold,
            )

    def _infer_header_levels(self, text: str) -> str:
        """
        Infer and rewrite header levels in the markdown text.

        This function analyzes the document structure to infer proper header levels:
        - First header is always level 1
        - If there's content between headers, the next header stays at the same level
        - If there's no content between headers, the next header goes one level deeper
        - Header levels never exceed 6 (the maximum in markdown)

        This is useful for documents where all headers are at the same level, such as
        output from document conversion tools like docling.
        """
        logger.debug("Inferring and rewriting header levels")

        # find headers
        pattern = r"(?m)^(#{1,6}) (.+)$"
        matches = list(re.finditer(pattern, text))

        if not matches:
            logger.info("No headers found in document; skipping header level inference.")
            return text

        modified_text = text
        offset = 0  # track offset due to length changes in headers

        # track header structure
        current_level = 1
        header_stack = [1]  # always start with level 1

        for i, match in enumerate(matches):
            original_header = match.group(0)
            header_text = match.group(2).strip()

            # check if there's content between this header and the previous one
            has_content = False
            if i > 0:
                prev_end = matches[i - 1].end()
                current_start = match.start()
                content_between = text[prev_end:current_start].strip()
                has_content = bool(content_between)

            # first header is always level 1
            if i == 0:
                inferred_level = 1
            elif has_content:
                # stay at the same level if there's content
                inferred_level = current_level
            else:
                # go one level deeper if there's no content
                inferred_level = min(current_level + 1, 6)

            # update tracking variables
            current_level = inferred_level
            header_stack = header_stack[:inferred_level]
            while len(header_stack) < inferred_level:
                header_stack.append(1)

            # new header with inferred level
            new_prefix = "#" * inferred_level
            new_header = f"{new_prefix} {header_text}"

            # replace old header
            start_pos = match.start() + offset
            end_pos = match.end() + offset
            modified_text = modified_text[:start_pos] + new_header + modified_text[end_pos:]

            # update offset
            offset += len(new_header) - len(original_header)

        logger.info("Rewrote {num_headers} headers with inferred levels.", num_headers=len(matches))
        return modified_text

    def _split_text_by_markdown_headers(self, text: str) -> list[dict]:
        """Split text by ATX-style headers (#) and create chunks with appropriate metadata."""
        logger.debug("Splitting text by markdown headers")

        # find headers
        pattern = r"(?m)^(#{1,6}) (.+)$"
        matches = list(re.finditer(pattern, text))

        # return unsplit if no headers found
        if not matches:
            logger.info("No headers found in document; returning full document as single chunk.")
            return [{"content": text, "meta": {"header": None, "parentheaders": []}}]

        # process headers and build chunks
        chunks: list[dict] = []
        header_stack: list[Optional[str]] = [None] * 6
        active_parents: list[str] = []

        for i, match in enumerate(matches):
            # extract header info
            header_prefix = match.group(1)
            header_text = match.group(2).strip()
            level = len(header_prefix)

            # get content
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            # update header stack to track nesting
            header_stack[level - 1] = header_text
            for j in range(level, 6):
                header_stack[j] = None

            # skip splits w/o content
            if not content:
                # Add as parent for subsequent headers
                active_parents = [h for h in header_stack[: level - 1] if h is not None]
                active_parents.append(header_text)
                continue

            # get parent headers
            parentheaders = list(active_parents)

            logger.debug(
                "Creating chunk for header '{header_text}' at level {level}", header_text=header_text, level=level
            )

            chunks.append(
                {
                    "content": f"{header_prefix} {header_text}\n{content}",
                    "meta": {"header": header_text, "parentheaders": parentheaders},
                }
            )

            # reset active parents
            active_parents = [h for h in header_stack[: level - 1] if h is not None]

        logger.info("Split into {num_chunks} chunks by markdown headers.", num_chunks=len(chunks))
        return chunks

    def _apply_secondary_splitting(self, documents: list[Document]) -> list[Document]:
        """
        Apply secondary splitting while preserving header metadata and structure.

        Ensures page counting is maintained across splits.
        """
        if self.secondary_split == "none":
            return documents

        logger.info("Applying secondary splitting by {secondary_split}", secondary_split=self.secondary_split)
        result_docs = []

        for doc in documents:
            if doc.content is None:
                result_docs.append(doc)
                continue

            # extract header information
            header_match = re.search(r"(#{1,6}) (.+)(?:\n|$)", doc.content)
            content_for_splitting: str = doc.content
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
                    _, current_page = self._count_page_breaks_and_update(secondary_splits[i - 1].content, current_page)

                # set page number to meta
                split.meta["page_number"] = current_page

                # preserve header metadata
                for key in ["header", "parentheaders"]:
                    if key in doc.meta:
                        split.meta[key] = doc.meta[key]

                result_docs.append(split)

        # assign unique, sequential split_id to all final chunks
        for idx, doc in enumerate(result_docs):
            if doc.meta is None:
                doc.meta = {}
            doc.meta["split_id"] = idx

        logger.info("Secondary splitting complete. Final count: {final_count} documents.", final_count=len(result_docs))
        return result_docs

    def _flatten_dict(self, d: dict, prefix: str = "", target_dict: Optional[dict] = None) -> dict:
        """Flatten a nested dictionary, concatenating keys with underscores."""
        if target_dict is None:
            target_dict = {}
        for key, value in d.items():
            new_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten_dict(value, f"{new_key}_", target_dict)
            else:
                target_dict[new_key] = value
        return target_dict

    def _count_page_breaks_and_update(self, content: str, current_page: int) -> tuple[int, int]:
        """
        Count page breaks in content and return updated page count.

        :param content: Content to check for page breaks
        :param current_page: Current page number
        :return: Tuple of (page_breaks_count, new_current_page)
        """
        if not isinstance(content, str):
            return 0, current_page

        page_breaks = content.count(self.page_break_character)
        new_page_number = current_page + page_breaks

        if page_breaks > 0:
            logger.debug(
                "Found {page_breaks} page breaks, page number updated: {old} → {new}",
                page_breaks=page_breaks,
                old=current_page,
                new=new_page_number,
            )

        return page_breaks, new_page_number

    def _split_documents_by_markdown_headers(self, documents: list[Document]) -> list[Document]:
        """Split a list of documents by markdown headers, preserving metadata."""
        result_docs = []
        for doc in documents:
            logger.debug("Splitting document with id={doc_id}", doc_id=doc.id)
            if doc.content is None:
                continue
            splits = self._split_text_by_markdown_headers(doc.content)
            docs = []
            total_pages = self._calculate_total_pages(doc.content, doc.meta.get("total_pages", 0) if doc.meta else 0)

            current_page = doc.meta.get("page_number", 1) if doc.meta else 1
            logger.debug(
                "Starting page number: {current_page}, Total pages: {total_pages}",
                current_page=current_page,
                total_pages=total_pages,
            )
            for split in splits:
                meta = {}
                if doc.meta:
                    meta = self._flatten_dict(doc.meta)
                meta.update({"source_id": doc.id, "total_pages": total_pages, "page_number": current_page})
                _, current_page = self._count_page_breaks_and_update(split["content"], current_page)
                if split.get("meta"):
                    meta.update(self._flatten_dict(split.get("meta") or {}))
                docs.append(Document(content=split["content"], meta=meta))
            logger.debug(
                "Split into {num_docs} documents for id={doc_id}, final page: {current_page}",
                num_docs=len(docs),
                doc_id=doc.id,
                current_page=current_page,
            )
            result_docs.extend(docs)
        return result_docs

    def _calculate_total_pages(self, content: str, existing_total: int = 0) -> int:
        """Calculate total pages based on content and existing metadata."""
        if existing_total > 0:
            return existing_total

        if not isinstance(content, str):
            return 1

        return content.count(self.page_break_character) + 1

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], infer_header_levels: Optional[bool] = None) -> dict[str, list[Document]]:
        """
        Run the markdown header splitter with optional secondary splitting.

        :param documents: List of documents to split
        :param infer_header_levels: If True, attempts to infer and rewrite header levels before splitting.
            If None, uses the value from initialization.

        :returns: A dictionary with the following key:
            - `documents`: List of documents with the split texts. Each document includes:
            - A metadata field `source_id` to track the original document.
            - A metadata field `page_number` to track the original page number.
            - All other metadata copied from the original document.
        """
        # validate input documents
        for doc in documents:
            if not isinstance(doc.content, str):
                raise ValueError("MarkdownHeaderSplitter only works with text documents (str content).")

        infer_header_levels = infer_header_levels if infer_header_levels is not None else self.infer_header_levels

        processed_documents = []
        for doc in documents:
            # skip empty documents
            if not doc.content or not doc.content.strip():
                continue
            if infer_header_levels:
                content = self._infer_header_levels(doc.content)
                processed_documents.append(Document(content=content, meta=doc.meta, id=doc.id))
            else:
                processed_documents.append(doc)

        if not processed_documents:
            return {"documents": []}

        header_split_docs = self._split_documents_by_markdown_headers(processed_documents)
        logger.info("Header splitting produced {num_docs} documents", num_docs=len(header_split_docs))

        # secondary splitting if configured
        final_docs = (
            self._apply_secondary_splitting(header_split_docs) if self.secondary_split != "none" else header_split_docs
        )

        # assign split_id if not already done in secondary splitting
        if self.secondary_split == "none":
            for idx, doc in enumerate(final_docs):
                if doc.meta is None:
                    doc.meta = {}
                doc.meta["split_id"] = idx

        return {"documents": final_docs}
