import re
from typing import Callable, Literal, Optional

from haystack import Document, component, logging
from haystack.components.preprocessors import DocumentSplitter

logger = logging.getLogger(__name__)


class _CustomDocumentSplitter(DocumentSplitter):
    """
    Internal helper class that extends DocumentSplitter to support splitting functions.

    This class handles splitting functions that return dictionaries with 'content' and 'meta'
    keys instead of just strings. For internal use only within the MarkdownHeaderSplitter.
    """

    def __init__(
        self,
        split_by: str = "function",
        splitting_function: Optional[Callable] = None,
        page_break_character: str = "\\f",
    ):
        """
        Initialize the _CustomDocumentSplitter.

        :param split_by: The method to split by. Must be "function" for custom splitting functions.
        :param splitting_function: A custom function that takes a string and returns a list of dicts
            with 'content' and optional 'meta'.
        :param page_break_character: Character used to identify page breaks.
            Defaults to form feed ("\\f").
        """
        super().__init__(split_by=split_by, splitting_function=splitting_function)
        self.page_break_character = page_break_character

    def _flatten_dict(self, d: dict, prefix: str = "", target_dict: Optional[dict] = None) -> dict:
        """Helper method to flatten a nested dictionary."""
        if target_dict is None:
            target_dict = {}

        for key, value in d.items():
            new_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                self._flatten_dict(value, f"{new_key}_", target_dict)
            else:
                target_dict[new_key] = value

        return target_dict

    def _process_split_content(self, split_content: str, split_index: int) -> int:
        """Process the content of a split and return the number of page breaks."""
        if not isinstance(split_content, str):
            return 0

        page_breaks = split_content.count(self.page_break_character)
        if page_breaks > 0:
            logger.debug(
                "Found {page_breaks} page breaks in split {split_index}",
                page_breaks=page_breaks,
                split_index=split_index,
            )
        return page_breaks

    def _split_by_function(self, doc: Document) -> list[Document]:
        """Split document using a custom function that returns dictionaries with 'content' and 'meta'."""
        logger.debug("Splitting document with id={doc_id}", doc_id=doc.id)
        splits = self.splitting_function(doc.content)
        docs = []

        # calculate total pages and set current page
        total_pages = doc.meta.get("total_pages", 0) or doc.content.count(self.page_break_character) + 1
        current_page = doc.meta.get("page_number", 1)
        logger.debug(
            "Starting page number: {current_page}, Total pages: {total_pages}",
            current_page=current_page,
            total_pages=total_pages,
        )

        # get meta for each split
        for i, split in enumerate(splits):
            meta = {}
            if doc.meta:
                meta = self._flatten_dict(doc.meta)

            # add standard metadata (no split_id here)
            meta.update({"source_id": doc.id, "total_pages": total_pages, "page_number": current_page})

            # get page number based on page breaks
            page_breaks = self._process_split_content(split["content"], i)
            current_page += page_breaks

            # add split-specific metadata
            if split.get("meta"):
                meta.update(self._flatten_dict(split.get("meta")))

            docs.append(Document(content=split["content"], meta=meta))

        logger.debug(
            "Split into {num_docs} documents for id={doc_id}, final page: {current_page}",
            num_docs=len(docs),
            doc_id=doc.id,
            current_page=current_page,
        )
        return docs


@component
class MarkdownHeaderSplitter:
    """
    A custom component that splits documents at markdown headers with optional secondary splitting.
    """

    def __init__(
        self,
        infer_header_levels: bool = False,
        page_break_character: str = "\\f",
        secondary_split: Literal["none", "word", "passage", "period", "line"] = "none",
        split_length: int = 200,
        split_overlap: int = 0,
        split_threshold: int = 0,
    ):
        """
        Initialize the MarkdownHeaderSplitter.

        :param infer_header_levels: If True, attempts to infer and rewrite header levels based on content structure.
            Useful for documents where all headers use the same level. Defaults to False.
        :param page_break_character: Character used to identify page breaks. Defaults to form feed ("\\f").
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
        else:
            self.secondary_splitter = None

    def _infer_and_rewrite_header_levels(self, text: str) -> str:
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

    def _split_by_markdown_headers(self, text: str) -> list[dict]:
        """Split text by markdown headers and create chunks with appropriate metadata."""
        logger.debug("Splitting text by markdown headers")

        # find headers
        pattern = r"(?m)^(#{1,6}) (.+)$"
        matches = list(re.finditer(pattern, text))

        # return unsplit if no headers found
        if not matches:
            logger.info("No headers found in document; returning full document as single chunk.")
            return [{"content": text, "meta": {"header": None, "parentheaders": []}}]

        # process headers and build chunks
        chunks = []
        header_stack = [None] * 6
        active_parents = []

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
                active_parents = [h for h in header_stack[: level - 1] if h]
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
            active_parents = [h for h in header_stack[: level - 1] if h]

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
            # extract header information
            header_match = re.search(r"(#{1,6}) (.+)(?:\n|$)", doc.content)
            if header_match:
                content_for_splitting = doc.content[header_match.end() :]
            else:
                content_for_splitting = doc.content

            if not content_for_splitting.strip():  # skip empty content
                result_docs.append(doc)
                continue

            # track page from meta
            current_page = doc.meta.get("page_number", 1)

            # use the pre-initialized secondary splitter
            secondary_splits = self.secondary_splitter.run(
                documents=[Document(content=content_for_splitting, meta=doc.meta)]
            )["documents"]
            accumulated_page_breaks = 0  # track page breaks

            # split processing
            for i, split in enumerate(secondary_splits):
                # calculate page number for this split
                if i > 0:  # page break counting
                    prev_content = secondary_splits[i - 1].content
                    page_breaks = prev_content.count(self.page_break_character)
                    accumulated_page_breaks += page_breaks

                # set page number to meta
                split.meta["page_number"] = current_page + accumulated_page_breaks

                # preserve header metadata
                for key in ["header", "parentheaders"]:
                    if key in doc.meta:
                        split.meta[key] = doc.meta[key]

                result_docs.append(split)

        # assign unique, sequential split_id to all final chunks
        for idx, doc in enumerate(result_docs):
            doc.meta["split_id"] = idx

        logger.info("Secondary splitting complete. Final count: {final_count} documents.", final_count=len(result_docs))
        return result_docs

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], infer_header_levels: Optional[bool] = None) -> dict[str, list[Document]]:
        """
        Run the markdown header splitter with optional secondary splitting.

        :param documents: List of documents to split
        :param infer_header_levels: If True, attempts to infer and rewrite header levels before splitting.
            If None, uses the value from initialization.
        """
        infer_header_levels = infer_header_levels if infer_header_levels is not None else self.infer_header_levels

        # process documents - preprocess if told to
        processed_documents = []
        for doc in documents:
            if infer_header_levels:
                content = self._infer_and_rewrite_header_levels(doc.content)
                processed_documents.append(Document(content=content, meta=doc.meta, id=doc.id))
            else:
                processed_documents.append(doc)

        # split by markdown headers
        header_splitter = _CustomDocumentSplitter(
            split_by="function",
            splitting_function=lambda text: self._split_by_markdown_headers(text),
            page_break_character=self.page_break_character,
        )

        # get splits
        header_split_docs = header_splitter.run(documents=processed_documents)["documents"]
        logger.info("Header splitting produced {num_docs} documents", num_docs=len(header_split_docs))

        # apply secondary splitting if requested
        if self.secondary_split != "none":
            final_docs = self._apply_secondary_splitting(header_split_docs)
        else:
            final_docs = header_split_docs

        # assign unique, sequential split_id to all final chunks
        for idx, doc in enumerate(final_docs):
            doc.meta["split_id"] = idx

        return {"documents": final_docs}


# TODO: move to proper test file once ready
if __name__ == "__main__":
    print()
    print("===== Example 1: Regular splitting =====")
    splitter = MarkdownHeaderSplitter()
    content = """# Header 1
## Subheader 1.1
Content under subheader 1.1.
## Subheader 1.2
### Subheader 1.2.1
Content under subheader 1.2.1."""
    print("Original content:")
    print(content)
    example_doc = Document(content=content)
    result = splitter.run(documents=[example_doc])
    for doc in result["documents"]:
        print("\n---Document---")
        print(doc.content)
        print(doc.meta)

    print()
    print("===== Example 2: Splitting with header inference =====")
    splitter = MarkdownHeaderSplitter(infer_header_levels=True)
    content = """## Header 1
## Subheader 1.1
Content under subheader 1.1.
## Subheader 1.2
## Subheader 1.2.1
Content under subheader 1.2.1."""
    print("Original content:")
    print(content)
    example_doc = Document(content=content)
    result = splitter.run(documents=[example_doc])
    print("\nAfter header inference and splitting:")
    for doc in result["documents"]:
        print("\n---Document---")
        print(doc.content)
        print(doc.meta)
        print(doc.content)
        print(doc.meta)
