import logging
import re
from typing import Any, Dict, List, Literal, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.preprocessors import DocumentSplitter

logger = logging.getLogger(__name__)


class CustomDocumentSplitter(DocumentSplitter):
    """
    Custom DocumentSplitter that supports splitting functions returning dicts with 'content' and 'meta'.
    """

    def __init__(self, *args, page_break_character="\\f", **kwargs):
        super().__init__(*args, **kwargs)
        self.page_break_character = page_break_character

    def _flatten_dict(self, d: Dict, prefix: str = "", target_dict: Optional[Dict] = None) -> Dict:
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
            logger.debug(f"Found {page_breaks} page breaks in split {split_index}")
        return page_breaks

    def _split_by_function(self, doc: Document) -> List[Document]:
        """Split document using a custom function that returns dictionaries with 'content' and 'meta'."""
        logger.debug(f"Splitting document with id={doc.id}")
        splits = self.splitting_function(doc.content)
        docs = []

        # calculate total pages and set current page
        total_pages = doc.meta.get("total_pages", 0) or doc.content.count(self.page_break_character) + 1
        current_page = doc.meta.get("page_number", 1)
        logger.debug(f"Starting page number: {current_page}, Total pages: {total_pages}")

        # get meta for each split
        for i, split in enumerate(splits):
            meta = {}
            if doc.meta:
                meta = self._flatten_dict(doc.meta)

            # add standard metadata
            meta.update({"source_id": doc.id, "split_id": i, "total_pages": total_pages, "page_number": current_page})

            # get page number based on page breaks
            page_breaks = self._process_split_content(split["content"], i)
            current_page += page_breaks

            # add split-specific metadata
            if split.get("meta"):
                meta.update(self._flatten_dict(split.get("meta")))

            docs.append(Document(content=split["content"], meta=meta))

        logger.debug(f"Split into {len(docs)} documents for id={doc.id}, final page: {current_page}")
        return docs


@component
class MarkdownHeaderSplitter:
    """
    A custom component that splits documents at markdown headers with optional secondary splitting.

    :param enforce_first_header: If True, ensures the first header is always included in the parent headers.
        This is useful for docling outputs where header levels are uniformly detected and the first header
        is often overwritten. Defaults to False.
    :param page_break_character: Character used to identify page breaks. Defaults to form feed ("\\f").
    :param secondary_split: Optional secondary split condition after header splitting.
        Options are "none", "word", "passage", "period", "line". Defaults to "none".
    :param split_length: The maximum number of units in each split when using secondary splitting. Defaults to 200.
    :param split_overlap: The number of overlapping units for each split when using secondary splitting. Defaults to 0.
    :param split_threshold: The minimum number of units per split when using secondary splitting. Defaults to 0.
    """

    def __init__(
        self,
        enforce_first_header: bool = False,
        page_break_character: str = "\\f",
        secondary_split: Literal["none", "word", "passage", "period", "line"] = "none",
        split_length: int = 200,
        split_overlap: int = 0,
        split_threshold: int = 0,
    ):
        self.enforce_first_header = enforce_first_header
        self.page_break_character = page_break_character
        self.secondary_split = secondary_split
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_threshold = split_threshold

    def _split_by_markdown_headers(self, text: str, enforce_first_header: Optional[bool] = None) -> List[Dict]:
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
        first_header = matches[0].group(2).strip()

        for i, match in enumerate(matches):
            # Extract header info
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

            # enforce first header if needed
            if enforce_first_header and first_header and (not parentheaders or parentheaders[0] != first_header):
                parentheaders = [first_header] + [h for h in parentheaders if h != first_header]

            logger.debug(f"Creating chunk for header '{header_text}' at level {level}")

            chunks.append(
                {
                    "content": f"{header_prefix} {header_text}\n{content}",
                    "meta": {"header": header_text, "parentheaders": parentheaders},
                }
            )

            # reset active parents
            active_parents = [h for h in header_stack[: level - 1] if h]

        logger.info(f"Split into {len(chunks)} chunks by markdown headers.")
        return chunks

    def _apply_secondary_splitting(self, documents: List[Document]) -> List[Document]:
        """
        Apply secondary splitting while preserving header metadata and structure.

        Ensures page counting is maintained across splits.
        """
        if self.secondary_split == "none":
            return documents

        logger.info(f"Applying secondary splitting by {self.secondary_split}")
        result_docs = []

        for doc in documents:
            # extract header information
            header_match = re.search(r"(#{1,6}) (.+)(?:\n|$)", doc.content)
            if header_match:
                header_prefix = header_match.group(0) + "\n"
                content_for_splitting = doc.content[header_match.end() :]
            else:
                header_prefix = ""
                content_for_splitting = doc.content

            if not content_for_splitting.strip():  # skip empty content
                result_docs.append(doc)
                continue

            # track page from meta
            current_page = doc.meta.get("page_number", 1)

            secondary_splitter = DocumentSplitter(
                split_by=self.secondary_split,
                split_length=self.split_length,
                split_overlap=self.split_overlap,
                split_threshold=self.split_threshold,
            )

            # apply secondary splitting
            temp_doc = Document(content=content_for_splitting, meta=doc.meta)
            secondary_splits = secondary_splitter.run(documents=[temp_doc])["documents"]
            parent_headers = doc.meta.get("parentheaders", [])
            first_header = parent_headers[0] if parent_headers else None
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

                if header_prefix:  # add header prefix to content
                    split.content = header_prefix + split.content

                # preserve header metadata
                for key in ["header", "parentheaders"]:
                    if key in doc.meta:
                        split.meta[key] = doc.meta[key]

                # enforce first header if needed
                if self.enforce_first_header and first_header:
                    parentheaders = split.meta.get("parentheaders", [])
                    if not parentheaders:
                        split.meta["parentheaders"] = [first_header]
                    elif parentheaders[0] != first_header:
                        split.meta["parentheaders"] = [first_header] + [h for h in parentheaders if h != first_header]
                # preserve primary split ID
                if "split_id" in doc.meta:
                    split.meta["header_split_id"] = doc.meta["split_id"]

                result_docs.append(split)

        logger.info(f"Secondary splitting complete. Final count: {len(result_docs)} documents.")
        return result_docs

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], enforce_first_header: Optional[bool] = None) -> Dict[str, List[Document]]:
        """
        Run the markdown header splitter with optional secondary splitting.

        :param documents: List of documents to split
        :param enforce_first_header: If True, ensures the first header is included in all parentheaders.
            If None, uses the value from initialization.
        """
        logger.info(f"Processing {len(documents)} documents with enforce_first_header={enforce_first_header}")

        # split by markdown headers
        header_splitter = CustomDocumentSplitter(
            split_by="function",
            splitting_function=lambda text: self._split_by_markdown_headers(text, enforce_first_header),
            page_break_character=self.page_break_character,
        )

        # get splits
        header_split_docs = header_splitter.run(documents=documents)["documents"]
        logger.info(f"Header splitting produced {len(header_split_docs)} documents")

        # apply secondary splitting if requested
        if self.secondary_split != "none":
            final_docs = self._apply_secondary_splitting(header_split_docs)
        else:
            final_docs = header_split_docs

        return {"documents": final_docs}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary."""
        return default_to_dict(
            self,
            enforce_first_header=self.enforce_first_header,
            page_break_character=self.page_break_character,
            secondary_split=self.secondary_split,
            split_length=self.split_length,
            split_overlap=self.split_overlap,
            split_threshold=self.split_threshold,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarkdownHeaderSplitter":
        """Deserialize component from dictionary."""
        return default_from_dict(cls, data)
