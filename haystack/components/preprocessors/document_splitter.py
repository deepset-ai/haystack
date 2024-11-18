# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from more_itertools import windowed

from haystack import Document, component, logging
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.utils import deserialize_callable, serialize_callable

logger = logging.getLogger(__name__)

# Maps the 'split_by' argument to the actual char used to split the Documents.
# 'function' is not in the mapping cause it doesn't split on chars.
_SPLIT_BY_MAPPING = {"page": "\f", "passage": "\n\n", "sentence": ".", "word": " ", "line": "\n"}


@component
class DocumentSplitter:
    """
    Splits long documents into smaller chunks.

    This is a common preprocessing step during indexing.
    It helps Embedders create meaningful semantic representations
    and prevents exceeding language model context limits.

    The DocumentSplitter is compatible with the following DocumentStores:
    - [Astra](https://docs.haystack.deepset.ai/docs/astradocumentstore)
    - [Chroma](https://docs.haystack.deepset.ai/docs/chromadocumentstore) limited support, overlapping information is
      not stored
    - [Elasticsearch](https://docs.haystack.deepset.ai/docs/elasticsearch-document-store)
    - [OpenSearch](https://docs.haystack.deepset.ai/docs/opensearch-document-store)
    - [Pgvector](https://docs.haystack.deepset.ai/docs/pgvectordocumentstore)
    - [Pinecone](https://docs.haystack.deepset.ai/docs/pinecone-document-store) limited support, overlapping
       information is not stored
    - [Qdrant](https://docs.haystack.deepset.ai/docs/qdrant-document-store)
    - [Weaviate](https://docs.haystack.deepset.ai/docs/weaviatedocumentstore)

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.preprocessors import DocumentSplitter

    doc = Document(content="Moonlight shimmered softly, wolves howled nearby, night enveloped everything.")

    splitter = DocumentSplitter(split_by="word", split_length=3, split_overlap=0)
    result = splitter.run(documents=[doc])
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        split_by: Literal["function", "page", "passage", "sentence", "word", "line"] = "word",
        split_length: int = 200,
        split_overlap: int = 0,
        split_threshold: int = 0,
        splitting_function: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        Initialize DocumentSplitter.

        :param split_by: The unit for splitting your documents. Choose from `word` for splitting by spaces (" "),
            `sentence` for splitting by periods ("."), `page` for splitting by form feed ("\\f"),
            `passage` for splitting by double line breaks ("\\n\\n") or `line` for splitting each line ("\\n").
        :param split_length: The maximum number of units in each split.
        :param split_overlap: The number of overlapping units for each split.
        :param split_threshold: The minimum number of units per split. If a split has fewer units
            than the threshold, it's attached to the previous split.
        :param splitting_function: Necessary when `split_by` is set to "function".
            This is a function which must accept a single `str` as input and return a `list` of `str` as output,
            representing the chunks after splitting.
        """

        self.split_by = split_by
        if split_by not in ["function", "page", "passage", "sentence", "word", "line"]:
            raise ValueError("split_by must be one of 'function', 'word', 'sentence', 'page', 'passage' or 'line'.")
        if split_by == "function" and splitting_function is None:
            raise ValueError("When 'split_by' is set to 'function', a valid 'splitting_function' must be provided.")
        if split_length <= 0:
            raise ValueError("split_length must be greater than 0.")
        self.split_length = split_length
        if split_overlap < 0:
            raise ValueError("split_overlap must be greater than or equal to 0.")
        self.split_overlap = split_overlap
        self.split_threshold = split_threshold
        self.splitting_function = splitting_function

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Split documents into smaller parts.

        Splits documents by the unit expressed in `split_by`, with a length of `split_length`
        and an overlap of `split_overlap`.

        :param documents: The documents to split.

        :returns: A dictionary with the following key:
            - `documents`: List of documents with the split texts. Each document includes:
                - A metadata field `source_id` to track the original document.
                - A metadata field `page_number` to track the original page number.
                - All other metadata copied from the original document.

        :raises TypeError: if the input is not a list of Documents.
        :raises ValueError: if the content of a document is None.
        """

        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("DocumentSplitter expects a List of Documents as input.")

        split_docs: List[Document] = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    f"DocumentSplitter only works with text documents but content for document ID {doc.id} is None."
                )
            if doc.content == "":
                logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                continue
            split_docs += self._split(doc)
        return {"documents": split_docs}

    def _split(self, to_split: Document) -> List[Document]:
        # We already check this before calling _split but
        # we need to make linters happy
        if to_split.content is None:
            return []

        if self.split_by == "function" and self.splitting_function is not None:
            splits = self.splitting_function(to_split.content)
            docs: List[Document] = []
            for s in splits:
                meta = deepcopy(to_split.meta)
                meta["source_id"] = to_split.id
                docs.append(Document(content=s, meta=meta))
            return docs

        split_at = _SPLIT_BY_MAPPING[self.split_by]
        units = to_split.content.split(split_at)
        # Add the delimiter back to all units except the last one
        for i in range(len(units) - 1):
            units[i] += split_at

        text_splits, splits_pages, splits_start_idxs = self._concatenate_units(
            units, self.split_length, self.split_overlap, self.split_threshold
        )
        metadata = deepcopy(to_split.meta)
        metadata["source_id"] = to_split.id
        return self._create_docs_from_splits(
            text_splits=text_splits, splits_pages=splits_pages, splits_start_idxs=splits_start_idxs, meta=metadata
        )

    def _concatenate_units(
        self, elements: List[str], split_length: int, split_overlap: int, split_threshold: int
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Concatenates the elements into parts of split_length units.

        Keeps track of the original page number that each element belongs. If the length of the current units is less
        than the pre-defined `split_threshold`, it does not create a new split. Instead, it concatenates the current
        units with the last split, preventing the creation of excessively small splits.
        """

        text_splits: List[str] = []
        splits_pages: List[int] = []
        splits_start_idxs: List[int] = []
        cur_start_idx = 0
        cur_page = 1
        segments = windowed(elements, n=split_length, step=split_length - split_overlap)

        for seg in segments:
            current_units = [unit for unit in seg if unit is not None]
            txt = "".join(current_units)

            # check if length of current units is below split_threshold
            if len(current_units) < split_threshold and len(text_splits) > 0:
                # concatenate the last split with the current one
                text_splits[-1] += txt

            # NOTE: This line skips documents that have content=""
            elif len(txt) > 0:
                text_splits.append(txt)
                splits_pages.append(cur_page)
                splits_start_idxs.append(cur_start_idx)

            processed_units = current_units[: split_length - split_overlap]
            cur_start_idx += len("".join(processed_units))

            if self.split_by == "page":
                num_page_breaks = len(processed_units)
            else:
                num_page_breaks = sum(processed_unit.count("\f") for processed_unit in processed_units)

            cur_page += num_page_breaks

        return text_splits, splits_pages, splits_start_idxs

    def _create_docs_from_splits(
        self, text_splits: List[str], splits_pages: List[int], splits_start_idxs: List[int], meta: Dict[str, Any]
    ) -> List[Document]:
        """
        Creates Document objects from splits enriching them with page number and the metadata of the original document.
        """
        documents: List[Document] = []

        for i, (txt, split_idx) in enumerate(zip(text_splits, splits_start_idxs)):
            meta = deepcopy(meta)
            doc = Document(content=txt, meta=meta)
            doc.meta["page_number"] = splits_pages[i]
            doc.meta["split_id"] = i
            doc.meta["split_idx_start"] = split_idx
            documents.append(doc)

            if self.split_overlap <= 0:
                continue

            doc.meta["_split_overlap"] = []

            if i == 0:
                continue

            doc_start_idx = splits_start_idxs[i]
            previous_doc = documents[i - 1]
            previous_doc_start_idx = splits_start_idxs[i - 1]
            self._add_split_overlap_information(doc, doc_start_idx, previous_doc, previous_doc_start_idx)

        return documents

    @staticmethod
    def _add_split_overlap_information(
        current_doc: Document, current_doc_start_idx: int, previous_doc: Document, previous_doc_start_idx: int
    ):
        """
        Adds split overlap information to the current and previous Document's meta.

        :param current_doc: The Document that is being split.
        :param current_doc_start_idx: The starting index of the current Document.
        :param previous_doc: The Document that was split before the current Document.
        :param previous_doc_start_idx: The starting index of the previous Document.
        """
        overlapping_range = (current_doc_start_idx - previous_doc_start_idx, len(previous_doc.content))  # type: ignore

        if overlapping_range[0] < overlapping_range[1]:
            overlapping_str = previous_doc.content[overlapping_range[0] : overlapping_range[1]]  # type: ignore

            if current_doc.content.startswith(overlapping_str):  # type: ignore
                # add split overlap information to this Document regarding the previous Document
                current_doc.meta["_split_overlap"].append({"doc_id": previous_doc.id, "range": overlapping_range})

                # add split overlap information to previous Document regarding this Document
                overlapping_range = (0, overlapping_range[1] - overlapping_range[0])
                previous_doc.meta["_split_overlap"].append({"doc_id": current_doc.id, "range": overlapping_range})

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        serialized = default_to_dict(
            self,
            split_by=self.split_by,
            split_length=self.split_length,
            split_overlap=self.split_overlap,
            split_threshold=self.split_threshold,
        )
        if self.splitting_function:
            serialized["init_parameters"]["splitting_function"] = serialize_callable(self.splitting_function)
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentSplitter":
        """
        Deserializes the component from a dictionary.
        """
        init_params = data.get("init_parameters", {})

        splitting_function = init_params.get("splitting_function", None)
        if splitting_function:
            init_params["splitting_function"] = deserialize_callable(splitting_function)

        return default_from_dict(cls, data)
