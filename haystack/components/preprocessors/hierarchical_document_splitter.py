# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Literal, Set

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.preprocessors import DocumentSplitter


@component
class HierarchicalDocumentSplitter:
    """
    Splits a documents into different block sizes building a hierarchical tree structure of blocks of different sizes.

    The root node of the tree is the original document, the leaf nodes are the smallest blocks. The blocks in between
    are connected such that the smaller blocks are children of the parent-larger blocks.

    ## Usage example
    ```python
    from haystack import Document
    from haystack.components.preprocessors import HierarchicalDocumentSplitter

    doc = Document(content="This is a simple test document")
    splitter = HierarchicalDocumentSplitter(block_sizes={3, 2}, split_overlap=0, split_by="word")
    splitter.run([doc])
    >> {'documents': [Document(id=3f7..., content: 'This is a simple test document', meta: {'block_size': 0, 'parent_id': None, 'children_ids': ['5ff..', '8dc..'], 'level': 0}),
    >> Document(id=5ff.., content: 'This is a ', meta: {'block_size': 3, 'parent_id': '3f7..', 'children_ids': ['f19..', '52c..'], 'level': 1, 'source_id': '3f7..', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}),
    >> Document(id=8dc.., content: 'simple test document', meta: {'block_size': 3, 'parent_id': '3f7..', 'children_ids': ['39d..', 'e23..'], 'level': 1, 'source_id': '3f7..', 'page_number': 1, 'split_id': 1, 'split_idx_start': 10}),
    >> Document(id=f19.., content: 'This is ', meta: {'block_size': 2, 'parent_id': '5ff..', 'children_ids': [], 'level': 2, 'source_id': '5ff..', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}),
    >> Document(id=52c.., content: 'a ', meta: {'block_size': 2, 'parent_id': '5ff..', 'children_ids': [], 'level': 2, 'source_id': '5ff..', 'page_number': 1, 'split_id': 1, 'split_idx_start': 8}),
    >> Document(id=39d.., content: 'simple test ', meta: {'block_size': 2, 'parent_id': '8dc..', 'children_ids': [], 'level': 2, 'source_id': '8dc..', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}),
    >> Document(id=e23.., content: 'document', meta: {'block_size': 2, 'parent_id': '8dc..', 'children_ids': [], 'level': 2, 'source_id': '8dc..', 'page_number': 1, 'split_id': 1, 'split_idx_start': 12})]}
    ```
    """  # noqa: E501

    def __init__(
        self,
        block_sizes: Set[int],
        split_overlap: int = 0,
        split_by: Literal["word", "sentence", "page", "passage"] = "word",
    ):
        """
        Initialize HierarchicalDocumentSplitter.

        :param block_sizes: Set of block sizes to split the document into. The blocks are split in descending order.
        :param split_overlap: The number of overlapping units for each split.
        :param split_by: The unit for splitting your documents.
        """

        self.block_sizes = sorted(set(block_sizes), reverse=True)
        self.splitters: Dict[int, DocumentSplitter] = {}
        self.split_overlap = split_overlap
        self.split_by = split_by
        self._build_block_sizes()

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Builds a hierarchical document structure for each document in a list of documents.

        :param documents: List of Documents to split into hierarchical blocks.
        :returns: List of HierarchicalDocument
        """
        hierarchical_docs = []
        for doc in documents:
            hierarchical_docs.extend(self.build_hierarchy_from_doc(doc))
        return {"documents": hierarchical_docs}

    def _build_block_sizes(self):
        for block_size in self.block_sizes:
            self.splitters[block_size] = DocumentSplitter(
                split_length=block_size, split_overlap=self.split_overlap, split_by=self.split_by
            )
            self.splitters[block_size].warm_up()

    @staticmethod
    def _add_meta_data(document: Document):
        document.meta["__block_size"] = 0
        document.meta["__parent_id"] = None
        document.meta["__children_ids"] = []
        document.meta["__level"] = 0
        return document

    def build_hierarchy_from_doc(self, document: Document) -> List[Document]:
        """
        Build a hierarchical tree document structure from a single document.

        Given a document, this function splits the document into hierarchical blocks of different sizes represented
        as HierarchicalDocument objects.

        :param document: Document to split into hierarchical blocks.
        :returns:
            List of HierarchicalDocument
        """

        root = self._add_meta_data(document)
        current_level_nodes = [root]
        all_docs = []

        for block in self.block_sizes:
            next_level_nodes = []
            for doc in current_level_nodes:
                splitted_docs = self.splitters[block].run([doc])
                child_docs = splitted_docs["documents"]
                # if it's only one document skip
                if len(child_docs) == 1:
                    next_level_nodes.append(doc)
                    continue
                for child_doc in child_docs:
                    child_doc = self._add_meta_data(child_doc)
                    child_doc.meta["__level"] = doc.meta["__level"] + 1
                    child_doc.meta["__block_size"] = block
                    child_doc.meta["__parent_id"] = doc.id
                    all_docs.append(child_doc)
                    doc.meta["__children_ids"].append(child_doc.id)
                    next_level_nodes.append(child_doc)
            current_level_nodes = next_level_nodes

        return [root] + all_docs

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the component.

        :returns:
                Serialized dictionary representation of the component.
        """
        return default_to_dict(
            self, block_sizes=self.block_sizes, split_overlap=self.split_overlap, split_by=self.split_by
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchicalDocumentSplitter":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary to deserialize and create the component.

        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)
