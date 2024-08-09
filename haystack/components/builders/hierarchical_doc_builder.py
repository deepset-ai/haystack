# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Literal

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.preprocessors import DocumentSplitter


@component
class HierarchicalDocumentBuilder:
    """
    Splits a documents into different block sizes building a hierarchical tree structure of blocks of different sizes.

    The root node is the original document, the leaf nodes are the smallest blocks. The blocks in between are connected
    such that the smaller blocks are children of the parent-larger blocks.

    ## Usage example
    ```python
    from haystack import Document
    from haystack.components.builders import HierarchicalDocumentBuilder

    doc = Document(content="This is a test document")
    builder = HierarchicalDocumentBuilder(block_sizes=[10, 5, 2], split_overlap=0, split_by="word")
    builder.run([doc])
    >> {'documents': [Document(id=82c6fe8eaac88f2ac6773ef8909f4b28676fe09ea12657820e10b441df6142a4, content: 'This is a test document', meta: {'block_size': 10, 'parent_id': '2594f015eb8eca9dd088d8e7ecc9b25e16681842815088a7535566d6ac38cdcf', 'children_ids': ['f28b00099c27730918311da04f2460afbfea7970fe17a5a603a017ee6f262ce6'], 'level': 1, 'source_id': '2594f015eb8eca9dd088d8e7ecc9b25e16681842815088a7535566d6ac38cdcf', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}),
    >> Document(id=f28b00099c27730918311da04f2460afbfea7970fe17a5a603a017ee6f262ce6, content: 'This is a test document', meta: {'block_size': 5, 'parent_id': '82c6fe8eaac88f2ac6773ef8909f4b28676fe09ea12657820e10b441df6142a4', 'children_ids': ['a1efed06362249712c7c162a36bf19066efcbf3ac2c171591d704d3792cd1521', '27662dfa9bce83008e359ebac49fb9b54f19392e7e85c112a0182dcc69ab4484', '97396afcd387a736e9be6dd882b1d0de7ce91c809b81a231ad925cd1163ea397'], 'level': 2, 'source_id': '82c6fe8eaac88f2ac6773ef8909f4b28676fe09ea12657820e10b441df6142a4', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}),
    >> Document(id=a1efed06362249712c7c162a36bf19066efcbf3ac2c171591d704d3792cd1521, content: 'This is ', meta: {'block_size': 2, 'parent_id': 'f28b00099c27730918311da04f2460afbfea7970fe17a5a603a017ee6f262ce6', 'children_ids': [], 'level': 3, 'source_id': 'f28b00099c27730918311da04f2460afbfea7970fe17a5a603a017ee6f262ce6', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}),
    >> Document(id=27662dfa9bce83008e359ebac49fb9b54f19392e7e85c112a0182dcc69ab4484, content: 'a test ', meta: {'block_size': 2, 'parent_id': 'f28b00099c27730918311da04f2460afbfea7970fe17a5a603a017ee6f262ce6', 'children_ids': [], 'level': 3, 'source_id': 'f28b00099c27730918311da04f2460afbfea7970fe17a5a603a017ee6f262ce6', 'page_number': 1, 'split_id': 1, 'split_idx_start': 8}),
    >> Document(id=97396afcd387a736e9be6dd882b1d0de7ce91c809b81a231ad925cd1163ea397, content: 'document', meta: {'block_size': 2, 'parent_id': 'f28b00099c27730918311da04f2460afbfea7970fe17a5a603a017ee6f262ce6', 'children_ids': [], 'level': 3, 'source_id': 'f28b00099c27730918311da04f2460afbfea7970fe17a5a603a017ee6f262ce6', 'page_number': 1, 'split_id': 2, 'split_idx_start': 15})]}
    ```
    """  # noqa: E501

    def __init__(
        self,
        block_sizes: List[int],
        split_overlap: int = 0,
        split_by: Literal["word", "sentence", "page", "passage"] = "word",
    ):
        """
        Initialize HierarchicalDocumentBuilder.

        :param block_sizes: List of block sizes to split the document into. The blocks are split in descending order.
        :param split_overlap: The number of overlapping units for each split.
        :param split_by: The unit for splitting your documents.
        """

        if len(set(block_sizes)) != len(block_sizes):
            raise ValueError("block_sizes must not contain duplicates")
        self.block_sizes = sorted(set(block_sizes), reverse=True)
        self.split_overlap = split_overlap
        self.split_by = split_by

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Builds a hierarchical document structure for each document in a list of documents.

        :param documents: List of Documents to split into hierarchical blocks.
        :return: List of HierarchicalDocument
        """
        hierarchical_docs = []
        for doc in documents:
            hierarchical_docs.extend(self.build_hierarchy_from_doc(doc))
        return {"documents": hierarchical_docs}

    def _split_doc(self, doc: Document, block_size: int) -> List[Document]:
        splitter = DocumentSplitter(split_length=block_size, split_overlap=self.split_overlap, split_by=self.split_by)
        split_docs = splitter.run([doc])
        return split_docs["documents"]

    @staticmethod
    def _add_meta_data(document: Document):
        document.meta["block_size"] = 0
        document.meta["parent_id"] = None
        document.meta["children_ids"] = []
        document.meta["level"] = 0
        return document

    def build_hierarchy_from_doc(self, document: Document) -> List[Document]:
        """
        Build a hierarchical tree document structure from a single document.

        Given a document, this function splits the document into hierarchical blocks of different sizes represented
        as HierarchicalDocument objects

        :param document: Document to split into hierarchical blocks.
        :return:
            List of HierarchicalDocument
        """

        root = self._add_meta_data(document)
        current_level_nodes = [root]
        all_docs = []

        for block in self.block_sizes:
            next_level_nodes = []
            for doc in current_level_nodes:
                child_docs = self._split_doc(doc, block)
                for child_doc in child_docs:
                    child_doc = self._add_meta_data(child_doc)
                    child_doc.meta["level"] = doc.meta["level"] + 1
                    child_doc.meta["block_size"] = block
                    child_doc.meta["parent_id"] = doc.id
                    all_docs.append(child_doc)
                    doc.meta["children_ids"].append(child_doc.id)
                    next_level_nodes.append(child_doc)
            current_level_nodes = next_level_nodes

        return all_docs

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
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchicalDocumentBuilder":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary to deserialize and create the component.

        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)
