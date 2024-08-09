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

    doc = Document(content="This is a simple test document")
    builder = HierarchicalDocumentBuilder(block_sizes=[3, 2], split_overlap=0, split_by="word")
    builder.run([doc])
    >> {'documents': [Document(id=3f7e91e9a775ed0815606a0bc2f732b38b7682a84d5b23c06997a8dcfa849a0d, content: 'This is a simple test document', meta: {'block_size': 0, 'parent_id': None, 'children_ids': ['5ff4a36b5371580ac5f2ea21a5690dcc18802c7e5e187d57c5e2d312eee22dfd', '8dc5707ebe647dedab97db15d0fc82a5e551bbe54a9fb82f79b68b3e3046a3a2'], 'level': 0}),
    >> Document(id=5ff4a36b5371580ac5f2ea21a5690dcc18802c7e5e187d57c5e2d312eee22dfd, content: 'This is a ', meta: {'block_size': 3, 'parent_id': '3f7e91e9a775ed0815606a0bc2f732b38b7682a84d5b23c06997a8dcfa849a0d', 'children_ids': ['f196b211ebadd5f47afedff14284759b4654f0722c38976760b88d675e7dc8f6', '52c7e9fc53ae9aa734cc15d8624ae63468b423a7032077ee4cdcf524569274d3'], 'level': 1, 'source_id': '3f7e91e9a775ed0815606a0bc2f732b38b7682a84d5b23c06997a8dcfa849a0d', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}),
    >> Document(id=8dc5707ebe647dedab97db15d0fc82a5e551bbe54a9fb82f79b68b3e3046a3a2, content: 'simple test document', meta: {'block_size': 3, 'parent_id': '3f7e91e9a775ed0815606a0bc2f732b38b7682a84d5b23c06997a8dcfa849a0d', 'children_ids': ['39d299629a35051fc9ebb62a0594626546d6ec1b1de7cfcdfb03be58b769478e', 'e23ceb261772f539830384097e4f6c513205019cf422378078ff66dd4870f91a'], 'level': 1, 'source_id': '3f7e91e9a775ed0815606a0bc2f732b38b7682a84d5b23c06997a8dcfa849a0d', 'page_number': 1, 'split_id': 1, 'split_idx_start': 10}),
    >> Document(id=f196b211ebadd5f47afedff14284759b4654f0722c38976760b88d675e7dc8f6, content: 'This is ', meta: {'block_size': 2, 'parent_id': '5ff4a36b5371580ac5f2ea21a5690dcc18802c7e5e187d57c5e2d312eee22dfd', 'children_ids': [], 'level': 2, 'source_id': '5ff4a36b5371580ac5f2ea21a5690dcc18802c7e5e187d57c5e2d312eee22dfd', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}),
    >> Document(id=52c7e9fc53ae9aa734cc15d8624ae63468b423a7032077ee4cdcf524569274d3, content: 'a ', meta: {'block_size': 2, 'parent_id': '5ff4a36b5371580ac5f2ea21a5690dcc18802c7e5e187d57c5e2d312eee22dfd', 'children_ids': [], 'level': 2, 'source_id': '5ff4a36b5371580ac5f2ea21a5690dcc18802c7e5e187d57c5e2d312eee22dfd', 'page_number': 1, 'split_id': 1, 'split_idx_start': 8}),
    >> Document(id=39d299629a35051fc9ebb62a0594626546d6ec1b1de7cfcdfb03be58b769478e, content: 'simple test ', meta: {'block_size': 2, 'parent_id': '8dc5707ebe647dedab97db15d0fc82a5e551bbe54a9fb82f79b68b3e3046a3a2', 'children_ids': [], 'level': 2, 'source_id': '8dc5707ebe647dedab97db15d0fc82a5e551bbe54a9fb82f79b68b3e3046a3a2', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}),
    >> Document(id=e23ceb261772f539830384097e4f6c513205019cf422378078ff66dd4870f91a, content: 'document', meta: {'block_size': 2, 'parent_id': '8dc5707ebe647dedab97db15d0fc82a5e551bbe54a9fb82f79b68b3e3046a3a2', 'children_ids': [], 'level': 2, 'source_id': '8dc5707ebe647dedab97db15d0fc82a5e551bbe54a9fb82f79b68b3e3046a3a2', 'page_number': 1, 'split_id': 1, 'split_idx_start': 12})]}
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
                # if it's only one document skip
                if len(child_docs) == 1:
                    next_level_nodes.append(doc)
                    continue
                for child_doc in child_docs:
                    child_doc = self._add_meta_data(child_doc)
                    child_doc.meta["level"] = doc.meta["level"] + 1
                    child_doc.meta["block_size"] = block
                    child_doc.meta["parent_id"] = doc.id
                    all_docs.append(child_doc)
                    doc.meta["children_ids"].append(child_doc.id)
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
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchicalDocumentBuilder":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary to deserialize and create the component.

        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)
