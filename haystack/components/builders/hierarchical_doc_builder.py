# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Literal

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses.hierarchical_document import HierarchicalDocument


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
    >> {'documents': [HierarchicalDocument(id=06dc21e637ae25d526f36b3d456edf035ec6df7efb575092c54c6bd86c0061b7, content: 'This is a test document', meta: {'source_id': '2594f015eb8eca9dd088d8e7ecc9b25e16681842815088a7535566d6ac38cdcf', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}, children: ['5b6548b18d2d507c4c035f5d64a4f4a103d07f9f75b828ecffa72018d7fd2309'], level: 1, block_size: 10, parent_id: 2594f015eb8eca9dd088d8e7ecc9b25e16681842815088a7535566d6ac38cdcf),
    >> HierarchicalDocument(id=5b6548b18d2d507c4c035f5d64a4f4a103d07f9f75b828ecffa72018d7fd2309, content: 'This is a test document', meta: {'source_id': '06dc21e637ae25d526f36b3d456edf035ec6df7efb575092c54c6bd86c0061b7', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}, children: ['e795be3a1a9aa7429c4c78bbd1d63f2a8772b30ffd8464e1d43ba4dd0903511a', '0fa536df594f11d15d385b657ff176b7f3442fcc4ba90f3862f03432df99b9b3', 'c6c61c2ab43e75950cc08d2af3347598c28ab712dee58e3ef50cbab64c20f81b'], level: 2, block_size: 5, parent_id: 06dc21e637ae25d526f36b3d456edf035ec6df7efb575092c54c6bd86c0061b7),
    >> HierarchicalDocument(id=e795be3a1a9aa7429c4c78bbd1d63f2a8772b30ffd8464e1d43ba4dd0903511a, content: 'This is ', meta: {'source_id': '5b6548b18d2d507c4c035f5d64a4f4a103d07f9f75b828ecffa72018d7fd2309', 'page_number': 1, 'split_id': 0, 'split_idx_start': 0}, children: [], level: 3, block_size: 2, parent_id: 5b6548b18d2d507c4c035f5d64a4f4a103d07f9f75b828ecffa72018d7fd2309),
    >> HierarchicalDocument(id=0fa536df594f11d15d385b657ff176b7f3442fcc4ba90f3862f03432df99b9b3, content: 'a test ', meta: {'source_id': '5b6548b18d2d507c4c035f5d64a4f4a103d07f9f75b828ecffa72018d7fd2309', 'page_number': 1, 'split_id': 1, 'split_idx_start': 8}, children: [], level: 3, block_size: 2, parent_id: 5b6548b18d2d507c4c035f5d64a4f4a103d07f9f75b828ecffa72018d7fd2309),
    >> HierarchicalDocument(id=c6c61c2ab43e75950cc08d2af3347598c28ab712dee58e3ef50cbab64c20f81b, content: 'document', meta: {'source_id': '5b6548b18d2d507c4c035f5d64a4f4a103d07f9f75b828ecffa72018d7fd2309', 'page_number': 1, 'split_id': 2, 'split_idx_start': 15}, children: [], level: 3, block_size: 2, parent_id: 5b6548b18d2d507c4c035f5d64a4f4a103d07f9f75b828ecffa72018d7fd2309)]}
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

    @component.output_types(documents=List[HierarchicalDocument])
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

    def build_hierarchy_from_doc(self, document: Document) -> List[HierarchicalDocument]:
        """
        Build a hierarchical tree document structure from a single document.

        Given a document, this function splits the document into hierarchical blocks of different sizes represented
        as HierarchicalDocument objects

        :param document: Document to split into hierarchical blocks.
        :return:
            List of HierarchicalDocument
        """

        root = HierarchicalDocument(document)
        current_level_nodes = [root]
        all_docs = []

        for block in self.block_sizes:
            next_level_nodes = []
            for doc in current_level_nodes:
                child_docs = self._split_doc(doc, block)
                for child_doc in child_docs:
                    hierarchical_child_doc = HierarchicalDocument(child_doc)
                    hierarchical_child_doc.level = doc.level + 1
                    hierarchical_child_doc.block_size = block
                    hierarchical_child_doc.parent_id = doc.id
                    all_docs.append(hierarchical_child_doc)
                    doc.children_ids.append(hierarchical_child_doc.id)
                    next_level_nodes.append(hierarchical_child_doc)
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
