from typing import Any, Dict, List, Literal

from haystack import Document, component, default_to_dict
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses.hierarchical_document import HierarchicalDocument


@component
class HierarchicalDocumentBuilder:
    """
    Splits a documents into different block sizes building a hierarchical tree structure of blocks of different sizes.

    The root node is the original document, the leaf nodes are the smallest blocks. The blocks in between are connected
    such that the smaller blocks are children of the parent-larger blocks.

    ### Usage examples

    #### On its own

    """

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

    def _split_doc(self, doc: Document, block_size: int):
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
    def from_dict(cls, init_parameters):
        """
        Load a HierarchicalDocumentBuilder from a dictionary.

        :param init_parameters:
        :returns:
            HierarchicalDocumentBuilder object
        """
        return HierarchicalDocumentBuilder(**init_parameters)
