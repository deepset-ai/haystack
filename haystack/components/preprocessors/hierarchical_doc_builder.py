from typing import List

from haystack import Document, component
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses.document_hierarchical import HierarchicalDocument


class HierarchicalDocumentBuilder:
    """
    Splits a documents into different block sizes retaining the hierarchical structure.

    The root node is the original document, and the leaf nodes are the smallest blocks. The smaller blocks are c
    connected to the parent-larger blocks, which are connected to the original document.
    """

    def __init__(self, block_sizes: List[int]):
        if list(set(block_sizes)) != block_sizes:
            raise ValueError("block_sizes must not contain duplicates")
        self.block_sizes = sorted(set(block_sizes), reverse=True)

    @component.output_types(documents=List[HierarchicalDocument])
    def run(self, documents: List[Document]):
        """
        Build a hierarchical document structure from a list of documents.

        :param documents: List of Documents to split into hierarchical blocks.
        :return: List of HierarchicalDocument
        """
        hierarchical_docs = []
        for doc in documents:
            hierarchical_docs.extend(self.build_hierarchy_from_doc(doc))
        return hierarchical_docs

    @staticmethod
    def _split_doc(doc: Document, block_size: int) -> List[HierarchicalDocument]:
        """
        Split a document into multiple documents with a fixed block size.
        """
        splitter = DocumentSplitter(split_length=block_size, split_overlap=0, split_by="word")
        split_docs = splitter.run([doc])
        return split_docs["documents"]

    def build_hierarchy_from_doc(self, document: Document) -> List[HierarchicalDocument]:
        """
        Build a hierarchical tree document structure from a single document.
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
