from dataclasses import dataclass
from typing import List, Optional

from haystack import Document, component
from haystack.components.preprocessors import DocumentSplitter


@dataclass
class HierarchicalDocument(Document):
    """
    A HierarchicalDocument is a Document that has been split into multiple blocks of different sizes.

    It's used to represent a hierarchical tree structure when a Document is split into multiple bocks where each block
    is a child of the previous block. Each document/block has a parent and children. The exception are the root block,
    which is the original document, and the leaf blocks, which are the smallest blocks.
    """

    def __init__(self, document: Document):
        super().__init__()
        self.parent_id: Optional[str] = None
        self.children_ids: List[str] = []
        self.level: int = 0
        self.block_size: int = 0

    def __repr__(self):
        fields = []
        if self.content is not None:
            fields.append(
                f"content: '{self.content}'" if len(self.content) < 100 else f"content: '{self.content[:100]}...'"
            )
        if self.dataframe is not None:
            fields.append(f"dataframe: {self.dataframe.shape}")
        if self.blob is not None:
            fields.append(f"blob: {len(self.blob.data)} bytes")
        if len(self.meta) > 0:
            fields.append(f"meta: {self.meta}")
        if self.score is not None:
            fields.append(f"score: {self.score}")
        if self.embedding is not None:
            fields.append(f"embedding: vector of size {len(self.embedding)}")
        if self.sparse_embedding is not None:
            fields.append(f"sparse_embedding: vector with {len(self.sparse_embedding.indices)} non-zero elements")

        fields.append(f"children: {self.children_ids}")
        fields.append(f"level: {self.level}")
        fields.append(f"block_size: {self.block_size}")
        fields.append(f"parent: {self.parent_id}")
        fields_str = ", ".join(fields)

        return f"{self.__class__.__name__}(id={self.id}, {fields_str})"

    def _copy_from_document(self, document: Document):
        """
        Copy attributes from a Document to this HierarchicalDoc.
        """
        self.id = document.id
        self.content = document.content
        self.dataframe = document.dataframe
        self.blob = document.blob
        self.meta = document.meta
        self.score = document.score
        self.embedding = document.embedding
        self.sparse_embedding = document.sparse_embedding


class HierarchicalDocumentBuilder:
    """
    Splits a documents into different block sizes retaining the hierarchical structure.

    The root node is the original document, and the leaf nodes are the smallest blocks. The smaller blocks are c
    connected to the parent-larger blocks, which are connected to the original document.
    """

    def __init__(self, block_sizes: List[int]):
        self.block_sizes = sorted(block_sizes, reverse=True)

    @component.output_types(documents=List[HierarchicalDocument])
    def run(self, documents: List[Document]):
        """
        Build a hierarchical document structure.
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
