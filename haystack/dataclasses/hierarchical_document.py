from typing import List, Optional

from haystack import Document


class HierarchicalDocument(Document):
    """
    A HierarchicalDocument is a Document that has been split into multiple blocks of different sizes.

    This class is a subclass of Document, it only adds additional attributes to represent the hierarchical structure.

    It holds the hierarchical tree structure when a Document is split into multiple bocks where each smaller block
    is a child of a previous larger block. Each document/block has a parent and children.

    The exceptions are the root block, which is the original document, and the leaf blocks, which are the
    smallest blocks and have no children.
    """

    def __init__(self, document: Document):
        super().__init__()
        self.parent_id: Optional[str] = None
        self.children_ids: List[str] = []
        self.level: int = 0
        self.block_size: int = 0
        self._copy_from_document(document)

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
        fields.append(f"parent_id: {self.parent_id}")
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
