from haystack import Document
from haystack.dataclasses.hierarchical_document import HierarchicalDocument


def test_init():
    doc = HierarchicalDocument(Document(content="test text"))
    assert doc.parent_id is None
    assert doc.children_ids == []
    assert doc.level == 0
    assert doc.block_size == 0
