from haystack import Document
from haystack.dataclasses.hierarchical_document import HierarchicalDocument


def test_init():
    doc = HierarchicalDocument(Document(content="test text"))
    assert doc.parent_id is None
    assert doc.children_ids == []
    assert doc.level == 0
    assert doc.block_size == 0


def test_repr():
    doc = HierarchicalDocument(Document(content="test text"))
    assert "HierarchicalDocument" in doc.__repr__()
    assert "content: 'test text'" in doc.__repr__()
    assert "children: []" in doc.__repr__()
    assert "level: 0" in doc.__repr__()
    assert "block_size: 0" in doc.__repr__()
    assert "parent_id: None" in doc.__repr__()
