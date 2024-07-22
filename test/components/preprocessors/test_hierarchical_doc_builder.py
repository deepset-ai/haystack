from haystack import Document
from haystack.components.preprocessors.hierarchical_doc_builder import HierarchicalDocumentBuilder


def test_build():
    from haystack import Document
    from haystack.components.preprocessors.hierarchical_doc_builder import HierarchicalDocumentBuilder

    builder = HierarchicalDocumentBuilder(block_sizes=[10, 5, 2])
    text = "one two three four five six seven eight nine ten"

    doc = Document(content=text)
    docs = builder.build_hierarchy_from_doc(doc)
