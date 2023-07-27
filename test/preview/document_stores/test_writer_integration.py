import pytest

from haystack.preview import Document, Pipeline
from haystack.preview.components.writers.document_writer import DocumentWriter
from haystack.preview.document_stores import MemoryDocumentStore


@pytest.mark.integration
def test_writer_adds_documents_to_store():
    ds = MemoryDocumentStore()
    writer = DocumentWriter()

    pipeline = Pipeline()
    pipeline.add_store("memory", ds)
    pipeline.add_component("writer", writer)

    documents = [
        Document(content="This is the text of a document."),
        Document(content="This is the text of another document."),
    ]
    pipeline.run(data={"writer": DocumentWriter.Input(documents=documents)})
    assert sorted(ds.filter_documents()) == sorted(documents)
