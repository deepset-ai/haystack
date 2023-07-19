import pytest

from haystack.preview import Document
from haystack.preview.document_stores import MemoryDocumentStore
from haystack.preview.document_stores.writer import WriteToStore


@pytest.mark.unit
def test_writer_returns_list_of_documents(self):
    docstore = MemoryDocumentStore()
    writer = WriteToStore()
    documents = [
        Document(content="This is the text of the document."),
        Document(content="This is the text of another document."),
    ]
    writer.run(data=documents)
    assert docstore.count_documents() == 2
    assert sorted(docstore.filter_documents()) == sorted(documents)
