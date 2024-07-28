import pytest

from haystack import Document
from haystack.components.preprocessors import EmptyDocumentRemover


def test_empty_document_remover_run_method():
    documents = [
        Document(content="Hello World!"),
        Document(content=""),
        Document(content=None),
        Document(content="Content"),
    ]

    empty_document_remover = EmptyDocumentRemover()

    result = empty_document_remover.run(documents=documents)

    assert "documents" in result
    assert result["documents"] == [Document(content="Hello World!"), Document(content=""), Document(content="Content")]
