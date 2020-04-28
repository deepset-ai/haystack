from haystack.database.base import Document


def test_document_data_access():
    doc = Document(id=1, text="test")
    assert doc.text == "test"
    assert doc['text'] == "test"
