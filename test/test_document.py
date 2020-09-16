from haystack import Document


def test_document_data_access():
    doc = Document(text="test")
    assert doc.text == "test"
