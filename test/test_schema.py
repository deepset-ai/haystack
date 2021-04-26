from haystack import Document


def test_generate_doc_id_using_text():
    text1 = "text1"
    text2 = "text2"
    doc1_text1 = Document(text=text1, meta={"name": "doc1"})
    doc2_text1 = Document(text=text1, meta={"name": "doc2"})
    doc3_text2 = Document(text=text2, meta={"name": "doc3"})

    assert doc1_text1.id == doc2_text1.id
    assert doc1_text1.id != doc3_text2.id


def test_generate_doc_id_using_custom_list():
    text1 = "text1"
    text2 = "text2"

    doc1_text1_key1 = Document(text=text1, meta={"name": "doc1"}, id_hash_keys=["key1"])
    doc2_text1_key2 = Document(text=text1, meta={"name": "doc2"}, id_hash_keys=["key2"])
    doc3_text2_key1 = Document(text=text2, meta={"name": "doc3"}, id_hash_keys=["key1"])

    assert doc1_text1_key1.id != doc2_text1_key2.id
    assert doc1_text1_key1.id == doc3_text2_key1.id
