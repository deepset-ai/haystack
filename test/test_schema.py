from haystack import Document
from haystack import Label
from haystack import Answer

def test_equal_label():
    label1 = Label(query="some",
                   answer=Answer(answer="an answer",type="extractive", score=0.1, document_id=123),
                   document=Document(content="some text", content_type="text"),
                   is_correct_answer=True,
                   is_correct_document=True,
                   origin="user-feedback")
    label2 = Label(query="some",
                   answer=Answer(answer="annother answer", type="extractive", score=0.1, document_id=123),
                   document=Document(content="some text", content_type="text"),
                   is_correct_answer = True,
                   is_correct_document = True,
                   origin = "user-feedback")

    label3 = Label(query="some",
                   answer=Answer(answer="an answer",type="extractive", score=0.1, document_id=123),
                   document=Document(content="some text", content_type="text"),
                   is_correct_answer = True,
                   is_correct_document = True,
                   origin = "user-feedback")

    assert label3 == label1
    assert label2 != label1


def test_generate_doc_id_using_text():
    text1 = "text1"
    text2 = "text2"
    doc1_text1 = Document(content=text1, meta={"name": "doc1"})
    doc2_text1 = Document(content=text1, meta={"name": "doc2"})
    doc3_text2 = Document(content=text2, meta={"name": "doc3"})

    assert doc1_text1.id == doc2_text1.id
    assert doc1_text1.id != doc3_text2.id


def test_generate_doc_id_using_custom_list():
    text1 = "text1"
    text2 = "text2"

    doc1_text1 = Document(content=text1, meta={"name": "doc1"}, id_hash_keys=["key1", text1])
    doc2_text1 = Document(content=text1, meta={"name": "doc2"}, id_hash_keys=["key1", text1])
    doc3_text2 = Document(content=text2, meta={"name": "doc3"}, id_hash_keys=["key1", text2])

    assert doc1_text1.id == doc2_text1.id
    assert doc1_text1.id != doc3_text2.id
