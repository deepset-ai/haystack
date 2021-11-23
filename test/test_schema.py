from haystack.schema import Document, Label, Answer, Span
import numpy as np

LABELS = [
    Label(query="some",
                   answer=Answer(answer="an answer",type="extractive", score=0.1, document_id="123", offsets_in_document=[Span(start=1, end=3)]),
                   document=Document(content="some text", content_type="text"),
                   is_correct_answer=True,
                   is_correct_document=True,
                   origin="user-feedback"),
    Label(query="some",
                   answer=Answer(answer="annother answer", type="extractive", score=0.1, document_id="123"),
                   document=Document(content="some text", content_type="text"),
                   is_correct_answer = True,
                   is_correct_document = True,
                   origin = "user-feedback"),

    Label(query="some",
                   answer=Answer(answer="an answer",type="extractive", score=0.1, document_id="123", offsets_in_document=[Span(start=1, end=3)]),
                   document=Document(content="some text", content_type="text"),
                   is_correct_answer = True,
                   is_correct_document = True,
                   origin = "user-feedback")]


def test_no_answer_label():
    labels = [
        Label(
            query="question",
            answer=Answer(answer=""),
            is_correct_answer=True,
            is_correct_document=True,
            document=Document(content="some", id="777"),
            origin="gold-label",
        ),
        Label(
            query="question",
            answer=Answer(answer=""),
            is_correct_answer=True,
            is_correct_document=True,
            document=Document(content="some", id="777"),
            no_answer=True,
            origin="gold-label",
        ),
        Label(
            query="question",
            answer=Answer(answer="some"),
            is_correct_answer=True,
            is_correct_document=True,
            document=Document(content="some", id="777"),
            origin="gold-label",
        ),
        Label(
            query="question",
            answer=Answer(answer="some"),
            is_correct_answer=True,
            is_correct_document=True,
            document=Document(content="some", id="777"),
            no_answer=False,
            origin="gold-label",
        )
    ]

    assert labels[0].no_answer == True
    assert labels[1].no_answer == True
    assert labels[2].no_answer == False
    assert labels[3].no_answer == False


def test_equal_label():
    assert LABELS[2] == LABELS[0]
    assert LABELS[1] != LABELS[0]


def test_answer_to_json():
    a = Answer(answer="an answer",type="extractive", score=0.1, context="abc",
               offsets_in_document=[Span(start=1, end=10)],
               offsets_in_context=[Span(start=3, end=5)],
               document_id="123")
    j = a.to_json()
    assert type(j) == str
    assert len(j) > 30
    a_new = Answer.from_json(j)
    assert type(a_new.offsets_in_document[0]) == Span
    assert a_new == a


def test_answer_to_dict():
    a = Answer(answer="an answer",type="extractive", score=0.1, context="abc",
               offsets_in_document=[Span(start=1, end=10)],
               offsets_in_context=[Span(start=3, end=5)],
               document_id="123")
    j = a.to_dict()
    assert type(j) == dict
    a_new = Answer.from_dict(j)
    assert type(a_new.offsets_in_document[0]) == Span
    assert a_new == a


def test_label_to_json():
    j0 = LABELS[0].to_json()
    l_new = Label.from_json(j0)
    assert l_new == LABELS[0]


def test_label_to_json():
    j0 = LABELS[0].to_json()
    l_new = Label.from_json(j0)
    assert l_new == LABELS[0]
    assert l_new.answer.offsets_in_document[0].start == 1


def test_label_to_dict():
    j0 = LABELS[0].to_dict()
    l_new = Label.from_dict(j0)
    assert l_new == LABELS[0]
    assert l_new.answer.offsets_in_document[0].start == 1

def test_doc_to_json():
    # With embedding
    d = Document(content="some text", content_type="text", score=0.99988, meta={"name": "doc1"},
                 embedding=np.random.rand(768).astype(np.float32))
    j0 = d.to_json()
    d_new = Document.from_json(j0)
    assert d == d_new

    # No embedding
    d = Document(content="some text", content_type="text", score=0.99988, meta={"name": "doc1"},
                 embedding=None)
    j0 = d.to_json()
    d_new = Document.from_json(j0)
    assert d == d_new


def test_answer_postinit():
    a = Answer(answer="test", offsets_in_document=[{"start": 10, "end": 20}])
    assert a.meta == {}
    assert isinstance(a.offsets_in_document[0], Span)

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
