from haystack.schema import Document, Label, Answer, Span, MultiLabel, SpeechDocument, SpeechAnswer
import pytest
import numpy as np

from ..conftest import SAMPLES_PATH

LABELS = [
    Label(
        query="some",
        answer=Answer(
            answer="an answer",
            type="extractive",
            score=0.1,
            document_id="123",
            offsets_in_document=[Span(start=1, end=3)],
        ),
        document=Document(content="some text", content_type="text"),
        is_correct_answer=True,
        is_correct_document=True,
        origin="user-feedback",
    ),
    Label(
        query="some",
        answer=Answer(answer="annother answer", type="extractive", score=0.1, document_id="123"),
        document=Document(content="some text", content_type="text"),
        is_correct_answer=True,
        is_correct_document=True,
        origin="user-feedback",
    ),
    Label(
        query="some",
        answer=Answer(
            answer="an answer",
            type="extractive",
            score=0.1,
            document_id="123",
            offsets_in_document=[Span(start=1, end=3)],
        ),
        document=Document(content="some text", content_type="text"),
        is_correct_answer=True,
        is_correct_document=True,
        origin="user-feedback",
    ),
]


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
        ),
    ]

    assert labels[0].no_answer == True
    assert labels[1].no_answer == True
    assert labels[2].no_answer == False
    assert labels[3].no_answer == False


def test_equal_label():
    assert LABELS[2] == LABELS[0]
    assert LABELS[1] != LABELS[0]


def test_answer_to_json():
    a = Answer(
        answer="an answer",
        type="extractive",
        score=0.1,
        context="abc",
        offsets_in_document=[Span(start=1, end=10)],
        offsets_in_context=[Span(start=3, end=5)],
        document_id="123",
    )
    j = a.to_json()
    assert type(j) == str
    assert len(j) > 30
    a_new = Answer.from_json(j)
    assert type(a_new.offsets_in_document[0]) == Span
    assert a_new == a


def test_answer_to_dict():
    a = Answer(
        answer="an answer",
        type="extractive",
        score=0.1,
        context="abc",
        offsets_in_document=[Span(start=1, end=10)],
        offsets_in_context=[Span(start=3, end=5)],
        document_id="123",
    )
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
    d = Document(
        content="some text",
        content_type="text",
        score=0.99988,
        meta={"name": "doc1"},
        embedding=np.random.rand(768).astype(np.float32),
    )
    j0 = d.to_json()
    d_new = Document.from_json(j0)
    assert d == d_new

    # No embedding
    d = Document(content="some text", content_type="text", score=0.99988, meta={"name": "doc1"}, embedding=None)
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

    doc1_meta1_id_by_content = Document(content=text1, meta={"name": "doc1"}, id_hash_keys=["content"])
    doc1_meta2_id_by_content = Document(content=text1, meta={"name": "doc2"}, id_hash_keys=["content"])
    assert doc1_meta1_id_by_content.id == doc1_meta2_id_by_content.id

    doc1_meta1_id_by_content_and_meta = Document(content=text1, meta={"name": "doc1"}, id_hash_keys=["content", "meta"])
    doc1_meta2_id_by_content_and_meta = Document(content=text1, meta={"name": "doc2"}, id_hash_keys=["content", "meta"])
    assert doc1_meta1_id_by_content_and_meta.id != doc1_meta2_id_by_content_and_meta.id

    doc1_text1 = Document(content=text1, meta={"name": "doc1"}, id_hash_keys=["content"])
    doc3_text2 = Document(content=text2, meta={"name": "doc3"}, id_hash_keys=["content"])
    assert doc1_text1.id != doc3_text2.id

    with pytest.raises(ValueError):
        _ = Document(content=text1, meta={"name": "doc1"}, id_hash_keys=["content", "non_existing_field"])


def test_aggregate_labels_with_labels():
    label1_with_filter1 = Label(
        query="question",
        answer=Answer(answer="1"),
        is_correct_answer=True,
        is_correct_document=True,
        document=Document(content="some", id="777"),
        origin="gold-label",
        filters={"name": ["filename1"]},
    )
    label2_with_filter1 = Label(
        query="question",
        answer=Answer(answer="2"),
        is_correct_answer=True,
        is_correct_document=True,
        document=Document(content="some", id="777"),
        origin="gold-label",
        filters={"name": ["filename1"]},
    )
    label3_with_filter2 = Label(
        query="question",
        answer=Answer(answer="2"),
        is_correct_answer=True,
        is_correct_document=True,
        document=Document(content="some", id="777"),
        origin="gold-label",
        filters={"name": ["filename2"]},
    )
    label = MultiLabel(labels=[label1_with_filter1, label2_with_filter1])
    assert label.filters == {"name": ["filename1"]}
    with pytest.raises(ValueError):
        label = MultiLabel(labels=[label1_with_filter1, label3_with_filter2])


def test_multilabel_id():
    query1 = "question 1"
    query2 = "question 2"
    document1 = Document(content="something", id="1")
    document2 = Document(content="something else", id="2")
    answer1 = Answer(answer="answer 1")
    answer2 = Answer(answer="answer 2")
    filter1 = {"name": ["name 1"]}
    filter2 = {"name": ["name 1"], "author": ["author 1"]}
    label1a = Label(
        query=query1,
        document=document1,
        is_correct_answer=True,
        is_correct_document=True,
        origin="gold-label",
        answer=answer1,
        filters=filter1,
    )
    label1b = Label(
        query=query1,
        document=document2,
        is_correct_answer=False,
        is_correct_document=False,
        origin="gold-label",
        answer=answer2,
        filters=filter1,
    )
    label2a = Label(
        query=query2,
        document=document1,
        is_correct_answer=True,
        is_correct_document=True,
        origin="gold-label",
        answer=answer1,
        filters=filter2,
    )
    label2b = Label(
        query=query2,
        document=document1,
        is_correct_answer=True,
        is_correct_document=True,
        origin="user-feedback",
        answer=answer1,
        filters=filter2,
    )
    label2c = Label(
        query=query2,
        document=document1,
        is_correct_answer=False,
        is_correct_document=True,
        origin="user-feedback",
        answer=answer2,
        filters=filter2,
    )
    label3a = Label(
        query=query1,
        document=document1,
        is_correct_answer=True,
        is_correct_document=True,
        origin="gold-label",
        answer=answer1,
        filters=filter2,
    )
    multilabel_init_kwargs = [
        [
            {"labels": [label1a]},
            {"labels": [label1a, label1b]},
            {"labels": [label1a, label1b], "drop_negative_labels": True},  # label1b will be droped
        ],
        [
            {"labels": [label2a]},
            {"labels": [label2a, label2b]},
            {"labels": [label2a, label2b, label2c]},
            {"labels": [label2a, label2b, label2c], "drop_negative_labels": True},  # label2c will be droped
        ],
        [{"labels": [label3a]}],
    ]

    # any pair of MultiLabel instances with same query and filters should have the same generated id
    for multilabel_init_kwargs_1 in multilabel_init_kwargs:
        for multilabel_init_kwargs_a in multilabel_init_kwargs_1:
            for multilabel_init_kwargs_b in multilabel_init_kwargs_1:
                assert MultiLabel(**multilabel_init_kwargs_a).id == MultiLabel(**multilabel_init_kwargs_b).id

    # any pair of MultiLabel instances with different query or filters should have a different generated id
    for multilabel_init_kwargs_1 in multilabel_init_kwargs:
        for multilabel_init_kwargs_2 in multilabel_init_kwargs:
            if multilabel_init_kwargs_1 != multilabel_init_kwargs_2:
                for multilabel_init_kwargs_a in multilabel_init_kwargs_1:
                    for multilabel_init_kwargs_b in multilabel_init_kwargs_2:
                        assert MultiLabel(**multilabel_init_kwargs_a).id != MultiLabel(**multilabel_init_kwargs_b).id


def test_serialize_speech_document():
    speech_doc = SpeechDocument(
        id=12345,
        content_type="audio",
        content="this is the content of the document",
        content_audio=SAMPLES_PATH / "audio" / "this is the content of the document.wav",
        meta={"some": "meta"},
    )
    speech_doc_dict = speech_doc.to_dict()

    assert speech_doc_dict["content"] == "this is the content of the document"
    assert speech_doc_dict["content_audio"] == str(
        (SAMPLES_PATH / "audio" / "this is the content of the document.wav").absolute()
    )


def test_deserialize_speech_document():
    speech_doc = SpeechDocument(
        id=12345,
        content_type="audio",
        content="this is the content of the document",
        content_audio=SAMPLES_PATH / "audio" / "this is the content of the document.wav",
        meta={"some": "meta"},
    )
    assert speech_doc == SpeechDocument.from_dict(speech_doc.to_dict())


def test_serialize_speech_answer():
    speech_answer = SpeechAnswer(
        answer="answer",
        answer_audio=SAMPLES_PATH / "audio" / "answer.wav",
        context="the context for this answer is here",
        context_audio=SAMPLES_PATH / "audio" / "the context for this answer is here.wav",
    )
    speech_answer_dict = speech_answer.to_dict()

    assert speech_answer_dict["answer"] == "answer"
    assert speech_answer_dict["answer_audio"] == str((SAMPLES_PATH / "audio" / "answer.wav").absolute())
    assert speech_answer_dict["context"] == "the context for this answer is here"
    assert speech_answer_dict["context_audio"] == str(
        (SAMPLES_PATH / "audio" / "the context for this answer is here.wav").absolute()
    )


def test_deserialize_speech_answer():
    speech_answer = SpeechAnswer(
        answer="answer",
        answer_audio=SAMPLES_PATH / "audio" / "answer.wav",
        context="the context for this answer is here",
        context_audio=SAMPLES_PATH / "audio" / "the context for this answer is here.wav",
    )
    assert speech_answer == SpeechAnswer.from_dict(speech_answer.to_dict())
