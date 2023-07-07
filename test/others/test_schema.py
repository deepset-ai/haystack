import json

from haystack.schema import Document, EvaluationResult, Label, Answer, Span, MultiLabel, TableCell, _dict_factory
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def text_label():
    return Label(
        query="some",
        answer=Answer(
            answer="an answer",
            type="extractive",
            score=0.1,
            document_ids=["doc_1"],
            offsets_in_document=[Span(start=1, end=3)],
        ),
        document=Document(content="some text", content_type="text", id="doc_1"),
        is_correct_answer=True,
        is_correct_document=True,
        origin="user-feedback",
        created_at="2023-05-02 11:43:56",
        id="text_label_1",
    )


@pytest.fixture
def text_label_dict():
    return {
        "id": "text_label_1",
        "query": "some",
        "document": {
            "id": "doc_1",
            "content": "some text",
            "content_type": "text",
            "meta": {},
            "id_hash_keys": ["content"],
            "score": None,
            "embedding": None,
        },
        "is_correct_answer": True,
        "is_correct_document": True,
        "origin": "user-feedback",
        "answer": {
            "answer": "an answer",
            "type": "extractive",
            "score": 0.1,
            "context": None,
            "offsets_in_document": [{"start": 1, "end": 3}],
            "offsets_in_context": None,
            "document_ids": ["doc_1"],
            "meta": {},
        },
        "pipeline_id": None,
        "created_at": "2023-05-02 11:43:56",
        "updated_at": None,
        "meta": {},
        "filters": None,
    }


@pytest.fixture
def text_label_json(samples_path):
    with open(samples_path / "schema" / "text_label.json") as f1:
        data = json.load(f1)
    return data


@pytest.fixture
def table_label():
    return Label(
        query="some",
        answer=Answer(
            answer="text_2",
            type="extractive",
            score=0.1,
            document_ids=["123"],
            context=pd.DataFrame.from_records([{"col1": "text_1", "col2": 1}, {"col1": "text_2", "col2": 2}]),
            offsets_in_document=[TableCell(row=1, col=0)],
        ),
        document=Document(
            content=pd.DataFrame.from_records([{"col1": "text_1", "col2": 1}, {"col1": "text_2", "col2": 2}]),
            content_type="table",
            id="fe5cb68f8226776914781f6bd40ad718",
        ),
        is_correct_answer=True,
        is_correct_document=True,
        origin="user-feedback",
        created_at="2023-05-02 11:43:56",
        updated_at=None,
        id="fbd79f71-d690-4b21-bd0a-1094292b9809",
    )


@pytest.fixture
def table_label_dict():
    return {
        "id": "fbd79f71-d690-4b21-bd0a-1094292b9809",
        "query": "some",
        "document": {
            "id": "fe5cb68f8226776914781f6bd40ad718",
            "content": [["col1", "col2"], ["text_1", 1], ["text_2", 2]],
            "content_type": "table",
            "meta": {},
            "id_hash_keys": ["content"],
            "score": None,
            "embedding": None,
        },
        "is_correct_answer": True,
        "is_correct_document": True,
        "origin": "user-feedback",
        "answer": {
            "answer": "text_2",
            "type": "extractive",
            "score": 0.1,
            "context": [["col1", "col2"], ["text_1", 1], ["text_2", 2]],
            "offsets_in_document": [{"row": 1, "col": 0}],
            "offsets_in_context": None,
            "document_ids": ["123"],
            "meta": {},
        },
        "pipeline_id": None,
        "created_at": "2023-05-02 11:43:56",
        "updated_at": None,
        "meta": {},
        "filters": None,
    }


@pytest.fixture
def table_label_json(samples_path):
    with open(samples_path / "schema" / "table_label.json") as f1:
        data = json.load(f1)
    return data


@pytest.fixture
def text_answer():
    return Answer(
        answer="an answer",
        type="extractive",
        score=0.1,
        context="abc",
        offsets_in_document=[Span(start=1, end=10)],
        offsets_in_context=[Span(start=3, end=5)],
        document_ids=["123"],
    )


@pytest.fixture
def text_answer_dict():
    return {
        "answer": "an answer",
        "type": "extractive",
        "score": 0.1,
        "context": "abc",
        "offsets_in_document": [{"start": 1, "end": 10}],
        "offsets_in_context": [{"start": 3, "end": 5}],
        "document_ids": ["123"],
        "meta": {},
    }


@pytest.fixture
def text_answer_json(samples_path):
    with open(samples_path / "schema" / "text_answer.json") as f1:
        data = json.load(f1)
    return data


@pytest.fixture
def table_answer():
    return Answer(
        answer="text_2",
        type="extractive",
        score=0.1,
        context=pd.DataFrame.from_records([{"col1": "text_1", "col2": 1}, {"col1": "text_2", "col2": 2}]),
        offsets_in_document=[TableCell(row=1, col=0)],
        offsets_in_context=[TableCell(row=1, col=0)],
        document_ids=["123"],
    )


@pytest.fixture
def table_answer_dict():
    return {
        "answer": "text_2",
        "type": "extractive",
        "score": 0.1,
        "context": [["col1", "col2"], ["text_1", 1], ["text_2", 2]],
        "offsets_in_document": [{"row": 1, "col": 0}],
        "offsets_in_context": [{"row": 1, "col": 0}],
        "document_ids": ["123"],
        "meta": {},
    }


@pytest.fixture
def table_answer_json(samples_path):
    with open(samples_path / "schema" / "table_answer.json") as f1:
        data = json.load(f1)
    return data


@pytest.fixture
def text_doc():
    return Document(content="some text", content_type="text", id="doc1")


@pytest.fixture
def text_doc_dict():
    return {
        "content": "some text",
        "content_type": "text",
        "score": None,
        "meta": {},
        "id_hash_keys": ["content"],
        "embedding": None,
        "id": "doc1",
    }


@pytest.fixture
def text_doc_json(samples_path):
    with open(samples_path / "schema" / "text_doc.json") as f1:
        json_str = f1.read()
    return json_str


@pytest.fixture
def text_doc_with_embedding():
    return Document(content="some text", content_type="text", id="doc2", embedding=np.array([1.1, 2.2, 3.3, 4.4]))


@pytest.fixture
def text_doc_with_embedding_json(samples_path):
    with open(samples_path / "schema" / "text_doc_emb.json") as f1:
        json_str = f1.read()
    return json_str


@pytest.fixture
def table_doc():
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": [58, 47, 60],
        "number of movies": [87, 53, 69],
        "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
    }
    return Document(content=pd.DataFrame(data), content_type="table", id="doc1")


@pytest.fixture
def table_doc_dict():
    return {
        "content": [
            ["actors", "age", "number of movies", "date of birth"],
            ["brad pitt", 58, 87, "18 december 1963"],
            ["leonardo di caprio", 47, 53, "11 november 1974"],
            ["george clooney", 60, 69, "6 may 1961"],
        ],
        "content_type": "table",
        "score": None,
        "meta": {},
        "id_hash_keys": ["content"],
        "embedding": None,
        "id": "doc1",
    }


@pytest.fixture
def table_doc_json(samples_path):
    with open(samples_path / "schema" / "table_doc.json") as f1:
        json_str = f1.read()
    return json_str


@pytest.fixture
def table_doc_with_embedding():
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": [58, 47, 60],
        "number of movies": [87, 53, 69],
        "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
    }
    return Document(
        content=pd.DataFrame(data), content_type="table", id="doc2", embedding=np.array([1.1, 2.2, 3.3, 4.4])
    )


@pytest.fixture
def table_doc_with_embedding_json(samples_path):
    with open(samples_path / "schema" / "table_doc_emb.json") as f1:
        json_str = f1.read()
    return json_str


@pytest.mark.unit
def test_no_answer_label():
    label_no_answer = Label(
        query="question",
        answer=Answer(answer=""),
        is_correct_answer=True,
        is_correct_document=True,
        document=Document(content="some", id="777"),
        origin="gold-label",
    )
    label_with_answer = Label(
        query="question",
        answer=Answer(answer="some"),
        is_correct_answer=True,
        is_correct_document=True,
        document=Document(content="some", id="777"),
        origin="gold-label",
    )
    assert label_no_answer.no_answer
    assert not label_with_answer.no_answer


@pytest.mark.unit
def test_text_labels_with_identical_fields_are_equal(text_label):
    text_label_copy = Label(
        query="some",
        answer=Answer(
            answer="an answer",
            type="extractive",
            score=0.1,
            document_ids=["doc_1"],
            offsets_in_document=[Span(start=1, end=3)],
        ),
        document=Document(content="some text", content_type="text", id="doc_1"),
        is_correct_answer=True,
        is_correct_document=True,
        origin="user-feedback",
        created_at="2023-05-02 11:43:56",
        id="text_label_1",
    )
    assert text_label == text_label_copy


@pytest.mark.unit
def test_text_labels_with_different_fields_are_not_equal(text_label):
    text_label_different = Label(
        query="some",
        answer=Answer(
            answer="different answer",
            type="extractive",
            score=0.1,
            document_ids=["doc_1"],
            offsets_in_document=[Span(start=5, end=15)],
        ),
        document=Document(content="some text", content_type="text", id="doc_1"),
        is_correct_answer=True,
        is_correct_document=True,
        origin="user-feedback",
        created_at="2023-05-02 11:43:56",
        id="text_label_1",
    )
    assert text_label != text_label_different


@pytest.mark.unit
def test_label_from_json(text_label, text_label_json):
    text_label_from_json = Label.from_json(text_label_json)
    assert text_label_from_json == text_label


@pytest.mark.unit
def test_label_to_json(text_label, text_label_json):
    text_label_to_json = json.loads(text_label.to_json())
    assert text_label_to_json == text_label_json


@pytest.mark.unit
def test_text_label_from_dict(text_label, text_label_dict):
    text_label_from_dict = Label.from_dict(text_label_dict)
    assert text_label_from_dict == text_label


@pytest.mark.unit
def test_text_label_to_dict(text_label, text_label_dict):
    text_label_to_dict = text_label.to_dict()
    assert text_label_to_dict == text_label_dict


@pytest.mark.unit
def test_table_labels_with_identical_fields_are_equal(table_label):
    table_label_copy = Label(
        query="some",
        answer=Answer(
            answer="text_2",
            type="extractive",
            score=0.1,
            document_ids=["123"],
            context=pd.DataFrame.from_records([{"col1": "text_1", "col2": 1}, {"col1": "text_2", "col2": 2}]),
            offsets_in_document=[TableCell(row=1, col=0)],
        ),
        document=Document(
            content=pd.DataFrame.from_records([{"col1": "text_1", "col2": 1}, {"col1": "text_2", "col2": 2}]),
            content_type="table",
        ),
        is_correct_answer=True,
        is_correct_document=True,
        origin="user-feedback",
    )
    assert table_label == table_label_copy


@pytest.mark.unit
def test_table_labels_with_different_fields_are_not_equal(table_label):
    table_label_different = Label(
        query="some",
        answer=Answer(
            answer="text_1",
            type="extractive",
            score=0.1,
            document_ids=["123"],
            context=pd.DataFrame.from_records([{"col1": "text_1", "col2": 1}, {"col1": "text_2", "col2": 2}]),
            offsets_in_document=[TableCell(row=0, col=0)],
        ),
        document=Document(
            content=pd.DataFrame.from_records([{"col1": "text_1", "col2": 1}, {"col1": "text_2", "col2": 2}]),
            content_type="table",
        ),
        is_correct_answer=True,
        is_correct_document=True,
        origin="user-feedback",
    )
    assert table_label != table_label_different


@pytest.mark.unit
def test_table_label_from_json(table_label, table_label_json):
    table_label_from_json = Label.from_json(table_label_json)
    assert table_label_from_json == table_label


@pytest.mark.unit
def test_table_label_to_json(table_label, table_label_json):
    table_label_to_json = json.loads(table_label.to_json())
    assert table_label_to_json == table_label_json


@pytest.mark.unit
def test_table_label_from_dict(table_label, table_label_dict):
    table_label_from_dict = Label.from_dict(table_label_dict)
    assert table_label_from_dict == table_label


@pytest.mark.unit
def test_table_label_to_dict(table_label, table_label_dict):
    table_label_to_dict = table_label.to_dict()
    assert table_label_to_dict == table_label_dict


@pytest.mark.unit
def test_answer_to_json(text_answer, text_answer_json):
    text_answer_to_json = json.loads(text_answer.to_json())
    assert text_answer_to_json == text_answer_json


@pytest.mark.unit
def test_answer_from_json(text_answer, text_answer_json):
    text_answer_from_json = Answer.from_json(text_answer_json)
    assert text_answer_from_json == text_answer


@pytest.mark.unit
def test_answer_to_dict(text_answer, text_answer_dict):
    assert text_answer.to_dict() == text_answer_dict


@pytest.mark.unit
def test_answer_from_dict(text_answer, text_answer_dict):
    assert text_answer == Answer.from_dict(text_answer_dict)


@pytest.mark.unit
def test_table_answer_to_json(table_answer, table_answer_json):
    table_answer_to_json = json.loads(table_answer.to_json())
    assert table_answer_to_json == table_answer_json


@pytest.mark.unit
def test_table_answer_from_json(table_answer, table_answer_json):
    table_answer_from_json = Answer.from_json(table_answer_json)
    assert table_answer_from_json == table_answer


@pytest.mark.unit
def test_table_answer_to_dict(table_answer, table_answer_dict):
    assert table_answer.to_dict() == table_answer_dict


@pytest.mark.unit
def test_table_answer_from_dict(table_answer, table_answer_dict):
    assert table_answer == Answer.from_dict(table_answer_dict)


@pytest.mark.unit
def test_document_from_dict(text_doc, text_doc_dict):
    assert text_doc == Document.from_dict(text_doc_dict)


@pytest.mark.unit
def test_document_to_dict(text_doc, text_doc_dict):
    assert text_doc.to_dict() == text_doc_dict


@pytest.mark.unit
def test_table_document_from_dict(table_doc, table_doc_dict):
    assert table_doc == Document.from_dict(table_doc_dict)


@pytest.mark.unit
def test_table_document_to_dict(table_doc, table_doc_dict):
    assert table_doc.to_dict() == table_doc_dict


@pytest.mark.unit
def test_document_from_json_with_embedding(text_doc_with_embedding, text_doc_with_embedding_json):
    text_doc_emb_from_json = Document.from_json(text_doc_with_embedding_json)
    assert text_doc_with_embedding == text_doc_emb_from_json


@pytest.mark.unit
def test_document_from_json_without_embedding(text_doc, text_doc_json):
    text_doc_no_emb_from_json = Document.from_json(text_doc_json)
    assert text_doc == text_doc_no_emb_from_json


@pytest.mark.unit
def test_document_to_json_with_embedding(text_doc_with_embedding, text_doc_with_embedding_json):
    text_doc_emb_to_json = json.loads(text_doc_with_embedding.to_json())
    assert json.loads(text_doc_with_embedding_json) == text_doc_emb_to_json


@pytest.mark.unit
def test_document_to_json_without_embedding(text_doc, text_doc_json):
    text_doc_no_emb_to_json = json.loads(text_doc.to_json())
    assert json.loads(text_doc_json) == text_doc_no_emb_to_json


@pytest.mark.unit
def test_table_doc_from_json_with_embedding(table_doc_with_embedding, table_doc_with_embedding_json):
    table_doc_emb_from_json = Document.from_json(table_doc_with_embedding_json)
    assert table_doc_with_embedding == table_doc_emb_from_json


@pytest.mark.unit
def test_table_doc_from_json_without_embedding(table_doc, table_doc_json):
    table_doc_no_emb_from_json = Document.from_json(table_doc_json)
    assert table_doc == table_doc_no_emb_from_json


@pytest.mark.unit
def test_table_doc_to_json_with_embedding(table_doc_with_embedding, table_doc_with_embedding_json):
    # With embedding
    table_doc_emb_to_json = json.loads(table_doc_with_embedding.to_json())
    assert json.loads(table_doc_with_embedding_json) == table_doc_emb_to_json


@pytest.mark.unit
def test_table_doc_to_json_without_embedding(table_doc, table_doc_json):
    # No embedding
    table_doc_no_emb_to_json = json.loads(table_doc.to_json())
    assert json.loads(table_doc_json) == table_doc_no_emb_to_json


@pytest.mark.unit
def test_answer_postinit():
    a = Answer(answer="test", offsets_in_document=[{"start": 10, "end": 20}])
    assert a.meta == {}
    assert isinstance(a.offsets_in_document[0], Span)


@pytest.mark.unit
def test_table_answer_postinit():
    table_answer = Answer(answer="test", offsets_in_document=[{"row": 1, "col": 2}])
    assert table_answer.meta == {}
    assert isinstance(table_answer.offsets_in_document[0], TableCell)


@pytest.mark.unit
def test_generate_doc_id_using_text():
    text1 = "text1"
    text2 = "text2"
    doc1_text1 = Document(content=text1, meta={"name": "doc1"})
    doc2_text1 = Document(content=text1, meta={"name": "doc2"})
    doc3_text2 = Document(content=text2, meta={"name": "doc3"})

    assert doc1_text1.id == doc2_text1.id
    assert doc1_text1.id != doc3_text2.id


@pytest.mark.unit
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


@pytest.mark.unit
def test_generate_doc_id_custom_list_meta():
    text1 = "text1"
    text2 = "text2"

    doc1_text1 = Document(
        content=text1, meta={"name": "doc1", "url": "https://deepset.ai"}, id_hash_keys=["content", "meta.url"]
    )
    doc2_text1 = Document(
        content=text1, meta={"name": "doc2", "url": "https://deepset.ai"}, id_hash_keys=["content", "meta.url"]
    )
    assert doc1_text1.id == doc2_text1.id

    doc1_text1 = Document(content=text1, meta={"name": "doc1", "url": "https://deepset.ai"}, id_hash_keys=["meta.url"])
    doc2_text2 = Document(content=text2, meta={"name": "doc2", "url": "https://deepset.ai"}, id_hash_keys=["meta.url"])
    assert doc1_text1.id == doc2_text2.id

    doc1_text1 = Document(content=text1, meta={"name": "doc1", "url": "https://deepset.ai"}, id_hash_keys=["meta.url"])
    doc2_text2 = Document(
        content=text2, meta={"name": "doc2", "url": "https://deepset.ai"}, id_hash_keys=["meta.url", "meta.name"]
    )
    assert doc1_text1.id != doc2_text2.id


@pytest.mark.unit
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
        _ = MultiLabel(labels=[label1_with_filter1, label3_with_filter2])


@pytest.mark.unit
def test_multilabel_preserve_order():
    labels = [
        Label(
            id="0",
            query="question",
            answer=Answer(answer="answer1", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="1",
            query="question",
            answer=Answer(answer="answer2", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="2",
            query="question",
            answer=Answer(answer="answer3", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some other", id="333"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="3",
            query="question",
            answer=Answer(answer="", offsets_in_document=[Span(start=0, end=0)]),
            document=Document(content="some", id="777"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="4",
            query="question",
            answer=Answer(answer="answer5", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=False,
            is_correct_document=True,
            origin="gold-label",
        ),
    ]

    multilabel = MultiLabel(labels=labels)

    for i in range(0, 5):
        assert multilabel.labels[i].id == str(i)


@pytest.mark.unit
def test_multilabel_preserve_order_w_duplicates():
    labels = [
        Label(
            id="0",
            query="question",
            answer=Answer(
                answer="answer1",
                offsets_in_document=[Span(start=12, end=18)],
                offsets_in_context=[Span(start=1, end=7)],
            ),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="1",
            query="question",
            answer=Answer(
                answer="answer2",
                offsets_in_document=[Span(start=10, end=16)],
                offsets_in_context=[Span(start=0, end=6)],
            ),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="2",
            query="question",
            answer=Answer(
                answer="answer3",
                offsets_in_document=[Span(start=14, end=20)],
                offsets_in_context=[Span(start=2, end=8)],
            ),
            document=Document(content="some other", id="333"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="0",
            query="question",
            answer=Answer(
                answer="answer1",
                offsets_in_document=[Span(start=12, end=18)],
                offsets_in_context=[Span(start=1, end=7)],
            ),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="2",
            query="question",
            answer=Answer(
                answer="answer3",
                offsets_in_document=[Span(start=14, end=20)],
                offsets_in_context=[Span(start=2, end=8)],
            ),
            document=Document(content="some other", id="333"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
    ]

    multilabel = MultiLabel(labels=labels)

    assert multilabel.query == "question"
    assert multilabel.answers == ["answer1", "answer2", "answer3"]
    assert multilabel.document_ids == ["123", "123", "333"]
    assert multilabel.contexts == ["some", "some", "some other"]
    assert multilabel.offsets_in_documents == [
        {"start": 12, "end": 18},
        {"start": 10, "end": 16},
        {"start": 14, "end": 20},
    ]
    assert multilabel.offsets_in_contexts == [{"start": 1, "end": 7}, {"start": 0, "end": 6}, {"start": 2, "end": 8}]

    for i in range(0, 3):
        assert multilabel.labels[i].id == str(i)


@pytest.mark.unit
def test_multilabel_id():
    query1 = "question 1"
    query2 = "question 2"
    document1 = Document(content="something", id="1")
    answer1 = Answer(answer="answer 1")
    filter1 = {"name": ["name 1"]}
    filter2 = {"name": ["name 1"], "author": ["author 1"]}
    label1 = Label(
        query=query1,
        document=document1,
        is_correct_answer=True,
        is_correct_document=True,
        origin="gold-label",
        answer=answer1,
        filters=filter1,
    )
    label2 = Label(
        query=query2,
        document=document1,
        is_correct_answer=True,
        is_correct_document=True,
        origin="gold-label",
        answer=answer1,
        filters=filter2,
    )
    label3 = Label(
        query=query1,
        document=document1,
        is_correct_answer=True,
        is_correct_document=True,
        origin="gold-label",
        answer=answer1,
        filters=filter2,
    )

    assert MultiLabel(labels=[label1]).id == "33a3e58e13b16e9d6ec682ffe59ccc89"
    assert MultiLabel(labels=[label2]).id == "1b3ad38b629db7b0e869373b01bc32b1"
    assert MultiLabel(labels=[label3]).id == "531445fa3bdf98b8598a3bea032bd605"


@pytest.mark.unit
def test_multilabel_with_doc_containing_dataframes():
    table = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    table_doc = Document(content=table, content_type="table", id="table1")
    label = Label(
        query="A question",
        document=table_doc,
        is_correct_answer=True,
        is_correct_document=True,
        origin="gold-label",
        answer=Answer(
            answer="1",
            context=table,
            offsets_in_document=[TableCell(0, 0)],
            offsets_in_context=[TableCell(0, 0)],
            document_ids=[table_doc.id],
        ),
    )
    multilabel = MultiLabel(labels=[label])
    assert multilabel.query == "A question"
    assert multilabel.contexts == ["   col1  col2\n0     1     3\n1     2     4"]
    assert multilabel.answers == ["1"]
    assert multilabel.document_ids == ["table1"]
    assert multilabel.offsets_in_documents == [{"row": 0, "col": 0}]
    assert multilabel.offsets_in_contexts == [{"row": 0, "col": 0}]


@pytest.mark.unit
def test_multilabel_serialization():
    label_dict = {
        "id": "011079cf-c93f-49e6-83bb-42cd850dce12",
        "query": "When was the final season first shown on TV?",
        "document": {
            "content": "\n\n\n\n\nThe eighth and final season of the fantasy drama television series ''Game of Thrones'', produced by HBO, premiered on April 14, 2019, and concluded on May 19, 2019. Unlike the first six seasons, which consisted of ten episodes each, and the seventh season, which consisted of seven episodes, the eighth season consists of only six episodes.\n\nThe final season depicts the culmination of the series' two primary conflicts: the G",
            "content_type": "text",
            "id": "9c82c97c9dc8ba6895893a53aafa610f",
            "meta": {},
            "score": None,
            "embedding": None,
        },
        "is_correct_answer": True,
        "is_correct_document": True,
        "origin": "user-feedback",
        "answer": {
            "answer": "April 14",
            "type": "extractive",
            "score": None,
            "context": "\n\n\n\n\nThe eighth and final season of the fantasy drama television series ''Game of Thrones'', produced by HBO, premiered on April 14, 2019, and concluded on May 19, 2019. Unlike the first six seasons, which consisted of ten episodes each, and the seventh season, which consisted of seven episodes, the eighth season consists of only six episodes.\n\nThe final season depicts the culmination of the series' two primary conflicts: the G",
            "offsets_in_document": [{"start": 124, "end": 132}],
            "offsets_in_context": None,
            "document_ids": None,
            "meta": {},
        },
        "no_answer": False,
        "pipeline_id": None,
        "created_at": "2022-07-22T13:29:33.699781+00:00",
        "updated_at": "2022-07-22T13:29:33.784895+00:00",
        "meta": {"answer_id": "374394", "document_id": "604995", "question_id": "345530"},
        "filters": None,
    }

    label = Label.from_dict(label_dict)
    original_multilabel = MultiLabel([label])

    deserialized_multilabel = MultiLabel.from_dict(original_multilabel.to_dict())
    assert deserialized_multilabel == original_multilabel
    assert deserialized_multilabel.labels[0] == label

    json_deserialized_multilabel = MultiLabel.from_json(original_multilabel.to_json())
    assert json_deserialized_multilabel == original_multilabel
    assert json_deserialized_multilabel.labels[0] == label


@pytest.mark.unit
def test_table_multilabel_serialization():
    tabel_label_dict = {
        "id": "011079cf-c93f-49e6-83bb-42cd850dce12",
        "query": "What is the first number?",
        "document": {
            "content": [["col1", "col2"], [1, 3], [2, 4]],
            "content_type": "table",
            "id": "table1",
            "meta": {},
            "score": None,
            "embedding": None,
        },
        "is_correct_answer": True,
        "is_correct_document": True,
        "origin": "user-feedback",
        "answer": {
            "answer": "1",
            "type": "extractive",
            "score": None,
            "context": [["col1", "col2"], [1, 3], [2, 4]],
            "offsets_in_document": [{"row": 0, "col": 0}],
            "offsets_in_context": [{"row": 0, "col": 0}],
            "document_ids": ["table1"],
            "meta": {},
        },
        "no_answer": False,
        "pipeline_id": None,
        "created_at": "2022-07-22T13:29:33.699781+00:00",
        "updated_at": "2022-07-22T13:29:33.784895+00:00",
        "meta": {"answer_id": "374394", "document_id": "604995", "question_id": "345530"},
        "filters": None,
    }

    label = Label.from_dict(tabel_label_dict)
    original_multilabel = MultiLabel([label])

    deserialized_multilabel = MultiLabel.from_dict(original_multilabel.to_dict())
    assert deserialized_multilabel == original_multilabel
    assert deserialized_multilabel.labels[0] == label

    json_deserialized_multilabel = MultiLabel.from_json(original_multilabel.to_json())
    assert json_deserialized_multilabel == original_multilabel
    assert json_deserialized_multilabel.labels[0] == label


@pytest.mark.unit
def test_span_in():
    assert 10 in Span(5, 15)
    assert 20 not in Span(1, 15)


@pytest.mark.unit
def test_span_in_edges():
    assert 5 in Span(5, 15)
    assert 15 not in Span(5, 15)


@pytest.mark.unit
def test_span_in_other_values():
    assert 10.0 in Span(5, 15)
    assert "10" in Span(5, 15)
    with pytest.raises(ValueError):
        assert "hello" in Span(5, 15)


@pytest.mark.unit
def test_assert_span_vs_span():
    assert Span(10, 11) in Span(5, 15)
    assert Span(5, 10) in Span(5, 15)
    assert not Span(10, 15) in Span(5, 15)
    assert not Span(5, 15) in Span(5, 15)
    assert Span(5, 14) in Span(5, 15)

    assert not Span(0, 1) in Span(5, 15)
    assert not Span(0, 10) in Span(5, 15)
    assert not Span(10, 20) in Span(5, 15)


@pytest.mark.unit
def test_id_hash_keys_not_ignored():
    # Test that two documents with the same content but different metadata get assigned different ids if and only if
    # id_hash_keys is set to 'meta'
    doc1 = Document(content="hello world", meta={"doc_id": "1"}, id_hash_keys=["meta"])
    doc2 = Document(content="hello world", meta={"doc_id": "2"}, id_hash_keys=["meta"])
    assert doc1.id != doc2.id
    doc3 = Document(content="hello world", meta={"doc_id": "3"})
    doc4 = Document(content="hello world", meta={"doc_id": "4"})
    assert doc3.id == doc4.id


@pytest.mark.unit
def test_legacy_answer_document_id():
    legacy_label = {
        "id": "123",
        "query": "Who made the PDF specification?",
        "document": {
            "content": "Some content",
            "content_type": "text",
            "score": None,
            "id": "fc18c987a8312e72a47fb1524f230bb0",
            "meta": {},
            "embedding": [0.1, 0.2, 0.3],
        },
        "answer": {
            "answer": "Adobe Systems",
            "type": "extractive",
            "context": "Some content",
            "offsets_in_context": [{"start": 60, "end": 73}],
            "offsets_in_document": [{"start": 60, "end": 73}],
            # legacy document_id answer
            "document_id": "fc18c987a8312e72a47fb1524f230bb0",
            "meta": {},
            "score": None,
        },
        "is_correct_answer": True,
        "is_correct_document": True,
        "origin": "user-feedback",
        "pipeline_id": "some-123",
    }

    answer = Answer.from_dict(legacy_label["answer"])
    assert answer.document_ids == ["fc18c987a8312e72a47fb1524f230bb0"]

    label = Label.from_dict(legacy_label)
    assert label.answer.document_ids == ["fc18c987a8312e72a47fb1524f230bb0"]


@pytest.mark.unit
def test_legacy_answer_document_id_is_none():
    legacy_label = {
        "id": "123",
        "query": "Who made the PDF specification?",
        "document": {
            "content": "Some content",
            "content_type": "text",
            "score": None,
            "id": "fc18c987a8312e72a47fb1524f230bb0",
            "meta": {},
            "embedding": [0.1, 0.2, 0.3],
        },
        "answer": {
            "answer": "Adobe Systems",
            "type": "extractive",
            "context": "Some content",
            "offsets_in_context": [{"start": 60, "end": 73}],
            "offsets_in_document": [{"start": 60, "end": 73}],
            # legacy document_id answer
            "document_id": None,
            "meta": {},
            "score": None,
        },
        "is_correct_answer": True,
        "is_correct_document": True,
        "origin": "user-feedback",
        "pipeline_id": "some-123",
    }

    answer = Answer.from_dict(legacy_label["answer"])
    assert answer.document_ids is None

    label = Label.from_dict(legacy_label)
    assert label.answer.document_ids is None


@pytest.mark.unit
def test_dict_factory():
    data = [
        ("key1", "some_value"),
        ("key2", ["val1", "val2"]),
        ("key3", pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})),
    ]
    result = _dict_factory(data)
    assert result["key1"] == "some_value"
    assert result["key2"] == ["val1", "val2"]
    assert result["key3"] == [["col1", "col2"], [1, 3], [2, 4]]


@pytest.mark.unit
def test_evaluation_result_append():
    df1 = pd.DataFrame({"col1": [1, 2], "index": [3, 4]})
    df2 = pd.DataFrame({"col1": [5, 6], "index": [7, 8]})
    df_expected = pd.DataFrame({"col1": [1, 2, 5, 6], "index": [3, 4, 7, 8]})

    eval_result = EvaluationResult()
    eval_result.append("test", df1)
    pd.testing.assert_frame_equal(eval_result["test"], df1)
    assert isinstance(eval_result["test"].index, pd.RangeIndex)

    eval_result.append("test", df2)
    pd.testing.assert_frame_equal(eval_result["test"], df_expected)
    assert isinstance(eval_result["test"].index, pd.RangeIndex)
