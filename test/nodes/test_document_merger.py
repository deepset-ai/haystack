import pytest

from haystack import Document
from haystack.nodes.other.document_merger import DocumentMerger


@pytest.fixture
def doc_dicts():
    return [
        {
            "meta": {
                "name": "name_1",
                "year": "2020",
                "month": "01",
                "flat_field": 1,
                "nested_field": {1: 2, "a": 5, "c": {"3": 3}, "d": "I will be dropped by the meta merge algorithm"},
            },
            "content": "text_1",
        },
        {
            "meta": {
                "name": "name_2",
                "year": "2020",
                "month": "02",
                "flat_field": 1,
                "nested_field": {1: 2, "a": 5, "c": {"3": 3}},
            },
            "content": "text_2",
        },
        {
            "meta": {
                "name": "name_3",
                "year": "2020",
                "month": "03",
                "flat_field": 1,
                "nested_field": {1: 2, "a": 7, "c": {"3": 3}},
            },
            "content": "text_3",
        },
        {
            "meta": {
                "name": "name_4",
                "year": "2021",
                "month": "01",
                "flat_field": 1,
                "nested_field": {1: 2, "a": 5, "c": {"3": 3}},
            },
            "content": "text_4",
        },
        {
            "meta": {
                "name": "name_5",
                "year": "2021",
                "month": "02",
                "flat_field": 1,
                "nested_field": {1: 2, "a": 5, "c": {"3": 3}},
            },
            "content": "text_5",
        },
        {
            "meta": {
                "name": "name_6",
                "year": "2021",
                "month": "03",
                "flat_field": 1,
                "nested_field": {1: 2, "a": 5, "c": {"3": 3}},
            },
            "content": "text_6",
        },
    ]


@pytest.fixture
def documents(doc_dicts):
    return [Document.from_dict(doc) for doc in doc_dicts]


@pytest.mark.unit
def test_document_merger_merge(documents, doc_dicts):
    separator = "|"
    dm = DocumentMerger(separator=separator)
    merged_list = dm.merge(documents)

    assert len(merged_list) == 1
    assert merged_list[0].content == separator.join([doc["content"] for doc in doc_dicts])
    assert merged_list[0].meta == {"flat_field": 1, "nested_field": {1: 2, "c": {"3": 3}}}


@pytest.mark.unit
def test_document_merger_run(documents, doc_dicts):
    separator = "|"
    dm = DocumentMerger(separator=separator)
    result = dm.run(documents)

    assert len(result[0]["documents"]) == 1
    assert result[0]["documents"][0].content == separator.join([doc["content"] for doc in doc_dicts])
    assert result[0]["documents"][0].meta == {"flat_field": 1, "nested_field": {1: 2, "c": {"3": 3}}}


@pytest.mark.unit
def test_document_merger_run_batch(documents, doc_dicts):
    separator = "|"
    dm = DocumentMerger(separator=separator)
    batch_result = dm.run_batch([documents, documents])

    assert len(batch_result[0]["documents"]) == 2
    assert batch_result[0]["documents"][0][0].content == separator.join([doc["content"] for doc in doc_dicts])
    assert batch_result[0]["documents"][0][0].meta == {"flat_field": 1, "nested_field": {1: 2, "c": {"3": 3}}}
