import pytest

from haystack import Document
from haystack.nodes.preprocessor.merger import DocumentMerger

doc_dicts = [
    {
        "meta": {
            "name": "name_1",
            "year": "2020",
            "month": "01",
            "flat_field": 1,
            "nested_field": {1: 2, "a": 5, "c": {"3": 3}, "d": "I will be dropped by the meta merge algorithm"},
            "headlines": [{"content": "1", "start_idx": 5}],
            "page": 1,
        },
        "content": "text_1",
    },
    {
        "meta": {
            "name": "name_2",
            "year": "2020",
            "month": "02",
            "flat_field": 1,
            "nested_field": {1: 2, "a": 5, "c": {"3": 3}, "d": "I will be dropped by the meta merge algorithm"},
            "page": 2,
        },
        "content": "text_2",
    },
    {
        "meta": {
            "name": "name_3",
            "year": "2020",
            "month": "03",
            "flat_field": 1,
            "nested_field": {1: 2, "a": 7, "c": {"3": 3}, "d": "I will be dropped by the meta merge algorithm"},
            "headlines": [{"content": "3", "start_idx": 5}],
            "page": 3,
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
            "headlines": [{"content": "text", "start_idx": 0}, {"content": "4", "start_idx": 5}],
            "page": 4,
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
            "page": 5,
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
            "page": 6,
        },
        "content": "text_6",
    },
]


@pytest.fixture
def documents():
    return [Document.from_dict(doc) for doc in doc_dicts]


def test_document_merger_run(documents):
    separator = "|"
    dm = DocumentMerger(separator=separator)
    result, _ = dm.run(documents)

    print(result["documents"][0].content)

    assert len(result["documents"]) == 1
    assert result["documents"][0].content == separator.join([doc["content"] for doc in doc_dicts])
    assert result["documents"][0].meta == {
        "flat_field": 1,
        "nested_field": {1: 2, "c": {"3": 3}},
        "headlines": [
            {"content": "1", "start_idx": 5},
            {"content": "3", "start_idx": 19},
            {"content": "text", "start_idx": 21},
            {"content": "4", "start_idx": 26},
        ],
        "page": 1,
    }


def test_document_merger_run_batch(documents):
    separator = "|"
    dm = DocumentMerger(separator=separator)
    batch_result, _ = dm.run_batch([documents, documents])

    assert len(batch_result["documents"]) == 2
    assert len(batch_result["documents"][0]) == 1

    print(batch_result["documents"][0])
    assert batch_result["documents"][0][0].content == separator.join([doc["content"] for doc in doc_dicts])
    assert batch_result["documents"][0][0].meta == {
        "flat_field": 1,
        "nested_field": {1: 2, "c": {"3": 3}},
        "headlines": [
            {"content": "1", "start_idx": 5},
            {"content": "3", "start_idx": 19},
            {"content": "text", "start_idx": 21},
            {"content": "4", "start_idx": 26},
        ],
        "page": 1,
    }


def test_document_merger_run_with_no_docs():
    separator = "|"
    dm = DocumentMerger(separator=separator)
    result, _ = dm.run([])

    assert result["documents"] == []


def test_document_merger_run_with_window_size(documents):
    separator = "|"
    dm = DocumentMerger(separator=separator, window_size=2)
    result, _ = dm.run(documents)

    assert len(result["documents"]) == 3

    assert result["documents"][0].content == separator.join([doc["content"] for doc in doc_dicts[:2]])
    assert result["documents"][1].content == separator.join([doc["content"] for doc in doc_dicts[2:4]])
    assert result["documents"][2].content == separator.join([doc["content"] for doc in doc_dicts[4:]])

    assert result["documents"][0].meta == {
        "year": "2020",
        "flat_field": 1,
        "nested_field": {1: 2, "a": 5, "c": {"3": 3}, "d": "I will be dropped by the meta merge algorithm"},
        "headlines": [{"content": "1", "start_idx": 5}],
        "page": 1,
    }
    assert result["documents"][1].meta == {
        "flat_field": 1,
        "nested_field": {1: 2, "c": {"3": 3}},
        "headlines": [
            {"content": "3", "start_idx": 5},
            {"content": "text", "start_idx": 7},
            {"content": "4", "start_idx": 12},
        ],
        "page": 3,
    }
    assert result["documents"][2].meta == {
        "year": "2021",
        "flat_field": 1,
        "nested_field": {1: 2, "a": 5, "c": {"3": 3}},
        "page": 5,
    }


def test_document_merger_run_with_window_overlap(documents):
    separator = "|"
    dm = DocumentMerger(separator=separator, window_size=3, window_overlap=1)
    result, _ = dm.run(documents)

    assert len(result["documents"]) == 3

    assert result["documents"][0].content == separator.join([doc["content"] for doc in doc_dicts[:3]])
    assert result["documents"][1].content == separator.join([doc["content"] for doc in doc_dicts[2:5]])
    assert result["documents"][2].content == separator.join([doc["content"] for doc in doc_dicts[4:]])

    # Notice how due to the overlap, the headline from doc3 is duplicated between this and the following document
    # but the start_idx is different due to the different position in the merged document
    assert result["documents"][0].meta == {
        "year": "2020",
        "flat_field": 1,
        "nested_field": {1: 2, "c": {"3": 3}, "d": "I will be dropped by the meta merge algorithm"},
        "headlines": [{"content": "1", "start_idx": 5}, {"content": "3", "start_idx": 19}],
        "page": 1,
    }
    assert result["documents"][1].meta == {
        "flat_field": 1,
        "nested_field": {1: 2, "c": {"3": 3}},
        "headlines": [
            {"content": "3", "start_idx": 5},
            {"content": "text", "start_idx": 7},
            {"content": "4", "start_idx": 12},
        ],
        "page": 3,
    }
    assert result["documents"][2].meta == {
        "year": "2021",
        "flat_field": 1,
        "nested_field": {1: 2, "a": 5, "c": {"3": 3}},
        "page": 5,
    }


def test_document_merger_run_no_headlines(documents):
    separator = "|"
    dm = DocumentMerger(separator=separator, realign_headlines=False)
    result, _ = dm.run(documents)

    assert len(result["documents"]) == 1
    assert result["documents"][0].content == separator.join([doc["content"] for doc in doc_dicts])
    assert result["documents"][0].meta == {"flat_field": 1, "nested_field": {1: 2, "c": {"3": 3}}, "page": 1}


def test_document_merger_run_no_page_numbers(documents):
    separator = "|"
    dm = DocumentMerger(separator=separator, retain_page_number=False)
    result, _ = dm.run(documents)

    assert len(result["documents"]) == 1
    assert result["documents"][0].content == separator.join([doc["content"] for doc in doc_dicts])
    assert result["documents"][0].meta == {
        "flat_field": 1,
        "nested_field": {1: 2, "c": {"3": 3}},
        "headlines": [
            {"content": "1", "start_idx": 5},
            {"content": "3", "start_idx": 19},
            {"content": "text", "start_idx": 21},
            {"content": "4", "start_idx": 26},
        ],
    }
