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
            "headlines": [{"headline": "1", "start_idx": 5}],
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
            "headlines": [{"headline": "3", "start_idx": 5}],
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
            "headlines": [{"headline": "text", "start_idx": 0}, {"headline": "4", "start_idx": 5}],
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


def test_run(documents):
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
            {"headline": "1", "start_idx": 5},
            {"headline": "3", "start_idx": 19},
            {"headline": "text", "start_idx": 21},
            {"headline": "4", "start_idx": 26},
        ],
        "page": 1,
    }


def test_run_batch(documents):
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
            {"headline": "1", "start_idx": 5},
            {"headline": "3", "start_idx": 19},
            {"headline": "text", "start_idx": 21},
            {"headline": "4", "start_idx": 26},
        ],
        "page": 1,
    }


def test_run_with_no_docs():
    separator = "|"
    dm = DocumentMerger(separator=separator)
    result, _ = dm.run([])

    assert result["documents"] == []


def test_run_with_window_size(documents):
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
        "headlines": [{"headline": "1", "start_idx": 5}],
        "page": 1,
    }
    assert result["documents"][1].meta == {
        "flat_field": 1,
        "nested_field": {1: 2, "c": {"3": 3}},
        "headlines": [
            {"headline": "3", "start_idx": 5},
            {"headline": "text", "start_idx": 7},
            {"headline": "4", "start_idx": 12},
        ],
        "page": 3,
    }
    assert result["documents"][2].meta == {
        "year": "2021",
        "flat_field": 1,
        "nested_field": {1: 2, "a": 5, "c": {"3": 3}},
        "page": 5,
    }


def test_run_with_window_overlap(documents):
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
        "headlines": [{"headline": "1", "start_idx": 5}, {"headline": "3", "start_idx": 19}],
        "page": 1,
    }
    assert result["documents"][1].meta == {
        "flat_field": 1,
        "nested_field": {1: 2, "c": {"3": 3}},
        "headlines": [
            {"headline": "3", "start_idx": 5},
            {"headline": "text", "start_idx": 7},
            {"headline": "4", "start_idx": 12},
        ],
        "page": 3,
    }
    assert result["documents"][2].meta == {
        "year": "2021",
        "flat_field": 1,
        "nested_field": {1: 2, "a": 5, "c": {"3": 3}},
        "page": 5,
    }


def test_run_no_headlines(documents):
    separator = "|"
    dm = DocumentMerger(separator=separator, realign_headlines=False)
    result, _ = dm.run(documents)

    assert len(result["documents"]) == 1
    assert result["documents"][0].content == separator.join([doc["content"] for doc in doc_dicts])
    assert result["documents"][0].meta == {"flat_field": 1, "nested_field": {1: 2, "c": {"3": 3}}, "page": 1}


def test_run_no_page_numbers(documents):
    separator = "|"
    dm = DocumentMerger(separator=separator, retain_page_number=False)
    result, _ = dm.run(documents)

    assert len(result["documents"]) == 1
    assert result["documents"][0].content == separator.join([doc["content"] for doc in doc_dicts])
    assert result["documents"][0].meta == {
        "flat_field": 1,
        "nested_field": {1: 2, "c": {"3": 3}},
        "headlines": [
            {"headline": "1", "start_idx": 5},
            {"headline": "3", "start_idx": 19},
            {"headline": "text", "start_idx": 21},
            {"headline": "4", "start_idx": 26},
        ],
    }


def test_run_with_max_tokens():
    dm = DocumentMerger(separator="", window_size=0, max_tokens=3)
    results = dm.run(
        documents=[
            Document(content="a ", meta={"tokens_count": 1}),
            Document(content="b cd ", meta={"tokens_count": 2}),
            Document(content="e fg h ", meta={"tokens_count": 3}),
            Document(content="ij k ", meta={"tokens_count": 2}),
            Document(content="l m ", meta={"tokens_count": 2}),
            Document(content="n ", meta={"tokens_count": 1}),
        ]
    )[0]["documents"]

    assert results == [
        Document(content="a b cd ", meta={"tokens_count": 3}),
        Document(content="e fg h ", meta={"tokens_count": 3}),
        Document(content="ij k ", meta={"tokens_count": 2}),
        Document(content="l m n ", meta={"tokens_count": 3}),
    ]


def test_run_with_max_tokens_and_window_size():
    dm = DocumentMerger(separator="", window_size=2, max_tokens=5)
    results = dm.run(
        documents=[
            Document(content="a ", meta={"tokens_count": 1}),
            Document(content="b cd ", meta={"tokens_count": 2}),
            Document(content="e fg h ", meta={"tokens_count": 3}),
            Document(content="ij ", meta={"tokens_count": 1}),
            Document(content="k ", meta={"tokens_count": 1}),
            Document(content="l m n o ", meta={"tokens_count": 4}),
            Document(content="p ", meta={"tokens_count": 1}),
        ]
    )[0]["documents"]

    assert results == [
        Document(content="a b cd ", meta={"tokens_count": 3}),
        Document(content="e fg h ij ", meta={"tokens_count": 4}),
        Document(content="k l m n o ", meta={"tokens_count": 5}),
        Document(content="p ", meta={"tokens_count": 1}),
    ]


def test_run_with_too_long_documents(caplog):
    dm = DocumentMerger(separator="", window_size=0, max_tokens=3)
    results = dm.run(
        documents=[
            Document(content="a ", meta={"tokens_count": 1}),
            Document(content="b cd ", meta={"tokens_count": 2}),
            Document(content="e fg h ", meta={"tokens_count": 3}),
            Document(content="l m n o p q ", meta={"tokens_count": 6}),
            Document(content="a1 ", meta={"tokens_count": 1}),
            Document(content="b1 cd1 ", meta={"tokens_count": 2}),
        ]
    )[0]["documents"]

    assert results == [
        Document(content="a b cd ", meta={"tokens_count": 3}),
        Document(content="e fg h ", meta={"tokens_count": 3}),
        Document(content="l m n o p q ", meta={"tokens_count": 6}),
        Document(content="a1 b1 cd1 ", meta={"tokens_count": 3}),
    ]
    assert "max_tokens" in caplog.text
