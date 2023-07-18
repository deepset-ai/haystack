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


@pytest.mark.unit
def test_lost_in_the_middle_order_odd():
    docs = [
        Document("1"),
        Document("2"),
        Document("3"),
        Document("4"),
        Document("5"),
        Document("6"),
        Document("7"),
        Document("8"),
        Document("9"),
    ]
    dm = DocumentMerger(order="lost_in_the_middle")
    result, _ = dm.run(docs)
    assert result["documents"][0].content == "1 3 5 7 9 8 6 4 2"


@pytest.mark.unit
def test_batch_lost_in_the_middle_order_():
    docs = [
        [Document("1"), Document("2"), Document("3"), Document("4")],
        [Document("5"), Document("6")],
        [Document("7"), Document("8"), Document("9")],
    ]
    dm = DocumentMerger(order="lost_in_the_middle")
    result, _ = dm.run_batch(docs)

    assert result["documents"][0][0].content == "1 3 4 2"
    assert result["documents"][1][0].content == "5 6"
    assert result["documents"][2][0].content == "7 9 8"


@pytest.mark.unit
def test_lost_in_the_middle_order_even():
    docs = [
        Document("1"),
        Document("2"),
        Document("3"),
        Document("4"),
        Document("5"),
        Document("6"),
        Document("7"),
        Document("8"),
        Document("9"),
        Document("10"),
    ]
    dm = DocumentMerger(order="lost_in_the_middle")
    result, _ = dm.run(docs)
    assert result["documents"][0].content == "1 3 5 7 9 10 8 6 4 2"


@pytest.mark.unit
def test_lost_in_the_middle_order_corner():
    dm = DocumentMerger(order="lost_in_the_middle")

    # empty doc list
    docs = []
    result, _ = dm.run(docs)
    assert len(result["documents"]) == 0

    # single doc
    docs = [Document("1")]
    result, _ = dm.run(docs)
    assert result["documents"][0].content == "1"

    # two docs
    docs = [Document("1"), Document("2")]
    result, _ = dm.run(docs)
    assert result["documents"][0].content == "1 2"


@pytest.mark.unit
def test_document_merger_init():
    dm = DocumentMerger()
    assert dm.order == "default"
    assert dm.separator == " "
    assert not dm.word_count_threshold


@pytest.mark.unit
def test_document_merger_init_with_some_unknown_order():
    with pytest.raises(ValueError, match="Unknown DocumentMerger order"):
        DocumentMerger(order="some_unknown_order")


@pytest.mark.unit
def test_document_merger_with_word_count_threshold():
    dm = DocumentMerger(order="lost_in_the_middle", word_count_threshold=6)
    docs = [
        Document("word1"),
        Document("word2"),
        Document("word3"),
        Document("word4"),
        Document("word5"),
        Document("word6"),
        Document("word7"),
        Document("word8"),
        Document("word9"),
    ]
    result, _ = dm.run(docs)
    assert result["documents"][0].content == "word1 word3 word5 word6 word4 word2"

    dm = DocumentMerger(order="lost_in_the_middle", word_count_threshold=9)
    result, _ = dm.run(docs)
    assert result["documents"][0].content == "word1 word3 word5 word7 word9 word8 word6 word4 word2"
