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
    # tests that lost_in_the_middle order works with an odd number of documents
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
    # tests that lost_in_the_middle order works with a batch of documents
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
    # tests that lost_in_the_middle order works with an even number of documents
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
    # tests that lost_in_the_middle order works with some basic corner cases
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
    # tests that DocumentMerger initializes with default values
    dm = DocumentMerger()
    assert dm.order == "default"
    assert dm.separator == " "
    assert not dm.word_count_threshold


@pytest.mark.unit
def test_document_merger_init_with_some_unknown_order():
    # tests that DocumentMerger raises a ValueError when an unknown order is provided
    with pytest.raises(ValueError, match="Unknown DocumentMerger order"):
        DocumentMerger(order="some_unknown_order")


@pytest.mark.unit
def test_document_merger_with_word_count_threshold():
    # tests that lost_in_the_middle with word_count_threshold produces a single document with truncated merged content
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


@pytest.mark.unit
def test_lost_in_the_middle_order_merge_two_documents_custom_separator():
    #  tests that merging two documents with lost_in_the_middle order and custom separator
    #  produces a single document with merged content
    merger = DocumentMerger(order="lost_in_the_middle", separator="|")
    doc1 = Document(content="This is the first document.")
    doc2 = Document(content="This is the second document.")
    merged_doc = merger.merge([doc1, doc2])[0]
    assert merged_doc.content == "This is the first document.|This is the second document."


@pytest.mark.unit
def test_default_order_merge_multiple_documents_word_count_threshold():
    #  tests that merging multiple documents with default order and word count threshold produces a single
    #  document with truncated merged content
    merger = DocumentMerger(word_count_threshold=10)
    doc1 = Document(content="This is the first document.")
    doc2 = Document(content="This is the second document.")
    doc3 = Document(content="This is the third document.")
    merged_doc = merger.merge([doc1, doc2, doc3])[0]
    assert merged_doc.content == "This is the first document. This is the second document."


@pytest.mark.unit
def test_merge_empty_list_of_documents():
    #  tests that merging an empty list of documents raises a ValueError
    merger = DocumentMerger()
    with pytest.raises(ValueError):
        merger.merge([])


@pytest.mark.unit
def test_merge_non_textual_documents():
    #  tests that merging a list of non-textual documents raises a ValueError
    merger = DocumentMerger()
    doc1 = Document(content="This is a textual document.")
    doc2 = Document(content_type="image", content="This is a non-textual document.")
    with pytest.raises(ValueError):
        merger.merge([doc1, doc2])


@pytest.mark.unit
def test_separator_and_word_count_threshold_edge_cases():
    #  tests that the merge function handles edge cases for separator and word_count_threshold
    merger = DocumentMerger()
    documents = [
        Document(content="This is the first document."),
        Document(content="This is the second document."),
        Document(content="This is the third document."),
    ]
    # Test with None separator and None word_count_threshold
    result = merger.merge(documents=documents, separator=None)
    assert len(result) == 1
    assert result[0].content == "This is the first document. This is the second document. This is the third document."

    # Test with empty separator and None word_count_threshold
    result = merger.merge(documents=documents, separator="")
    assert len(result) == 1
    assert result[0].content == "This is the first document.This is the second document.This is the third document."
