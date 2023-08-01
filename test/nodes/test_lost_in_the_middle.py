import pytest

from haystack import Document
from haystack.nodes.ranker.lost_in_the_middle import LostInTheMiddleRanker


@pytest.mark.unit
def test_lost_in_the_middle_order_odd():
    # tests that lost_in_the_middle order works with an odd number of documents
    docs = [Document(str(i)) for i in range(1, 10)]
    ranker = LostInTheMiddleRanker()
    result, _ = ranker.run(query="", documents=docs)
    assert result["documents"]
    expected_order = "1 3 5 7 9 8 6 4 2".split()
    assert all(doc.content == expected_order[idx] for idx, doc in enumerate(result["documents"]))


@pytest.mark.unit
def test_batch_lost_in_the_middle_order():
    # tests that lost_in_the_middle order works with a batch of documents
    docs = [
        [Document("1"), Document("2"), Document("3"), Document("4")],
        [Document("5"), Document("6")],
        [Document("7"), Document("8"), Document("9")],
    ]
    ranker = LostInTheMiddleRanker()
    result, _ = ranker.run_batch(queries=[""], documents=docs)

    assert " ".join(doc.content for doc in result["documents"][0]) == "1 3 4 2"
    assert " ".join(doc.content for doc in result["documents"][1]) == "5 6"
    assert " ".join(doc.content for doc in result["documents"][2]) == "7 9 8"


@pytest.mark.unit
def test_lost_in_the_middle_order_even():
    # tests that lost_in_the_middle order works with an even number of documents
    docs = [Document(str(i)) for i in range(1, 11)]
    ranker = LostInTheMiddleRanker()
    result, _ = ranker.run(query="", documents=docs)
    expected_order = "1 3 5 7 9 10 8 6 4 2".split()
    assert all(doc.content == expected_order[idx] for idx, doc in enumerate(result["documents"]))


@pytest.mark.unit
def test_lost_in_the_middle_order_two_docs():
    # tests that lost_in_the_middle order works with two documents
    ranker = LostInTheMiddleRanker()

    # two docs
    docs = [Document("1"), Document("2")]
    result, _ = ranker.run(query="", documents=docs)
    assert result["documents"][0].content == "1"
    assert result["documents"][1].content == "2"


@pytest.mark.unit
def test_lost_in_the_middle_init():
    # tests that LostInTheMiddleRanker initializes with default values
    ranker = LostInTheMiddleRanker()
    assert ranker.word_count_threshold is None
    assert ranker.truncate_document is False

    ranker = LostInTheMiddleRanker(word_count_threshold=10, truncate_document=True)
    assert ranker.word_count_threshold == 10
    assert ranker.truncate_document is True

    with pytest.raises(
        ValueError, match="If truncate_document is set to True, you must specify a word_count_threshold"
    ):
        LostInTheMiddleRanker(truncate_document=True)


@pytest.mark.unit
def test_lost_in_the_middle_with_word_count_threshold():
    # tests that lost_in_the_middle with word_count_threshold works as expected
    ranker = LostInTheMiddleRanker(word_count_threshold=6)
    docs = [Document("word" + str(i)) for i in range(1, 10)]
    result, _ = ranker.run(query="", documents=docs)
    expected_order = "word1 word3 word5 word6 word4 word2".split()
    assert all(doc.content == expected_order[idx] for idx, doc in enumerate(result["documents"]))

    ranker = LostInTheMiddleRanker(word_count_threshold=9)
    result, _ = ranker.run(query="", documents=docs)
    expected_order = "word1 word3 word5 word7 word9 word8 word6 word4 word2".split()
    assert all(doc.content == expected_order[idx] for idx, doc in enumerate(result["documents"]))


@pytest.mark.unit
def test_word_count_threshold_greater_than_total_number_of_words_returns_all_documents():
    ranker = LostInTheMiddleRanker(word_count_threshold=100)
    docs = [Document("word" + str(i)) for i in range(1, 10)]
    ordered_docs = ranker.predict(query="test", documents=docs)
    assert len(ordered_docs) == len(docs)
    expected_order = "word1 word3 word5 word7 word9 word8 word6 word4 word2".split()
    assert all(doc.content == expected_order[idx] for idx, doc in enumerate(ordered_docs))


@pytest.mark.unit
def test_truncation_with_threshold():
    # tests that truncation works as expected
    ranker = LostInTheMiddleRanker(word_count_threshold=9, truncate_document=True)
    docs = [
        Document("word1 word1"),
        Document("word2 word2"),
        Document("word3 word3"),
        Document("word4 word4"),
        Document("word5 word5"),
        Document("word6 word6"),
        Document("word7 word7"),
        Document("word8 word8"),
        Document("word9 word9"),
    ]
    result, _ = ranker.run(query="", documents=docs)
    expected_order = "word1 word1 word3 word3 word5 word4 word4 word2 word2"
    assert expected_order == " ".join(doc.content for doc in result["documents"])


@pytest.mark.unit
def test_empty_documents_returns_empty_list():
    ranker = LostInTheMiddleRanker()
    assert ranker.predict(query="test", documents=[]) == []


@pytest.mark.unit
def test_list_of_one_document_returns_same_document():
    ranker = LostInTheMiddleRanker()
    doc = Document(content="test", content_type="text")
    assert ranker.predict(query="test", documents=[doc]) == [doc]


@pytest.mark.unit
def test_non_textual_documents():
    #  tests that merging a list of non-textual documents raises a ValueError
    ranker = LostInTheMiddleRanker()
    doc1 = Document(content="This is a textual document.")
    doc2 = Document(content_type="image", content="This is a non-textual document.")
    with pytest.raises(ValueError, match="Some provided documents are not textual"):
        ranker.reorder_documents([doc1, doc2])


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [1, 2, 3, 4, 5, 6, 7, 8])
def test_lost_in_the_middle_order_odd_with_top_k(top_k: int):
    # tests that lost_in_the_middle order works with an odd number of documents and a top_k parameter
    docs = [Document(str(i)) for i in range(1, 10)]
    ranker = LostInTheMiddleRanker()
    result = ranker._exclude_middle_elements(docs, top_k=top_k)
    assert len(result) == top_k

    reverse_top_k = len(docs) - top_k
    middle = len(docs) // 2 + 1
    expected_docs = docs[: middle - reverse_top_k // 2] + docs[middle + reverse_top_k // 2 + reverse_top_k % 2 :]
    assert result == expected_docs


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [1, 2, 3, 4, 5, 6, 7, 8])
def test_lost_in_the_middle_order_even_with_top_k(top_k: int):
    # tests that lost_in_the_middle order works with an even number of documents and a top_k parameter
    docs = [Document(str(i)) for i in range(1, 9)]
    ranker = LostInTheMiddleRanker()
    result = ranker._exclude_middle_elements(docs, top_k=top_k)
    assert len(result) == top_k

    reverse_top_k = len(docs) - top_k
    middle = len(docs) // 2
    expected_docs = docs[: middle - reverse_top_k // 2] + docs[middle + reverse_top_k // 2 + reverse_top_k % 2 :]
    assert result == expected_docs


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [-5, 15])
def test_lost_in_the_middle_order_odd_with_invalid_top_k(top_k: int):
    # tests that lost_in_the_middle order works with an odd number of documents and an invalid top_k parameter
    docs = [Document(str(i)) for i in range(1, 10)]
    ranker = LostInTheMiddleRanker()
    with pytest.raises(ValueError):
        ranker._exclude_middle_elements(docs, top_k=top_k)


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [-5, 15])
def test_lost_in_the_middle_order_even_with_invalid_top_k(top_k: int):
    # tests that lost_in_the_middle order works with an even number of documents and an invalid top_k parameter
    docs = [Document(str(i)) for i in range(1, 9)]
    ranker = LostInTheMiddleRanker()
    with pytest.raises(ValueError):
        ranker._exclude_middle_elements(docs, top_k=top_k)
