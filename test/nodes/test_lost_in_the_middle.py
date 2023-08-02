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

    ranker = LostInTheMiddleRanker(word_count_threshold=10)
    assert ranker.word_count_threshold == 10


@pytest.mark.unit
def test_lost_in_the_middle_init_invalid_word_count_threshold():
    # tests that LostInTheMiddleRanker raises an error when word_count_threshold is <= 0
    with pytest.raises(ValueError, match="Invalid value for word_count_threshold"):
        LostInTheMiddleRanker(word_count_threshold=0)

    with pytest.raises(ValueError, match="Invalid value for word_count_threshold"):
        LostInTheMiddleRanker(word_count_threshold=-5)


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
@pytest.mark.parametrize("top_k", [1, 2, 3, 4, 5, 6, 7, 8, 12, 20])
def test_lost_in_the_middle_order_with_postive_top_k(top_k: int):
    # tests that lost_in_the_middle order works with an odd number of documents and a top_k parameter
    docs = [Document(str(i)) for i in range(1, 10)]
    ranker = LostInTheMiddleRanker()
    result = ranker.predict(query="irrelevant", documents=docs, top_k=top_k)
    if top_k < len(docs):
        # top_k is less than the number of documents, so only the top_k documents should be returned in LITM order
        assert len(result) == top_k
        expected_order = ranker.predict(query="irrelevant", documents=[Document(str(i)) for i in range(1, top_k + 1)])
        assert result == expected_order
    else:
        # top_k is greater than the number of documents, so all documents should be returned in LITM order
        assert len(result) == len(docs)
        assert result == ranker.predict(query="irrelevant", documents=docs)


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [-20, -10, -5, -1])
def test_lost_in_the_middle_order_with_negative_top_k(top_k: int):
    # tests that lost_in_the_middle order works with an odd number of documents and an invalid top_k parameter
    docs = [Document(str(i)) for i in range(1, 10)]
    ranker = LostInTheMiddleRanker()
    result = ranker.predict(query="irrelevant", documents=docs, top_k=top_k)
    if top_k < len(docs) * -1:
        assert len(result) == 0  # top_k is too negative, so no documents should be returned
    else:
        # top_k is negative, subtract it from the total number of documents to get the expected number of documents
        expected_docs = ranker.predict(query="irrelevant", documents=docs, top_k=len(docs) + top_k)
        assert result == expected_docs
