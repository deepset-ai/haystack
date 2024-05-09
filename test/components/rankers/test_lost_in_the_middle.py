# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack import Document
from haystack.components.rankers.lost_in_the_middle import LostInTheMiddleRanker


class TestLostInTheMiddleRanker:
    def test_lost_in_the_middle_order_odd(self):
        # tests that lost_in_the_middle order works with an odd number of documents
        docs = [Document(content=str(i)) for i in range(1, 10)]
        ranker = LostInTheMiddleRanker()
        result = ranker.run(documents=docs)
        assert result["documents"]
        expected_order = "1 3 5 7 9 8 6 4 2".split()
        assert all(doc.content == expected_order[idx] for idx, doc in enumerate(result["documents"]))

    def test_lost_in_the_middle_order_even(self):
        # tests that lost_in_the_middle order works with an even number of documents
        docs = [Document(content=str(i)) for i in range(1, 11)]
        ranker = LostInTheMiddleRanker()
        result = ranker.run(documents=docs)
        expected_order = "1 3 5 7 9 10 8 6 4 2".split()
        assert all(doc.content == expected_order[idx] for idx, doc in enumerate(result["documents"]))

    def test_lost_in_the_middle_order_two_docs(self):
        # tests that lost_in_the_middle order works with two documents
        ranker = LostInTheMiddleRanker()
        # two docs
        docs = [Document(content="1"), Document(content="2")]
        result = ranker.run(documents=docs)
        assert result["documents"][0].content == "1"
        assert result["documents"][1].content == "2"

    def test_lost_in_the_middle_init(self):
        # tests that LostInTheMiddleRanker initializes with default values
        ranker = LostInTheMiddleRanker()
        assert ranker.word_count_threshold is None

        ranker = LostInTheMiddleRanker(word_count_threshold=10)
        assert ranker.word_count_threshold == 10

    def test_lost_in_the_middle_init_invalid_word_count_threshold(self):
        # tests that LostInTheMiddleRanker raises an error when word_count_threshold is <= 0
        with pytest.raises(ValueError, match="Invalid value for word_count_threshold"):
            LostInTheMiddleRanker(word_count_threshold=0)

        with pytest.raises(ValueError, match="Invalid value for word_count_threshold"):
            LostInTheMiddleRanker(word_count_threshold=-5)

    def test_lost_in_the_middle_with_word_count_threshold(self):
        # tests that lost_in_the_middle with word_count_threshold works as expected
        ranker = LostInTheMiddleRanker(word_count_threshold=6)
        docs = [Document(content="word" + str(i)) for i in range(1, 10)]
        # result, _ = ranker.run(query="", documents=docs)
        result = ranker.run(documents=docs)
        expected_order = "word1 word3 word5 word6 word4 word2".split()
        assert all(doc.content == expected_order[idx] for idx, doc in enumerate(result["documents"]))

        ranker = LostInTheMiddleRanker(word_count_threshold=9)
        # result, _ = ranker.run(query="", documents=docs)
        result = ranker.run(documents=docs)
        expected_order = "word1 word3 word5 word7 word9 word8 word6 word4 word2".split()
        assert all(doc.content == expected_order[idx] for idx, doc in enumerate(result["documents"]))

    def test_word_count_threshold_greater_than_total_number_of_words_returns_all_documents(self):
        ranker = LostInTheMiddleRanker(word_count_threshold=100)
        docs = [Document(content="word" + str(i)) for i in range(1, 10)]
        ordered_docs = ranker.run(documents=docs)
        # assert len(ordered_docs) == len(docs)
        expected_order = "word1 word3 word5 word7 word9 word8 word6 word4 word2".split()
        assert all(doc.content == expected_order[idx] for idx, doc in enumerate(ordered_docs["documents"]))

    def test_empty_documents_returns_empty_list(self):
        ranker = LostInTheMiddleRanker()
        result = ranker.run(documents=[])
        assert result == {"documents": []}

    def test_list_of_one_document_returns_same_document(self):
        ranker = LostInTheMiddleRanker()
        doc = Document(content="test")
        assert ranker.run(documents=[doc]) == {"documents": [doc]}

    @pytest.mark.parametrize("top_k", [1, 2, 3, 4, 5, 6, 7, 8, 12, 20])
    def test_lost_in_the_middle_order_with_top_k(self, top_k: int):
        # tests that lost_in_the_middle order works with an odd number of documents and a top_k parameter
        docs = [Document(content=str(i)) for i in range(1, 10)]
        ranker = LostInTheMiddleRanker()
        result = ranker.run(documents=docs, top_k=top_k)
        if top_k < len(docs):
            # top_k is less than the number of documents, so only the top_k documents should be returned in LITM order
            assert len(result["documents"]) == top_k
            expected_order = ranker.run(documents=[Document(content=str(i)) for i in range(1, top_k + 1)])
            assert result == expected_order
        else:
            # top_k is greater than the number of documents, so all documents should be returned in LITM order
            assert len(result["documents"]) == len(docs)
            assert result == ranker.run(documents=docs)
