# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack import Document
from haystack.components.preprocessors import DocumentSplitter


class TestDocumentSplitter:
    def test_non_text_document(self):
        with pytest.raises(
            ValueError, match="DocumentSplitter only works with text documents but content for document ID"
        ):
            splitter = DocumentSplitter()
            splitter.run(documents=[Document()])

    def test_single_doc(self):
        with pytest.raises(TypeError, match="DocumentSplitter expects a List of Documents as input."):
            splitter = DocumentSplitter()
            splitter.run(documents=Document())

    def test_empty_list(self):
        splitter = DocumentSplitter()
        res = splitter.run(documents=[])
        assert res == {"documents": []}

    def test_unsupported_split_by(self):
        with pytest.raises(ValueError, match="split_by must be one of 'word', 'sentence', 'page' or 'passage'."):
            DocumentSplitter(split_by="unsupported")

    def test_unsupported_split_length(self):
        with pytest.raises(ValueError, match="split_length must be greater than 0."):
            DocumentSplitter(split_length=0)

    def test_unsupported_split_overlap(self):
        with pytest.raises(ValueError, match="split_overlap must be greater than or equal to 0."):
            DocumentSplitter(split_overlap=-1)

    def test_split_by_word(self):
        splitter = DocumentSplitter(split_by="word", split_length=10)
        result = splitter.run(
            documents=[
                Document(
                    content="This is a text with some words. There is a second sentence. And there is a third sentence."
                )
            ]
        )
        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "This is a text with some words. There is a "
        assert result["documents"][1].content == "second sentence. And there is a third sentence."

    def test_split_by_word_with_threshold(self):
        splitter = DocumentSplitter(split_by="word", split_length=15, split_threshold=10)
        result = splitter.run(
            documents=[
                Document(
                    content="This is a text with some words. There is a second sentence. And there is a third sentence."
                )
            ]
        )
        assert len(result["documents"]) == 1
        assert (
            result["documents"][0].content
            == "This is a text with some words. There is a second sentence. And there is a third sentence."
        )

    def test_split_by_word_multiple_input_docs(self):
        splitter = DocumentSplitter(split_by="word", split_length=10)
        result = splitter.run(
            documents=[
                Document(
                    content="This is a text with some words. There is a second sentence. And there is a third sentence."
                ),
                Document(
                    content="This is a different text with some words. There is a second sentence. And there is a third sentence. And there is a fourth sentence."
                ),
            ]
        )
        assert len(result["documents"]) == 5
        assert result["documents"][0].content == "This is a text with some words. There is a "
        assert result["documents"][1].content == "second sentence. And there is a third sentence."
        assert result["documents"][2].content == "This is a different text with some words. There is "
        assert result["documents"][3].content == "a second sentence. And there is a third sentence. And "
        assert result["documents"][4].content == "there is a fourth sentence."

    def test_split_by_sentence(self):
        splitter = DocumentSplitter(split_by="sentence", split_length=1)
        result = splitter.run(
            documents=[
                Document(
                    content="This is a text with some words. There is a second sentence. And there is a third sentence."
                )
            ]
        )
        assert len(result["documents"]) == 3
        assert result["documents"][0].content == "This is a text with some words."
        assert result["documents"][1].content == " There is a second sentence."
        assert result["documents"][2].content == " And there is a third sentence."

    def test_split_by_passage(self):
        splitter = DocumentSplitter(split_by="passage", split_length=1)
        result = splitter.run(
            documents=[
                Document(
                    content="This is a text with some words. There is a second sentence.\n\nAnd there is a third sentence.\n\n And another passage."
                )
            ]
        )
        assert len(result["documents"]) == 3
        assert result["documents"][0].content == "This is a text with some words. There is a second sentence.\n\n"
        assert result["documents"][1].content == "And there is a third sentence.\n\n"
        assert result["documents"][2].content == " And another passage."

    def test_split_by_page(self):
        splitter = DocumentSplitter(split_by="page", split_length=1)
        result = splitter.run(
            documents=[
                Document(
                    content="This is a text with some words. There is a second sentence.\f And there is a third sentence.\f And another passage."
                )
            ]
        )
        assert len(result["documents"]) == 3
        assert result["documents"][0].content == "This is a text with some words. There is a second sentence.\x0c"
        assert result["documents"][1].content == " And there is a third sentence.\x0c"
        assert result["documents"][2].content == " And another passage."

    def test_split_by_word_with_overlap(self):
        splitter = DocumentSplitter(split_by="word", split_length=10, split_overlap=2)
        result = splitter.run(
            documents=[
                Document(
                    content="This is a text with some words. There is a second sentence. And there is a third sentence."
                )
            ]
        )
        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "This is a text with some words. There is a "
        assert result["documents"][1].content == "is a second sentence. And there is a third sentence."

    def test_source_id_stored_in_metadata(self):
        splitter = DocumentSplitter(split_by="word", split_length=10)
        doc1 = Document(content="This is a text with some words.")
        doc2 = Document(content="This is a different text with some words.")
        result = splitter.run(documents=[doc1, doc2])
        assert result["documents"][0].meta["source_id"] == doc1.id
        assert result["documents"][1].meta["source_id"] == doc2.id

    def test_copy_metadata(self):
        splitter = DocumentSplitter(split_by="word", split_length=10)
        documents = [
            Document(content="Text.", meta={"name": "doc 0"}),
            Document(content="Text.", meta={"name": "doc 1"}),
        ]
        result = splitter.run(documents=documents)
        assert len(result["documents"]) == 2
        assert result["documents"][0].id != result["documents"][1].id
        for doc, split_doc in zip(documents, result["documents"]):
            assert doc.meta.items() <= split_doc.meta.items()
            assert split_doc.content == "Text."

    def test_add_page_number_to_metadata_with_no_overlap_word_split(self):
        splitter = DocumentSplitter(split_by="word", split_length=2)
        doc1 = Document(content="This is some text.\f This text is on another page.")
        doc2 = Document(content="This content has two.\f\f page brakes.")
        result = splitter.run(documents=[doc1, doc2])

        expected_pages = [1, 1, 2, 2, 2, 1, 1, 3]
        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_no_overlap_sentence_split(self):
        splitter = DocumentSplitter(split_by="sentence", split_length=1)
        doc1 = Document(content="This is some text.\f This text is on another page.")
        doc2 = Document(content="This content has two.\f\f page brakes.")
        result = splitter.run(documents=[doc1, doc2])

        expected_pages = [1, 1, 1, 1]
        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_no_overlap_passage_split(self):
        splitter = DocumentSplitter(split_by="passage", split_length=1)
        doc1 = Document(
            content="This is a text with some words.\f There is a second sentence.\n\nAnd there is a third sentence.\n\nAnd more passages.\n\n\f And another passage."
        )
        result = splitter.run(documents=[doc1])

        expected_pages = [1, 2, 2, 2]
        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_no_overlap_page_split(self):
        splitter = DocumentSplitter(split_by="page", split_length=1)
        doc1 = Document(
            content="This is a text with some words. There is a second sentence.\f And there is a third sentence.\f And another passage."
        )
        result = splitter.run(documents=[doc1])
        expected_pages = [1, 2, 3]
        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

        splitter = DocumentSplitter(split_by="page", split_length=2)
        doc1 = Document(
            content="This is a text with some words. There is a second sentence.\f And there is a third sentence.\f And another passage."
        )
        result = splitter.run(documents=[doc1])
        expected_pages = [1, 3]

        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_overlap_word_split(self):
        splitter = DocumentSplitter(split_by="word", split_length=3, split_overlap=1)
        doc1 = Document(content="This is some text. And\f this text is on another page.")
        doc2 = Document(content="This content has two.\f\f page brakes.")
        result = splitter.run(documents=[doc1, doc2])

        expected_pages = [1, 1, 1, 2, 2, 1, 1, 3]
        for doc, p in zip(result["documents"], expected_pages):
            print(doc.content, doc.meta, p)
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_overlap_sentence_split(self):
        splitter = DocumentSplitter(split_by="sentence", split_length=2, split_overlap=1)
        doc1 = Document(content="This is some text. And this is more text.\f This text is on another page. End.")
        doc2 = Document(content="This content has two.\f\f page brakes. More text.")
        result = splitter.run(documents=[doc1, doc2])

        expected_pages = [1, 1, 1, 2, 1, 1]
        for doc, p in zip(result["documents"], expected_pages):
            print(doc.content, doc.meta, p)
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_overlap_passage_split(self):
        splitter = DocumentSplitter(split_by="passage", split_length=2, split_overlap=1)
        doc1 = Document(
            content="This is a text with some words.\f There is a second sentence.\n\nAnd there is a third sentence.\n\nAnd more passages.\n\n\f And another passage."
        )
        result = splitter.run(documents=[doc1])

        expected_pages = [1, 2, 2]
        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_overlap_page_split(self):
        splitter = DocumentSplitter(split_by="page", split_length=2, split_overlap=1)
        doc1 = Document(
            content="This is a text with some words. There is a second sentence.\f And there is a third sentence.\f And another passage."
        )
        result = splitter.run(documents=[doc1])
        expected_pages = [1, 2, 3]

        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p
