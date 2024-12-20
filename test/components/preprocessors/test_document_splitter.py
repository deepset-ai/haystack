# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

import re

import pytest

from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.utils import deserialize_callable, serialize_callable


# custom split function for testing
def custom_split(text):
    return text.split(".")


def merge_documents(documents):
    """Merge a list of doc chunks into a single doc by concatenating their content, eliminating overlapping content."""
    sorted_docs = sorted(documents, key=lambda doc: doc.meta["split_idx_start"])
    merged_text = ""
    last_idx_end = 0
    for doc in sorted_docs:
        start = doc.meta["split_idx_start"]  # start of the current content

        # if the start of the current content is before the end of the last appended content, adjust it
        if start < last_idx_end:
            start = last_idx_end

        # append the non-overlapping part to the merged text
        merged_text += doc.content[start - doc.meta["split_idx_start"] :]

        # update the last end index
        last_idx_end = doc.meta["split_idx_start"] + len(doc.content)

    return merged_text


class TestSplittingByFunctionOrCharacterRegex:
    def test_non_text_document(self):
        with pytest.raises(
            ValueError, match="DocumentSplitter only works with text documents but content for document ID"
        ):
            splitter = DocumentSplitter()
            splitter.warm_up()
            splitter.run(documents=[Document()])
            assert "DocumentSplitter only works with text documents but content for document ID" in caplog.text

    def test_single_doc(self):
        with pytest.raises(TypeError, match="DocumentSplitter expects a List of Documents as input."):
            splitter = DocumentSplitter()
            splitter.warm_up()
            splitter.run(documents=Document())

    def test_empty_list(self):
        splitter = DocumentSplitter()
        splitter.warm_up()
        res = splitter.run(documents=[])
        assert res == {"documents": []}

    def test_unsupported_split_by(self):
        with pytest.raises(ValueError, match="split_by must be one of "):
            DocumentSplitter(split_by="unsupported")

    def test_undefined_function(self):
        with pytest.raises(ValueError, match="When 'split_by' is set to 'function', a valid 'splitting_function'"):
            DocumentSplitter(split_by="function", splitting_function=None)

    def test_unsupported_split_length(self):
        with pytest.raises(ValueError, match="split_length must be greater than 0."):
            DocumentSplitter(split_length=0)

    def test_unsupported_split_overlap(self):
        with pytest.raises(ValueError, match="split_overlap must be greater than or equal to 0."):
            DocumentSplitter(split_overlap=-1)

    def test_split_by_word(self):
        splitter = DocumentSplitter(split_by="word", split_length=10)
        text = "This is a text with some words. There is a second sentence. And there is a third sentence."
        splitter.warm_up()
        result = splitter.run(documents=[Document(content=text)])
        docs = result["documents"]
        assert len(docs) == 2
        assert docs[0].content == "This is a text with some words. There is a "
        assert docs[0].meta["split_id"] == 0
        assert docs[0].meta["split_idx_start"] == text.index(docs[0].content)
        assert docs[1].content == "second sentence. And there is a third sentence."
        assert docs[1].meta["split_id"] == 1
        assert docs[1].meta["split_idx_start"] == text.index(docs[1].content)

    def test_split_by_word_with_threshold(self):
        splitter = DocumentSplitter(split_by="word", split_length=15, split_threshold=10)
        splitter.warm_up()
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
        text1 = "This is a text with some words. There is a second sentence. And there is a third sentence."
        text2 = "This is a different text with some words. There is a second sentence. And there is a third sentence. And there is a fourth sentence."
        splitter.warm_up()
        result = splitter.run(documents=[Document(content=text1), Document(content=text2)])
        docs = result["documents"]
        assert len(docs) == 5
        # doc 0
        assert docs[0].content == "This is a text with some words. There is a "
        assert docs[0].meta["split_id"] == 0
        assert docs[0].meta["split_idx_start"] == text1.index(docs[0].content)
        # doc 1
        assert docs[1].content == "second sentence. And there is a third sentence."
        assert docs[1].meta["split_id"] == 1
        assert docs[1].meta["split_idx_start"] == text1.index(docs[1].content)
        # doc 2
        assert docs[2].content == "This is a different text with some words. There is "
        assert docs[2].meta["split_id"] == 0
        assert docs[2].meta["split_idx_start"] == text2.index(docs[2].content)
        # doc 3
        assert docs[3].content == "a second sentence. And there is a third sentence. And "
        assert docs[3].meta["split_id"] == 1
        assert docs[3].meta["split_idx_start"] == text2.index(docs[3].content)
        # doc 4
        assert docs[4].content == "there is a fourth sentence."
        assert docs[4].meta["split_id"] == 2
        assert docs[4].meta["split_idx_start"] == text2.index(docs[4].content)

    def test_split_by_period(self):
        splitter = DocumentSplitter(split_by="period", split_length=1)
        text = "This is a text with some words. There is a second sentence. And there is a third sentence."
        splitter.warm_up()
        result = splitter.run(documents=[Document(content=text)])
        docs = result["documents"]
        assert len(docs) == 3
        assert docs[0].content == "This is a text with some words."
        assert docs[0].meta["split_id"] == 0
        assert docs[0].meta["split_idx_start"] == text.index(docs[0].content)
        assert docs[1].content == " There is a second sentence."
        assert docs[1].meta["split_id"] == 1
        assert docs[1].meta["split_idx_start"] == text.index(docs[1].content)
        assert docs[2].content == " And there is a third sentence."
        assert docs[2].meta["split_id"] == 2
        assert docs[2].meta["split_idx_start"] == text.index(docs[2].content)

    def test_split_by_passage(self):
        splitter = DocumentSplitter(split_by="passage", split_length=1)
        text = "This is a text with some words. There is a second sentence.\n\nAnd there is a third sentence.\n\n And another passage."
        splitter.warm_up()
        result = splitter.run(documents=[Document(content=text)])
        docs = result["documents"]
        assert len(docs) == 3
        assert docs[0].content == "This is a text with some words. There is a second sentence.\n\n"
        assert docs[0].meta["split_id"] == 0
        assert docs[0].meta["split_idx_start"] == text.index(docs[0].content)
        assert docs[1].content == "And there is a third sentence.\n\n"
        assert docs[1].meta["split_id"] == 1
        assert docs[1].meta["split_idx_start"] == text.index(docs[1].content)
        assert docs[2].content == " And another passage."
        assert docs[2].meta["split_id"] == 2
        assert docs[2].meta["split_idx_start"] == text.index(docs[2].content)

    def test_split_by_page(self):
        splitter = DocumentSplitter(split_by="page", split_length=1)
        text = "This is a text with some words. There is a second sentence.\f And there is a third sentence.\f And another passage."
        splitter.warm_up()
        result = splitter.run(documents=[Document(content=text)])
        docs = result["documents"]
        assert len(docs) == 3
        assert docs[0].content == "This is a text with some words. There is a second sentence.\f"
        assert docs[0].meta["split_id"] == 0
        assert docs[0].meta["split_idx_start"] == text.index(docs[0].content)
        assert docs[0].meta["page_number"] == 1
        assert docs[1].content == " And there is a third sentence.\f"
        assert docs[1].meta["split_id"] == 1
        assert docs[1].meta["split_idx_start"] == text.index(docs[1].content)
        assert docs[1].meta["page_number"] == 2
        assert docs[2].content == " And another passage."
        assert docs[2].meta["split_id"] == 2
        assert docs[2].meta["split_idx_start"] == text.index(docs[2].content)
        assert docs[2].meta["page_number"] == 3

    def test_split_by_function(self):
        splitting_function = lambda s: s.split(".")
        splitter = DocumentSplitter(split_by="function", splitting_function=splitting_function)
        splitter.warm_up()
        text = "This.Is.A.Test"
        result = splitter.run(documents=[Document(id="1", content=text, meta={"key": "value"})])
        docs = result["documents"]

        assert len(docs) == 4
        assert docs[0].content == "This"
        assert docs[0].meta == {"key": "value", "source_id": "1"}
        assert docs[1].content == "Is"
        assert docs[1].meta == {"key": "value", "source_id": "1"}
        assert docs[2].content == "A"
        assert docs[2].meta == {"key": "value", "source_id": "1"}
        assert docs[3].content == "Test"
        assert docs[3].meta == {"key": "value", "source_id": "1"}

        splitting_function = lambda s: re.split(r"[\s]{2,}", s)
        splitter = DocumentSplitter(split_by="function", splitting_function=splitting_function)
        text = "This       Is\n  A  Test"
        splitter.warm_up()
        result = splitter.run(documents=[Document(id="1", content=text, meta={"key": "value"})])
        docs = result["documents"]
        assert len(docs) == 4
        assert docs[0].content == "This"
        assert docs[0].meta == {"key": "value", "source_id": "1"}
        assert docs[1].content == "Is"
        assert docs[1].meta == {"key": "value", "source_id": "1"}
        assert docs[2].content == "A"
        assert docs[2].meta == {"key": "value", "source_id": "1"}
        assert docs[3].content == "Test"
        assert docs[3].meta == {"key": "value", "source_id": "1"}

    def test_split_by_word_with_overlap(self):
        splitter = DocumentSplitter(split_by="word", split_length=10, split_overlap=2)
        text = "This is a text with some words. There is a second sentence. And there is a third sentence."
        splitter.warm_up()
        result = splitter.run(documents=[Document(content=text)])
        docs = result["documents"]
        assert len(docs) == 2
        # doc 0
        assert docs[0].content == "This is a text with some words. There is a "
        assert docs[0].meta["split_id"] == 0
        assert docs[0].meta["split_idx_start"] == text.index(docs[0].content)
        assert docs[0].meta["_split_overlap"][0]["range"] == (0, 5)
        assert docs[1].content[0:5] == "is a "
        # doc 1
        assert docs[1].content == "is a second sentence. And there is a third sentence."
        assert docs[1].meta["split_id"] == 1
        assert docs[1].meta["split_idx_start"] == text.index(docs[1].content)
        assert docs[1].meta["_split_overlap"][0]["range"] == (38, 43)
        assert docs[0].content[38:43] == "is a "

    def test_split_by_line(self):
        splitter = DocumentSplitter(split_by="line", split_length=1)
        text = "This is a text with some words.\nThere is a second sentence.\nAnd there is a third sentence."
        splitter.warm_up()
        result = splitter.run(documents=[Document(content=text)])
        docs = result["documents"]

        assert len(docs) == 3
        assert docs[0].content == "This is a text with some words.\n"
        assert docs[0].meta["split_id"] == 0
        assert docs[0].meta["split_idx_start"] == text.index(docs[0].content)
        assert docs[1].content == "There is a second sentence.\n"
        assert docs[1].meta["split_id"] == 1
        assert docs[1].meta["split_idx_start"] == text.index(docs[1].content)
        assert docs[2].content == "And there is a third sentence."
        assert docs[2].meta["split_id"] == 2
        assert docs[2].meta["split_idx_start"] == text.index(docs[2].content)

    def test_source_id_stored_in_metadata(self):
        splitter = DocumentSplitter(split_by="word", split_length=10)
        doc1 = Document(content="This is a text with some words.")
        doc2 = Document(content="This is a different text with some words.")
        splitter.warm_up()
        result = splitter.run(documents=[doc1, doc2])
        assert result["documents"][0].meta["source_id"] == doc1.id
        assert result["documents"][1].meta["source_id"] == doc2.id

    def test_copy_metadata(self):
        splitter = DocumentSplitter(split_by="word", split_length=10)
        documents = [
            Document(content="Text.", meta={"name": "doc 0"}),
            Document(content="Text.", meta={"name": "doc 1"}),
        ]
        splitter.warm_up()
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
        splitter.warm_up()
        result = splitter.run(documents=[doc1, doc2])

        expected_pages = [1, 1, 2, 2, 2, 1, 1, 3]
        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_no_overlap_period_split(self):
        splitter = DocumentSplitter(split_by="period", split_length=1)
        doc1 = Document(content="This is some text.\f This text is on another page.")
        doc2 = Document(content="This content has two.\f\f page brakes.")
        splitter.warm_up()
        result = splitter.run(documents=[doc1, doc2])

        expected_pages = [1, 1, 1, 1]
        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_no_overlap_passage_split(self):
        splitter = DocumentSplitter(split_by="passage", split_length=1)
        doc1 = Document(
            content="This is a text with some words.\f There is a second sentence.\n\nAnd there is a third sentence.\n\nAnd more passages.\n\n\f And another passage."
        )
        splitter.warm_up()
        result = splitter.run(documents=[doc1])

        expected_pages = [1, 2, 2, 2]
        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_no_overlap_page_split(self):
        splitter = DocumentSplitter(split_by="page", split_length=1)
        doc1 = Document(
            content="This is a text with some words. There is a second sentence.\f And there is a third sentence.\f And another passage."
        )
        splitter.warm_up()
        result = splitter.run(documents=[doc1])
        expected_pages = [1, 2, 3]
        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

        splitter = DocumentSplitter(split_by="page", split_length=2)
        doc1 = Document(
            content="This is a text with some words. There is a second sentence.\f And there is a third sentence.\f And another passage."
        )
        splitter.warm_up()
        result = splitter.run(documents=[doc1])
        expected_pages = [1, 3]

        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_overlap_word_split(self):
        splitter = DocumentSplitter(split_by="word", split_length=3, split_overlap=1)
        doc1 = Document(content="This is some text. And\f this text is on another page.")
        doc2 = Document(content="This content has two.\f\f page brakes.")
        splitter.warm_up()
        result = splitter.run(documents=[doc1, doc2])

        expected_pages = [1, 1, 1, 2, 2, 1, 1, 3]
        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_overlap_period_split(self):
        splitter = DocumentSplitter(split_by="period", split_length=2, split_overlap=1)
        doc1 = Document(content="This is some text. And this is more text.\f This text is on another page. End.")
        doc2 = Document(content="This content has two.\f\f page brakes. More text.")
        splitter.warm_up()
        result = splitter.run(documents=[doc1, doc2])

        expected_pages = [1, 1, 1, 2, 1, 1]
        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_overlap_passage_split(self):
        splitter = DocumentSplitter(split_by="passage", split_length=2, split_overlap=1)
        doc1 = Document(
            content="This is a text with some words.\f There is a second sentence.\n\nAnd there is a third sentence.\n\nAnd more passages.\n\n\f And another passage."
        )
        splitter.warm_up()
        result = splitter.run(documents=[doc1])

        expected_pages = [1, 2, 2]
        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

    def test_add_page_number_to_metadata_with_overlap_page_split(self):
        splitter = DocumentSplitter(split_by="page", split_length=2, split_overlap=1)
        doc1 = Document(
            content="This is a text with some words. There is a second sentence.\f And there is a third sentence.\f And another passage."
        )
        splitter.warm_up()
        result = splitter.run(documents=[doc1])
        expected_pages = [1, 2, 3]

        for doc, p in zip(result["documents"], expected_pages):
            assert doc.meta["page_number"] == p

    def test_add_split_overlap_information(self):
        splitter = DocumentSplitter(split_length=10, split_overlap=5, split_by="word")
        text = "This is a text with some words. There is a second sentence. And a third sentence."
        doc = Document(content="This is a text with some words. There is a second sentence. And a third sentence.")
        splitter.warm_up()
        docs = splitter.run(documents=[doc])["documents"]

        # check split_overlap is added to all the documents
        assert len(docs) == 3
        # doc 0
        assert docs[0].content == "This is a text with some words. There is a "
        assert docs[0].meta["split_id"] == 0
        assert docs[0].meta["split_idx_start"] == text.index(docs[0].content)  # 0
        assert docs[0].meta["_split_overlap"][0]["range"] == (0, 23)
        assert docs[1].content[0:23] == "some words. There is a "
        # doc 1
        assert docs[1].content == "some words. There is a second sentence. And a third "
        assert docs[1].meta["split_id"] == 1
        assert docs[1].meta["split_idx_start"] == text.index(docs[1].content)  # 20
        assert docs[1].meta["_split_overlap"][0]["range"] == (20, 43)
        assert docs[1].meta["_split_overlap"][1]["range"] == (0, 29)
        assert docs[0].content[20:43] == "some words. There is a "
        assert docs[2].content[0:29] == "second sentence. And a third "
        # doc 2
        assert docs[2].content == "second sentence. And a third sentence."
        assert docs[2].meta["split_id"] == 2
        assert docs[2].meta["split_idx_start"] == text.index(docs[2].content)  # 43
        assert docs[2].meta["_split_overlap"][0]["range"] == (23, 52)
        assert docs[1].content[23:52] == "second sentence. And a third "

        # reconstruct the original document content from the split documents
        assert doc.content == merge_documents(docs)

    def test_to_dict(self):
        """
        Test the to_dict method of the DocumentSplitter class.
        """
        splitter = DocumentSplitter(split_by="word", split_length=10, split_overlap=2, split_threshold=5)
        serialized = splitter.to_dict()

        assert serialized["type"] == "haystack.components.preprocessors.document_splitter.DocumentSplitter"
        assert serialized["init_parameters"]["split_by"] == "word"
        assert serialized["init_parameters"]["split_length"] == 10
        assert serialized["init_parameters"]["split_overlap"] == 2
        assert serialized["init_parameters"]["split_threshold"] == 5
        assert "splitting_function" not in serialized["init_parameters"]

    def test_to_dict_with_splitting_function(self):
        """
        Test the to_dict method of the DocumentSplitter class when a custom splitting function is provided.
        """

        splitter = DocumentSplitter(split_by="function", splitting_function=custom_split)
        serialized = splitter.to_dict()

        assert serialized["type"] == "haystack.components.preprocessors.document_splitter.DocumentSplitter"
        assert serialized["init_parameters"]["split_by"] == "function"
        assert "splitting_function" in serialized["init_parameters"]
        assert callable(deserialize_callable(serialized["init_parameters"]["splitting_function"]))

    def test_from_dict(self):
        """
        Test the from_dict class method of the DocumentSplitter class.
        """
        data = {
            "type": "haystack.components.preprocessors.document_splitter.DocumentSplitter",
            "init_parameters": {"split_by": "word", "split_length": 10, "split_overlap": 2, "split_threshold": 5},
        }
        splitter = DocumentSplitter.from_dict(data)

        assert splitter.split_by == "word"
        assert splitter.split_length == 10
        assert splitter.split_overlap == 2
        assert splitter.split_threshold == 5
        assert splitter.splitting_function is None

    def test_from_dict_with_splitting_function(self):
        """
        Test the from_dict class method of the DocumentSplitter class when a custom splitting function is provided.
        """

        def custom_split(text):
            return text.split(".")

        data = {
            "type": "haystack.components.preprocessors.document_splitter.DocumentSplitter",
            "init_parameters": {"split_by": "function", "splitting_function": serialize_callable(custom_split)},
        }
        splitter = DocumentSplitter.from_dict(data)

        assert splitter.split_by == "function"
        assert callable(splitter.splitting_function)
        assert splitter.splitting_function("a.b.c") == ["a", "b", "c"]

    def test_roundtrip_serialization(self):
        """
        Test the round-trip serialization of the DocumentSplitter class.
        """
        original_splitter = DocumentSplitter(split_by="word", split_length=10, split_overlap=2, split_threshold=5)
        serialized = original_splitter.to_dict()
        deserialized_splitter = DocumentSplitter.from_dict(serialized)

        assert original_splitter.split_by == deserialized_splitter.split_by
        assert original_splitter.split_length == deserialized_splitter.split_length
        assert original_splitter.split_overlap == deserialized_splitter.split_overlap
        assert original_splitter.split_threshold == deserialized_splitter.split_threshold

    def test_roundtrip_serialization_with_splitting_function(self):
        """
        Test the round-trip serialization of the DocumentSplitter class when a custom splitting function is provided.
        """

        original_splitter = DocumentSplitter(split_by="function", splitting_function=custom_split)
        serialized = original_splitter.to_dict()
        deserialized_splitter = DocumentSplitter.from_dict(serialized)

        assert original_splitter.split_by == deserialized_splitter.split_by
        assert callable(deserialized_splitter.splitting_function)
        assert deserialized_splitter.splitting_function("a.b.c") == ["a", "b", "c"]

    def test_run_empty_document(self):
        """
        Test if the component runs correctly with an empty document.
        """
        splitter = DocumentSplitter()
        doc = Document(content="")
        splitter.warm_up()
        results = splitter.run([doc])
        assert results["documents"] == []

    def test_run_document_only_whitespaces(self):
        """
        Test if the component runs correctly with a document containing only whitespaces.
        """
        splitter = DocumentSplitter()
        doc = Document(content="  ")
        splitter.warm_up()
        results = splitter.run([doc])
        assert results["documents"][0].content == "  "


class TestSplittingNLTKSentenceSplitter:
    @pytest.mark.parametrize(
        "sentences, expected_num_sentences",
        [
            (["The sun set.", "Moonlight shimmered softly, wolves howled nearby, night enveloped everything."], 0),
            (["The sun set.", "It was a dark night ..."], 0),
            (["The sun set.", " The moon was full."], 1),
            (["The sun.", " The moon."], 1),  # Ignores the first sentence
            (["Sun", "Moon"], 1),  # Ignores the first sentence even if its inclusion would be < split_overlap
        ],
    )
    def test_number_of_sentences_to_keep(self, sentences: List[str], expected_num_sentences: int) -> None:
        num_sentences = DocumentSplitter._number_of_sentences_to_keep(
            sentences=sentences, split_length=5, split_overlap=2
        )
        assert num_sentences == expected_num_sentences

    def test_number_of_sentences_to_keep_split_overlap_zero(self) -> None:
        sentences = [
            "Moonlight shimmered softly, wolves howled nearby, night enveloped everything.",
            " It was a dark night ...",
            " The moon was full.",
        ]
        num_sentences = DocumentSplitter._number_of_sentences_to_keep(
            sentences=sentences, split_length=5, split_overlap=0
        )
        assert num_sentences == 0

    def test_run_split_by_sentence_1(self) -> None:
        document_splitter = DocumentSplitter(
            split_by="sentence",
            split_length=2,
            split_overlap=0,
            split_threshold=0,
            language="en",
            use_split_rules=True,
            extend_abbreviations=True,
        )

        text = (
            "Moonlight shimmered softly, wolves howled nearby, night enveloped everything. It was a dark night ... "
            "The moon was full."
        )
        document_splitter.warm_up()
        documents = document_splitter.run(documents=[Document(content=text)])["documents"]

        assert len(documents) == 2
        assert (
            documents[0].content == "Moonlight shimmered softly, wolves howled nearby, night enveloped "
            "everything. It was a dark night ... "
        )
        assert documents[1].content == "The moon was full."

    def test_run_split_by_sentence_2(self) -> None:
        document_splitter = DocumentSplitter(
            split_by="sentence",
            split_length=1,
            split_overlap=0,
            split_threshold=0,
            language="en",
            use_split_rules=False,
            extend_abbreviations=True,
        )

        text = (
            "This is a test sentence with many many words that exceeds the split length and should not be repeated. "
            "This is another test sentence. (This is a third test sentence.) "
            "This is the last test sentence."
        )
        document_splitter.warm_up()
        documents = document_splitter.run(documents=[Document(content=text)])["documents"]

        assert len(documents) == 4
        assert (
            documents[0].content
            == "This is a test sentence with many many words that exceeds the split length and should not be repeated. "
        )
        assert documents[0].meta["page_number"] == 1
        assert documents[0].meta["split_id"] == 0
        assert documents[0].meta["split_idx_start"] == text.index(documents[0].content)
        assert documents[1].content == "This is another test sentence. "
        assert documents[1].meta["page_number"] == 1
        assert documents[1].meta["split_id"] == 1
        assert documents[1].meta["split_idx_start"] == text.index(documents[1].content)
        assert documents[2].content == "(This is a third test sentence.) "
        assert documents[2].meta["page_number"] == 1
        assert documents[2].meta["split_id"] == 2
        assert documents[2].meta["split_idx_start"] == text.index(documents[2].content)
        assert documents[3].content == "This is the last test sentence."
        assert documents[3].meta["page_number"] == 1
        assert documents[3].meta["split_id"] == 3
        assert documents[3].meta["split_idx_start"] == text.index(documents[3].content)

    def test_run_split_by_sentence_3(self) -> None:
        document_splitter = DocumentSplitter(
            split_by="sentence",
            split_length=1,
            split_overlap=0,
            split_threshold=0,
            language="en",
            use_split_rules=True,
            extend_abbreviations=True,
        )
        document_splitter.warm_up()

        text = "Sentence on page 1.\fSentence on page 2. \fSentence on page 3. \f\f Sentence on page 5."
        documents = document_splitter.run(documents=[Document(content=text)])["documents"]

        assert len(documents) == 4
        assert documents[0].content == "Sentence on page 1.\f"
        assert documents[0].meta["page_number"] == 1
        assert documents[0].meta["split_id"] == 0
        assert documents[0].meta["split_idx_start"] == text.index(documents[0].content)
        assert documents[1].content == "Sentence on page 2. \f"
        assert documents[1].meta["page_number"] == 2
        assert documents[1].meta["split_id"] == 1
        assert documents[1].meta["split_idx_start"] == text.index(documents[1].content)
        assert documents[2].content == "Sentence on page 3. \f\f "
        assert documents[2].meta["page_number"] == 3
        assert documents[2].meta["split_id"] == 2
        assert documents[2].meta["split_idx_start"] == text.index(documents[2].content)
        assert documents[3].content == "Sentence on page 5."
        assert documents[3].meta["page_number"] == 5
        assert documents[3].meta["split_id"] == 3
        assert documents[3].meta["split_idx_start"] == text.index(documents[3].content)

    def test_run_split_by_sentence_4(self) -> None:
        document_splitter = DocumentSplitter(
            split_by="sentence",
            split_length=2,
            split_overlap=1,
            split_threshold=0,
            language="en",
            use_split_rules=True,
            extend_abbreviations=True,
        )
        document_splitter.warm_up()

        text = "Sentence on page 1.\fSentence on page 2. \fSentence on page 3. \f\f Sentence on page 5."
        documents = document_splitter.run(documents=[Document(content=text)])["documents"]

        assert len(documents) == 3
        assert documents[0].content == "Sentence on page 1.\fSentence on page 2. \f"
        assert documents[0].meta["page_number"] == 1
        assert documents[0].meta["split_id"] == 0
        assert documents[0].meta["split_idx_start"] == text.index(documents[0].content)
        assert documents[1].content == "Sentence on page 2. \fSentence on page 3. \f\f "
        assert documents[1].meta["page_number"] == 2
        assert documents[1].meta["split_id"] == 1
        assert documents[1].meta["split_idx_start"] == text.index(documents[1].content)
        assert documents[2].content == "Sentence on page 3. \f\f Sentence on page 5."
        assert documents[2].meta["page_number"] == 3
        assert documents[2].meta["split_id"] == 2
        assert documents[2].meta["split_idx_start"] == text.index(documents[2].content)

    def test_run_split_by_word_respect_sentence_boundary(self) -> None:
        document_splitter = DocumentSplitter(
            split_by="word",
            split_length=3,
            split_overlap=0,
            split_threshold=0,
            language="en",
            respect_sentence_boundary=True,
        )
        document_splitter.warm_up()

        text = (
            "Moonlight shimmered softly, wolves howled nearby, night enveloped everything. It was a dark night.\f"
            "The moon was full."
        )
        documents = document_splitter.run(documents=[Document(content=text)])["documents"]

        assert len(documents) == 3
        assert documents[0].content == "Moonlight shimmered softly, wolves howled nearby, night enveloped everything. "
        assert documents[0].meta["page_number"] == 1
        assert documents[0].meta["split_id"] == 0
        assert documents[0].meta["split_idx_start"] == text.index(documents[0].content)
        assert documents[1].content == "It was a dark night.\f"
        assert documents[1].meta["page_number"] == 1
        assert documents[1].meta["split_id"] == 1
        assert documents[1].meta["split_idx_start"] == text.index(documents[1].content)
        assert documents[2].content == "The moon was full."
        assert documents[2].meta["page_number"] == 2
        assert documents[2].meta["split_id"] == 2
        assert documents[2].meta["split_idx_start"] == text.index(documents[2].content)

    def test_run_split_by_word_respect_sentence_boundary_no_repeats(self) -> None:
        document_splitter = DocumentSplitter(
            split_by="word",
            split_length=13,
            split_overlap=3,
            split_threshold=0,
            language="en",
            respect_sentence_boundary=True,
            use_split_rules=False,
            extend_abbreviations=False,
        )
        document_splitter.warm_up()
        text = (
            "This is a test sentence with many many words that exceeds the split length and should not be repeated. "
            "This is another test sentence. (This is a third test sentence.) "
            "This is the last test sentence."
        )
        documents = document_splitter.run([Document(content=text)])["documents"]
        assert len(documents) == 3
        assert (
            documents[0].content
            == "This is a test sentence with many many words that exceeds the split length and should not be repeated. "
        )
        assert "This is a test sentence with many many words" not in documents[1].content
        assert "This is a test sentence with many many words" not in documents[2].content

    def test_run_split_by_word_respect_sentence_boundary_with_split_overlap_and_page_breaks(self) -> None:
        document_splitter = DocumentSplitter(
            split_by="word",
            split_length=8,
            split_overlap=1,
            split_threshold=0,
            language="en",
            use_split_rules=True,
            extend_abbreviations=True,
            respect_sentence_boundary=True,
        )
        document_splitter.warm_up()

        text = (
            "Sentence on page 1. Another on page 1.\fSentence on page 2. Another on page 2.\f"
            "Sentence on page 3. Another on page 3.\f\f Sentence on page 5."
        )
        documents = document_splitter.run(documents=[Document(content=text)])["documents"]

        assert len(documents) == 6
        assert documents[0].content == "Sentence on page 1. Another on page 1.\f"
        assert documents[0].meta["page_number"] == 1
        assert documents[0].meta["split_id"] == 0
        assert documents[0].meta["split_idx_start"] == text.index(documents[0].content)
        assert documents[1].content == "Another on page 1.\fSentence on page 2. "
        assert documents[1].meta["page_number"] == 1
        assert documents[1].meta["split_id"] == 1
        assert documents[1].meta["split_idx_start"] == text.index(documents[1].content)
        assert documents[2].content == "Sentence on page 2. Another on page 2.\f"
        assert documents[2].meta["page_number"] == 2
        assert documents[2].meta["split_id"] == 2
        assert documents[2].meta["split_idx_start"] == text.index(documents[2].content)
        assert documents[3].content == "Another on page 2.\fSentence on page 3. "
        assert documents[3].meta["page_number"] == 2
        assert documents[3].meta["split_id"] == 3
        assert documents[3].meta["split_idx_start"] == text.index(documents[3].content)
        assert documents[4].content == "Sentence on page 3. Another on page 3.\f\f "
        assert documents[4].meta["page_number"] == 3
        assert documents[4].meta["split_id"] == 4
        assert documents[4].meta["split_idx_start"] == text.index(documents[4].content)
        assert documents[5].content == "Another on page 3.\f\f Sentence on page 5."
        assert documents[5].meta["page_number"] == 3
        assert documents[5].meta["split_id"] == 5
        assert documents[5].meta["split_idx_start"] == text.index(documents[5].content)

    def test_respect_sentence_boundary_checks(self):
        # this combination triggers the warning
        splitter = DocumentSplitter(split_by="sentence", split_length=10, respect_sentence_boundary=True)
        assert splitter.respect_sentence_boundary == False

    def test_sentence_serialization(self):
        """Test serialization with NLTK sentence splitting configuration and using non-default values"""
        splitter = DocumentSplitter(
            split_by="sentence",
            language="de",
            use_split_rules=False,
            extend_abbreviations=False,
            respect_sentence_boundary=False,
        )
        serialized = splitter.to_dict()
        deserialized = DocumentSplitter.from_dict(serialized)

        assert deserialized.split_by == "sentence"
        assert hasattr(deserialized, "sentence_splitter")
        assert deserialized.language == "de"
        assert deserialized.use_split_rules == False
        assert deserialized.extend_abbreviations == False
        assert deserialized.respect_sentence_boundary == False

    def test_nltk_serialization_roundtrip(self):
        """Test complete serialization roundtrip with actual document splitting"""
        splitter = DocumentSplitter(
            split_by="sentence",
            language="de",
            use_split_rules=False,
            extend_abbreviations=False,
            respect_sentence_boundary=False,
        )
        serialized = splitter.to_dict()
        deserialized_splitter = DocumentSplitter.from_dict(serialized)
        assert splitter.split_by == deserialized_splitter.split_by

    def test_respect_sentence_boundary_serialization(self):
        """Test serialization with respect_sentence_boundary option"""
        splitter = DocumentSplitter(split_by="word", respect_sentence_boundary=True, language="de")
        serialized = splitter.to_dict()
        deserialized = DocumentSplitter.from_dict(serialized)

        assert deserialized.respect_sentence_boundary == True
        assert hasattr(deserialized, "sentence_splitter")
        assert deserialized.language == "de"
