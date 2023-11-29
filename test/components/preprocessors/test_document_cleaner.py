import logging

import pytest

from haystack import Document
from haystack.components.preprocessors import DocumentCleaner


class TestDocumentCleaner:
    def test_init(self):
        cleaner = DocumentCleaner()
        assert cleaner.remove_empty_lines is True
        assert cleaner.remove_extra_whitespaces is True
        assert cleaner.remove_repeated_substrings is False
        assert cleaner.remove_substrings is None
        assert cleaner.remove_regex is None

    def test_non_text_document(self, caplog):
        with caplog.at_level(logging.WARNING):
            cleaner = DocumentCleaner()
            cleaner.run(documents=[Document()])
            assert "DocumentCleaner only cleans text documents but document.content for document ID" in caplog.text

    def test_single_document(self):
        with pytest.raises(TypeError, match="DocumentCleaner expects a List of Documents as input."):
            cleaner = DocumentCleaner()
            cleaner.run(documents=Document())

    def test_empty_list(self):
        cleaner = DocumentCleaner()
        result = cleaner.run(documents=[])
        assert result == {"documents": []}

    def test_remove_empty_lines(self):
        cleaner = DocumentCleaner(remove_extra_whitespaces=False)
        result = cleaner.run(
            documents=[
                Document(
                    content="This is a text with some words. "
                    ""
                    "There is a second sentence. "
                    ""
                    "And there is a third sentence."
                )
            ]
        )
        assert len(result["documents"]) == 1
        assert (
            result["documents"][0].content
            == "This is a text with some words. There is a second sentence. And there is a third sentence."
        )

    def test_remove_whitespaces(self):
        cleaner = DocumentCleaner(remove_empty_lines=False)
        result = cleaner.run(
            documents=[
                Document(
                    content=" This is a text with some words. "
                    ""
                    "There is a second sentence.  "
                    ""
                    "And there  is a third sentence. "
                )
            ]
        )
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == (
            "This is a text with some words. " "" "There is a second sentence. " "" "And there is a third sentence."
        )

    def test_remove_substrings(self):
        cleaner = DocumentCleaner(remove_substrings=["This", "A", "words", "🪲"])
        result = cleaner.run(documents=[Document(content="This is a text with some words.🪲")])
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == " is a text with some ."

    def test_remove_regex(self):
        cleaner = DocumentCleaner(remove_regex=r"\s\s+")
        result = cleaner.run(documents=[Document(content="This is a  text with   some words.")])
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "This is a text with some words."

    def test_remove_repeated_substrings(self):
        cleaner = DocumentCleaner(
            remove_empty_lines=False, remove_extra_whitespaces=False, remove_repeated_substrings=True
        )

        text = """First PageThis is a header.
        Page  of
        2
        4
        Lorem ipsum dolor sit amet
        This is a footer number 1
        This is footer number 2This is a header.
        Page  of
        3
        4
        Sid ut perspiciatis unde
        This is a footer number 1
        This is footer number 2This is a header.
        Page  of
        4
        4
        Sed do eiusmod tempor.
        This is a footer number 1
        This is footer number 2"""

        expected_text = """First Page 2
        4
        Lorem ipsum dolor sit amet 3
        4
        Sid ut perspiciatis unde 4
        4
        Sed do eiusmod tempor."""
        result = cleaner.run(documents=[Document(content=text)])
        assert result["documents"][0].content == expected_text

    def test_copy_metadata(self):
        cleaner = DocumentCleaner()
        documents = [
            Document(content="Text. ", meta={"name": "doc 0"}),
            Document(content="Text. ", meta={"name": "doc 1"}),
        ]
        result = cleaner.run(documents=documents)
        assert len(result["documents"]) == 2
        assert result["documents"][0].id != result["documents"][1].id
        for doc, cleaned_doc in zip(documents, result["documents"]):
            assert doc.meta == cleaned_doc.meta
            assert cleaned_doc.content == "Text."
