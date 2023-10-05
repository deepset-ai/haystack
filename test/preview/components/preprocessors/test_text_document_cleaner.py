import pytest

from haystack.preview import Document
from haystack.preview.components.preprocessors import TextDocumentCleaner


class TestTextDocumentCleaner:
    @pytest.mark.unit
    def test_non_text_document(self):
        with pytest.raises(
            ValueError, match="TextDocumentCleaner only works with text documents but document.text for document ID"
        ):
            cleaner = TextDocumentCleaner()
            cleaner.run(documents=[Document()])

    @pytest.mark.unit
    def test_single_doc(self):
        with pytest.raises(TypeError, match="TextDocumentCleaner expects a List of Documents as input."):
            cleaner = TextDocumentCleaner()
            cleaner.run(documents=Document())

    @pytest.mark.unit
    def test_empty_list(self):
        cleaner = TextDocumentCleaner()
        result = cleaner.run(documents=[])
        assert result == {"documents": []}

    @pytest.mark.unit
    def test_clean_empty_lines(self):
        cleaner = TextDocumentCleaner(remove_extra_whitespaces=False)
        result = cleaner.run(
            documents=[
                Document(
                    text="This is a text with some words. "
                    ""
                    "There is a second sentence. "
                    ""
                    "And there is a third sentence."
                )
            ]
        )
        assert len(result["documents"]) == 1
        assert (
            result["documents"][0].text
            == "This is a text with some words. There is a second sentence. And there is a third sentence."
        )

    @pytest.mark.unit
    def test_clean_whitespaces(self):
        cleaner = TextDocumentCleaner(clean_empty_lines=False)
        result = cleaner.run(
            documents=[
                Document(
                    text=" This is a text with some words. "
                    ""
                    "There is a second sentence.  "
                    ""
                    "And there  is a third sentence. "
                )
            ]
        )
        assert len(result["documents"]) == 1
        assert result["documents"][0].text == (
            "This is a text with some words. " "" "There is a second sentence. " "" "And there is a third sentence."
        )

    @pytest.mark.unit
    def test_remove_substrings(self):
        cleaner = TextDocumentCleaner(remove_substrings=["This", "A", "words"])
        result = cleaner.run(documents=[Document(text="This is a text with some words.")])
        assert len(result["documents"]) == 1
        assert result["documents"][0].text == (" is a text with some .")

    @pytest.mark.unit
    def test_remove_regex(self):
        cleaner = TextDocumentCleaner(remove_regex=r"\s\s+")
        result = cleaner.run(documents=[Document(text="This is a  text with   some words.")])
        assert len(result["documents"]) == 1
        assert result["documents"][0].text == ("This is a text with some words.")
