import logging

import pytest

from haystack.preview import Document
from haystack.preview.components.preprocessors import DocumentCleaner


class TestDocumentCleaner:
    @pytest.mark.unit
    def test_init(self):
        cleaner = DocumentCleaner()
        assert cleaner.remove_empty_lines == True
        assert cleaner.remove_extra_whitespaces == True
        assert cleaner.remove_repeated_substrings == False
        assert cleaner.remove_substrings is None
        assert cleaner.remove_regex is None

    @pytest.mark.unit
    def test_to_dict(self):
        cleaner = DocumentCleaner()
        data = cleaner.to_dict()
        assert data == {
            "type": "DocumentCleaner",
            "init_parameters": {
                "remove_empty_lines": True,
                "remove_extra_whitespaces": True,
                "remove_repeated_substrings": False,
                "remove_substrings": None,
                "remove_regex": None,
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        cleaner = DocumentCleaner(
            remove_empty_lines=False,
            remove_extra_whitespaces=False,
            remove_repeated_substrings=True,
            remove_substrings=["a", "b"],
            remove_regex=r"\s\s+",
        )
        data = cleaner.to_dict()
        assert data == {
            "type": "DocumentCleaner",
            "init_parameters": {
                "remove_empty_lines": False,
                "remove_extra_whitespaces": False,
                "remove_repeated_substrings": True,
                "remove_substrings": ["a", "b"],
                "remove_regex": r"\s\s+",
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "DocumentCleaner",
            "init_parameters": {
                "remove_empty_lines": False,
                "remove_extra_whitespaces": False,
                "remove_repeated_substrings": True,
                "remove_substrings": ["a", "b"],
                "remove_regex": r"\s\s+",
            },
        }
        cleaner = DocumentCleaner.from_dict(data)
        assert cleaner.remove_empty_lines == False
        assert cleaner.remove_extra_whitespaces == False
        assert cleaner.remove_repeated_substrings == True
        assert cleaner.remove_substrings == ["a", "b"]
        assert cleaner.remove_regex == r"\s\s+"

    @pytest.mark.unit
    def test_non_text_document(self, caplog):
        with caplog.at_level(logging.WARNING):
            cleaner = DocumentCleaner()
            cleaner.run(documents=[Document()])
            assert "DocumentCleaner only cleans text documents but document.text for document ID" in caplog.text

    @pytest.mark.unit
    def test_single_document(self):
        with pytest.raises(TypeError, match="DocumentCleaner expects a List of Documents as input."):
            cleaner = DocumentCleaner()
            cleaner.run(documents=Document())

    @pytest.mark.unit
    def test_empty_list(self):
        cleaner = DocumentCleaner()
        result = cleaner.run(documents=[])
        assert result == {"documents": []}

    @pytest.mark.unit
    def test_remove_empty_lines(self):
        cleaner = DocumentCleaner(remove_extra_whitespaces=False)
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
    def test_remove_whitespaces(self):
        cleaner = DocumentCleaner(remove_empty_lines=False)
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
        cleaner = DocumentCleaner(remove_substrings=["This", "A", "words", "🪲"])
        result = cleaner.run(documents=[Document(text="This is a text with some words.🪲")])
        assert len(result["documents"]) == 1
        assert result["documents"][0].text == " is a text with some ."

    @pytest.mark.unit
    def test_remove_regex(self):
        cleaner = DocumentCleaner(remove_regex=r"\s\s+")
        result = cleaner.run(documents=[Document(text="This is a  text with   some words.")])
        assert len(result["documents"]) == 1
        assert result["documents"][0].text == "This is a text with some words."

    @pytest.mark.unit
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
        result = cleaner.run(documents=[Document(text=text)])
        assert result["documents"][0].text == expected_text
