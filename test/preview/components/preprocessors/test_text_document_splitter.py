import pytest

from haystack.preview import Document
from haystack.preview.components.preprocessors import TextDocumentSplitter


class TestTextDocumentSplitter:
    @pytest.mark.unit
    def test_non_text_document(self):
        with pytest.raises(
            ValueError, match="TextDocumentSplitter only works with text documents but document.text is None."
        ):
            splitter = TextDocumentSplitter()
            splitter.run(document=Document())

    @pytest.mark.unit
    def test_unsupported_split_option(self):
        with pytest.raises(
            NotImplementedError, match="PreProcessor only supports 'passage', 'sentence' or 'word' split_by options."
        ):
            splitter = TextDocumentSplitter()
            splitter.run(document=Document(text="text"), split_by="unsupported")

    @pytest.mark.unit
    def test_split_by_word(self):
        splitter = TextDocumentSplitter()
        result = splitter.run(
            document=Document(
                text="This is a text with some words. There is a second sentence. And there is a third sentence."
            ),
            split_by="word",
            split_length=10,
        )
        assert len(result["documents"]) == 2
        assert result["documents"][0].text == "This is a text with some words. There is a"
        assert result["documents"][1].text == "second sentence. And there is a third sentence."

    @pytest.mark.unit
    def test_split_by_sentence(self):
        splitter = TextDocumentSplitter()
        result = splitter.run(
            document=Document(
                text="This is a text with some words. There is a second sentence. And there is a third sentence."
            ),
            split_by="sentence",
            split_length=1,
        )
        assert len(result["documents"]) == 3
        assert result["documents"][0].text == "This is a text with some words"
        assert result["documents"][1].text == " There is a second sentence"
        assert result["documents"][2].text == " And there is a third sentence"

    @pytest.mark.unit
    def test_split_by_passage(self):
        splitter = TextDocumentSplitter()
        result = splitter.run(
            document=Document(
                text="This is a text with some words. There is a second sentence.\n\nAnd there is a third sentence.\n\n And another passage."
            ),
            split_by="passage",
            split_length=1,
        )
        assert len(result["documents"]) == 3
        assert result["documents"][0].text == "This is a text with some words. There is a second sentence."
        assert result["documents"][1].text == "And there is a third sentence."
        assert result["documents"][2].text == " And another passage."

    @pytest.mark.unit
    def test_split_by_word_with_overlap(self):
        splitter = TextDocumentSplitter()
        result = splitter.run(
            document=Document(
                text="This is a text with some words. There is a second sentence. And there is a third sentence."
            ),
            split_by="word",
            split_length=10,
            split_overlap=2,
        )
        assert len(result["documents"]) == 2
        assert result["documents"][0].text == "This is a text with some words. There is a"
        assert result["documents"][1].text == "is a second sentence. And there is a third sentence."

    @pytest.mark.unit
    def test_to_dict(self):
        splitter = TextDocumentSplitter()
        data = splitter.to_dict()
        assert data == {
            "type": "TextDocumentSplitter",
            "init_parameters": {"split_by": "word", "split_length": 200, "split_overlap": 0},
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        splitter = TextDocumentSplitter(split_by="passage", split_length=100, split_overlap=1)
        data = splitter.to_dict()
        assert data == {
            "type": "TextDocumentSplitter",
            "init_parameters": {"split_by": "passage", "split_length": 100, "split_overlap": 1},
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "TextDocumentSplitter",
            "init_parameters": {"split_by": "passage", "split_length": 100, "split_overlap": 1},
        }
        splitter = TextDocumentSplitter.from_dict(data)
        assert splitter.split_by == "passage"
        assert splitter.split_length == 100
        assert splitter.split_overlap == 1
