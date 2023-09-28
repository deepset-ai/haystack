import pytest

from haystack.preview import Document
from haystack.preview.components.preprocessors import TextDocumentSplitter


class TestTextDocumentSplitter:
    @pytest.mark.unit
    def test_non_text_document(self):
        with pytest.raises(
            ValueError, match="TextDocumentSplitter only works with text documents but document.text for document ID"
        ):
            splitter = TextDocumentSplitter()
            splitter.run(documents=[Document()])

    @pytest.mark.unit
    def test_single_doc(self):
        with pytest.raises(TypeError, match="TextDocumentSplitter expects a List of Documents as input."):
            splitter = TextDocumentSplitter()
            splitter.run(documents=Document())

    @pytest.mark.unit
    def test_empty_list(self):
        with pytest.raises(TypeError, match="TextDocumentSplitter expects a List of Documents as input."):
            splitter = TextDocumentSplitter()
            splitter.run(documents=[])

    @pytest.mark.unit
    def test_unsupported_split_by(self):
        with pytest.raises(ValueError, match="split_by must be one of 'word', 'sentence' or 'passage'."):
            TextDocumentSplitter(split_by="unsupported")

    @pytest.mark.unit
    def test_unsupported_split_length(self):
        with pytest.raises(ValueError, match="split_length must be greater than 0."):
            TextDocumentSplitter(split_length=0)

    @pytest.mark.unit
    def test_unsupported_split_overlap(self):
        with pytest.raises(ValueError, match="split_overlap must be greater than or equal to 0."):
            TextDocumentSplitter(split_overlap=-1)

    @pytest.mark.unit
    def test_split_by_word(self):
        splitter = TextDocumentSplitter(split_by="word", split_length=10)
        result = splitter.run(
            documents=[
                Document(
                    text="This is a text with some words. There is a second sentence. And there is a third sentence."
                )
            ]
        )
        assert len(result["documents"]) == 2
        assert result["documents"][0].text == "This is a text with some words. There is a "
        assert result["documents"][1].text == "second sentence. And there is a third sentence."

    @pytest.mark.unit
    def test_split_by_word_multiple_input_docs(self):
        splitter = TextDocumentSplitter(split_by="word", split_length=10)
        result = splitter.run(
            documents=[
                Document(
                    text="This is a text with some words. There is a second sentence. And there is a third sentence."
                ),
                Document(
                    text="This is a different text with some words. There is a second sentence. And there is a third sentence. And there is a fourth sentence."
                ),
            ]
        )
        assert len(result["documents"]) == 5
        assert result["documents"][0].text == "This is a text with some words. There is a "
        assert result["documents"][1].text == "second sentence. And there is a third sentence."
        assert result["documents"][2].text == "This is a different text with some words. There is "
        assert result["documents"][3].text == "a second sentence. And there is a third sentence. And "
        assert result["documents"][4].text == "there is a fourth sentence."

    @pytest.mark.unit
    def test_split_by_sentence(self):
        splitter = TextDocumentSplitter(split_by="sentence", split_length=1)
        result = splitter.run(
            documents=[
                Document(
                    text="This is a text with some words. There is a second sentence. And there is a third sentence."
                )
            ]
        )
        assert len(result["documents"]) == 3
        assert result["documents"][0].text == "This is a text with some words."
        assert result["documents"][1].text == " There is a second sentence."
        assert result["documents"][2].text == " And there is a third sentence."

    @pytest.mark.unit
    def test_split_by_passage(self):
        splitter = TextDocumentSplitter(split_by="passage", split_length=1)
        result = splitter.run(
            documents=[
                Document(
                    text="This is a text with some words. There is a second sentence.\n\nAnd there is a third sentence.\n\n And another passage."
                )
            ]
        )
        assert len(result["documents"]) == 3
        assert result["documents"][0].text == "This is a text with some words. There is a second sentence.\n\n"
        assert result["documents"][1].text == "And there is a third sentence.\n\n"
        assert result["documents"][2].text == " And another passage."

    @pytest.mark.unit
    def test_split_by_word_with_overlap(self):
        splitter = TextDocumentSplitter(split_by="word", split_length=10, split_overlap=2)
        result = splitter.run(
            documents=[
                Document(
                    text="This is a text with some words. There is a second sentence. And there is a third sentence."
                )
            ]
        )
        assert len(result["documents"]) == 2
        assert result["documents"][0].text == "This is a text with some words. There is a "
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

    @pytest.mark.unit
    def test_source_id_stored_in_metadata(self):
        splitter = TextDocumentSplitter(split_by="word", split_length=10)
        doc1 = Document(text="This is a text with some words.")
        doc2 = Document(text="This is a different text with some words.")
        result = splitter.run(documents=[doc1, doc2])
        assert result["documents"][0].metadata["source_id"] == doc1.id
        assert result["documents"][1].metadata["source_id"] == doc2.id
