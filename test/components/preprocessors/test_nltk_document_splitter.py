from typing import List

import pytest
from haystack import Document
from pytest import LogCaptureFixture

from haystack.components.preprocessors.nltk_document_splitter import NLTKDocumentSplitter, SentenceSplitter
from haystack.utils import deserialize_callable


def test_init_warning_message(caplog: LogCaptureFixture) -> None:
    _ = NLTKDocumentSplitter(split_by="page", respect_sentence_boundary=True)
    assert "The 'respect_sentence_boundary' option is only supported for" in caplog.text


def custom_split(text):
    return text.split(".")


class TestNLTKDocumentSplitterSplitIntoUnits:
    def test_document_splitter_split_into_units_word(self) -> None:
        document_splitter = NLTKDocumentSplitter(
            split_by="word", split_length=3, split_overlap=0, split_threshold=0, language="en"
        )

        text = "Moonlight shimmered softly, wolves howled nearby, night enveloped everything."
        units = document_splitter._split_into_units(text=text, split_by="word")

        assert units == [
            "Moonlight ",
            "shimmered ",
            "softly, ",
            "wolves ",
            "howled ",
            "nearby, ",
            "night ",
            "enveloped ",
            "everything.",
        ]

    def test_document_splitter_split_into_units_sentence(self) -> None:
        document_splitter = NLTKDocumentSplitter(
            split_by="sentence", split_length=2, split_overlap=0, split_threshold=0, language="en"
        )
        document_splitter.warm_up()

        text = "Moonlight shimmered softly, wolves howled nearby, night enveloped everything. It was a dark night."
        units = document_splitter._split_into_units(text=text, split_by="sentence")

        assert units == [
            "Moonlight shimmered softly, wolves howled nearby, night enveloped everything. ",
            "It was a dark night.",
        ]

    def test_document_splitter_split_into_units_passage(self) -> None:
        document_splitter = NLTKDocumentSplitter(
            split_by="passage", split_length=2, split_overlap=0, split_threshold=0, language="en"
        )

        text = "Moonlight shimmered softly, wolves howled nearby, night enveloped everything.\n\nIt was a dark night."
        units = document_splitter._split_into_units(text=text, split_by="passage")

        assert units == [
            "Moonlight shimmered softly, wolves howled nearby, night enveloped everything.\n\n",
            "It was a dark night.",
        ]

    def test_document_splitter_split_into_units_page(self) -> None:
        document_splitter = NLTKDocumentSplitter(
            split_by="page", split_length=2, split_overlap=0, split_threshold=0, language="en"
        )

        text = "Moonlight shimmered softly, wolves howled nearby, night enveloped everything.\fIt was a dark night."
        units = document_splitter._split_into_units(text=text, split_by="page")

        assert units == [
            "Moonlight shimmered softly, wolves howled nearby, night enveloped everything.\f",
            "It was a dark night.",
        ]

    def test_document_splitter_split_into_units_raise_error(self) -> None:
        document_splitter = NLTKDocumentSplitter(
            split_by="word", split_length=3, split_overlap=0, split_threshold=0, language="en"
        )

        text = "Moonlight shimmered softly, wolves howled nearby, night enveloped everything."

        with pytest.raises(NotImplementedError):
            document_splitter._split_into_units(text=text, split_by="invalid")  # type: ignore


class TestNLTKDocumentSplitterNumberOfSentencesToKeep:
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
        num_sentences = NLTKDocumentSplitter._number_of_sentences_to_keep(
            sentences=sentences, split_length=5, split_overlap=2
        )
        assert num_sentences == expected_num_sentences

    def test_number_of_sentences_to_keep_split_overlap_zero(self) -> None:
        sentences = [
            "Moonlight shimmered softly, wolves howled nearby, night enveloped everything.",
            " It was a dark night ...",
            " The moon was full.",
        ]
        num_sentences = NLTKDocumentSplitter._number_of_sentences_to_keep(
            sentences=sentences, split_length=5, split_overlap=0
        )
        assert num_sentences == 0


class TestNLTKDocumentSplitterRun:
    def test_run_type_error(self) -> None:
        document_splitter = NLTKDocumentSplitter()
        with pytest.raises(TypeError):
            document_splitter.warm_up()
            document_splitter.run(documents=Document(content="Moonlight shimmered softly."))  # type: ignore

    def test_run_value_error(self) -> None:
        document_splitter = NLTKDocumentSplitter()
        with pytest.raises(ValueError):
            document_splitter.warm_up()
            document_splitter.run(documents=[Document(content=None)])

    def test_run_split_by_sentence_1(self) -> None:
        document_splitter = NLTKDocumentSplitter(
            split_by="sentence",
            split_length=2,
            split_overlap=0,
            split_threshold=0,
            language="en",
            use_split_rules=True,
            extend_abbreviations=True,
        )
        document_splitter.warm_up()

        text = (
            "Moonlight shimmered softly, wolves howled nearby, night enveloped everything. It was a dark night ... "
            "The moon was full."
        )
        documents = document_splitter.run(documents=[Document(content=text)])["documents"]

        assert len(documents) == 2
        assert (
            documents[0].content == "Moonlight shimmered softly, wolves howled nearby, night enveloped "
            "everything. It was a dark night ... "
        )
        assert documents[1].content == "The moon was full."

    def test_run_split_by_sentence_2(self) -> None:
        document_splitter = NLTKDocumentSplitter(
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
        document_splitter = NLTKDocumentSplitter(
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
        document_splitter = NLTKDocumentSplitter(
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


class TestNLTKDocumentSplitterRespectSentenceBoundary:
    def test_run_split_by_word_respect_sentence_boundary(self) -> None:
        document_splitter = NLTKDocumentSplitter(
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
        document_splitter = NLTKDocumentSplitter(
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
        document_splitter = NLTKDocumentSplitter(
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

    def test_to_dict(self):
        splitter = NLTKDocumentSplitter(split_by="word", split_length=10, split_overlap=2, split_threshold=5)
        serialized = splitter.to_dict()

        assert serialized["type"] == "haystack.components.preprocessors.nltk_document_splitter.NLTKDocumentSplitter"
        assert serialized["init_parameters"]["split_by"] == "word"
        assert serialized["init_parameters"]["split_length"] == 10
        assert serialized["init_parameters"]["split_overlap"] == 2
        assert serialized["init_parameters"]["split_threshold"] == 5
        assert serialized["init_parameters"]["language"] == "en"
        assert serialized["init_parameters"]["use_split_rules"] is True
        assert serialized["init_parameters"]["extend_abbreviations"] is True
        assert "splitting_function" not in serialized["init_parameters"]

    def test_to_dict_with_splitting_function(self):
        splitter = NLTKDocumentSplitter(split_by="function", splitting_function=custom_split)
        serialized = splitter.to_dict()

        assert serialized["type"] == "haystack.components.preprocessors.nltk_document_splitter.NLTKDocumentSplitter"
        assert serialized["init_parameters"]["split_by"] == "function"
        assert "splitting_function" in serialized["init_parameters"]
        assert callable(deserialize_callable(serialized["init_parameters"]["splitting_function"]))


class TestSentenceSplitter:
    def test_apply_split_rules_second_while_loop(self) -> None:
        text = "This is a test. (With a parenthetical statement.) And another sentence."
        spans = [(0, 15), (16, 50), (51, 74)]
        result = SentenceSplitter._apply_split_rules(text, spans)
        assert len(result) == 2
        assert result == [(0, 50), (51, 74)]

    def test_apply_split_rules_no_join(self) -> None:
        text = "This is a test. This is another test. And a third test."
        spans = [(0, 15), (16, 36), (37, 54)]
        result = SentenceSplitter._apply_split_rules(text, spans)
        assert len(result) == 3
        assert result == [(0, 15), (16, 36), (37, 54)]

    @pytest.mark.parametrize(
        "text,span,next_span,quote_spans,expected",
        [
            # triggers sentence boundary is inside a quote
            ('He said, "Hello World." Then left.', (0, 15), (16, 23), [(9, 23)], True)
        ],
    )
    def test_needs_join_cases(self, text, span, next_span, quote_spans, expected):
        result = SentenceSplitter._needs_join(text, span, next_span, quote_spans)
        assert result == expected, f"Expected {expected} for input: {text}, {span}, {next_span}, {quote_spans}"
