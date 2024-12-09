import pytest
from haystack.components.preprocessors.sentence_tokenizer import SentenceSplitter


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

    def test_document_splitter_split_into_units_sentence(self) -> None:
        text = "Moonlight shimmered softly, wolves howled nearby, night enveloped everything. It was a dark night."
        sentence_splitter = SentenceSplitter()
        units = sentence_splitter.split_sentences(text)
        assert units == [
            {
                "sentence": "Moonlight shimmered softly, wolves howled nearby, night enveloped everything.",
                "start": 0,
                "end": 77,
            },
            {"sentence": "It was a dark night.", "start": 78, "end": 98},
        ]
