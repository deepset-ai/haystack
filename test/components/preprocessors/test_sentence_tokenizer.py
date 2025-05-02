import time
import pytest
from unittest.mock import patch
from pathlib import Path

from haystack.components.preprocessors.sentence_tokenizer import SentenceSplitter
from haystack.components.preprocessors.sentence_tokenizer import QUOTE_SPANS_RE

from pytest import LogCaptureFixture


def test_apply_split_rules_no_join() -> None:
    text = "This is a test. This is another test. And a third test."
    spans = [(0, 15), (16, 36), (37, 54)]
    result = SentenceSplitter._apply_split_rules(text, spans)
    assert len(result) == 3
    assert result == [(0, 15), (16, 36), (37, 54)]


def test_apply_split_rules_join_case_1():
    text = 'He said "This is sentence one. This is sentence two." Then he left.'
    result = SentenceSplitter._apply_split_rules(text, [(0, 30), (31, 53), (54, 67)])
    assert len(result) == 2
    assert result == [(0, 53), (54, 67)]


def test_apply_split_rules_join_case_3():
    splitter = SentenceSplitter(language="en", use_split_rules=True)
    text = """
    1. First item
    2. Second item
    3. Third item."""
    spans = [(0, 7), (8, 25), (26, 44), (45, 56)]
    result = splitter._apply_split_rules(text, spans)
    assert len(result) == 1
    assert result == [(0, 56)]


def test_apply_split_rules_join_case_4() -> None:
    text = "This is a test. (With a parenthetical statement.) And another sentence."
    spans = [(0, 15), (16, 50), (51, 74)]
    result = SentenceSplitter._apply_split_rules(text, spans)
    assert len(result) == 2
    assert result == [(0, 50), (51, 74)]


@pytest.fixture
def mock_file_content():
    return "Mr.\nDr.\nProf."


def test_read_abbreviations_existing_file(tmp_path, mock_file_content):
    abbrev_dir = tmp_path / "data" / "abbreviations"
    abbrev_dir.mkdir(parents=True)
    abbrev_file = abbrev_dir / f"en.txt"
    abbrev_file.write_text(mock_file_content)

    with patch("haystack.components.preprocessors.sentence_tokenizer.Path") as mock_path:
        mock_path.return_value.parent.parent.parent = tmp_path
        result = SentenceSplitter._read_abbreviations("en")
        assert result == ["Mr.", "Dr.", "Prof."]


def test_read_abbreviations_missing_file(caplog: LogCaptureFixture):
    with patch("haystack.components.preprocessors.sentence_tokenizer.Path") as mock_path:
        mock_path.return_value.parent.parent = Path("/nonexistent")
        result = SentenceSplitter._read_abbreviations("pt")
        assert result == []
        assert "No abbreviations file found for pt. Using default abbreviations." in caplog.text


def test_quote_spans_regex():
    # double quotes
    text1 = 'He said "Hello world" and left.'
    matches1 = list(QUOTE_SPANS_RE.finditer(text1))
    assert len(matches1) == 1
    assert matches1[0].group() == '"Hello world"'

    # single quotes
    text2 = "She replied 'Goodbye world' and smiled."
    matches2 = list(QUOTE_SPANS_RE.finditer(text2))
    assert len(matches2) == 1
    assert matches2[0].group() == "'Goodbye world'"

    # multiple quotes
    text3 = 'First "quote" and second "quote" in same text.'
    matches3 = list(QUOTE_SPANS_RE.finditer(text3))
    assert len(matches3) == 2
    assert matches3[0].group() == '"quote"'
    assert matches3[1].group() == '"quote"'

    # quotes containing newlines
    text4 = 'Text with "quote\nspanning\nmultiple\nlines"'
    matches4 = list(QUOTE_SPANS_RE.finditer(text4))
    assert len(matches4) == 1
    assert matches4[0].group() == '"quote\nspanning\nmultiple\nlines"'

    # no quotes
    text5 = "This text has no quotes."
    matches5 = list(QUOTE_SPANS_RE.finditer(text5))
    assert len(matches5) == 0


def test_split_sentences_performance() -> None:
    # make sure our regex is not vulnerable to Regex Denial of Service (ReDoS)
    # https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS
    # this is a very long string, roughly 500 MB, but it should not take more than 2 seconds to process
    splitter = SentenceSplitter()
    text = " " + '"' * 20 + "A" * 50000000 + "B"
    start = time.time()
    _ = splitter.split_sentences(text)
    end = time.time()

    assert end - start < 2, f"Execution time exceeded 2 seconds: {end - start:.2f} seconds"
