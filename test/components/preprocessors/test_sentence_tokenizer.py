import pytest
from unittest.mock import patch
from pathlib import Path

from haystack.components.preprocessors.sentence_tokenizer import SentenceSplitter

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
