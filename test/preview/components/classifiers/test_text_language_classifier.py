import logging
import pytest

from haystack.preview import Document
from haystack.preview.components.classifiers import TextLanguageClassifier


class TestTextLanguageClassifier:
    @pytest.mark.unit
    def test_non_string_input(self):
        with pytest.raises(TypeError, match="TextLanguageClassifier expects a str as input."):
            classifier = TextLanguageClassifier()
            classifier.run(text=Document(content="This is an english sentence."))

    @pytest.mark.unit
    def test_list_of_string(self):
        with pytest.raises(TypeError, match="TextLanguageClassifier expects a str as input."):
            classifier = TextLanguageClassifier()
            classifier.run(text=["This is an english sentence."])

    @pytest.mark.unit
    def test_empty_string(self):
        classifier = TextLanguageClassifier()
        result = classifier.run(text="")
        assert result == {"unmatched": ""}

    @pytest.mark.unit
    def test_detect_language(self):
        classifier = TextLanguageClassifier()
        detected_language = classifier.detect_language("This is an english sentence.")
        assert detected_language == "en"

    @pytest.mark.unit
    def test_route_to_en(self):
        classifier = TextLanguageClassifier()
        english_sentence = "This is an english sentence."
        result = classifier.run(text=english_sentence)
        assert result == {"en": english_sentence}

    @pytest.mark.unit
    def test_route_to_unmatched(self):
        classifier = TextLanguageClassifier()
        german_sentence = "Ein deutscher Satz ohne Verb."
        result = classifier.run(text=german_sentence)
        assert result == {"unmatched": german_sentence}

    @pytest.mark.unit
    def test_warning_if_no_language_detected(self, caplog):
        with caplog.at_level(logging.WARNING):
            classifier = TextLanguageClassifier()
            classifier.run(text=".")
            assert "Langdetect cannot detect the language of text: ." in caplog.text
