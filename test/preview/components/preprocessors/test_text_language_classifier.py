import pytest
import logging

from haystack.preview import Document
from haystack.preview.components.preprocessors import TextLanguageClassifier


class TestTextLanguageClassifier:
    @pytest.mark.unit
    def test_non_string_input(self):
        with pytest.raises(TypeError, match="TextLanguageClassifier expects a list of str as input."):
            classifier = TextLanguageClassifier()
            classifier.run(strings=Document(text="This is an english sentence."))

    @pytest.mark.unit
    def test_single_string(self):
        with pytest.raises(TypeError, match="TextLanguageClassifier expects a list of str as input."):
            classifier = TextLanguageClassifier()
            classifier.run(strings="This is an english sentence.")

    @pytest.mark.unit
    def test_empty_list(self):
        classifier = TextLanguageClassifier()
        result = classifier.run(strings=[])
        assert result == {"en": [], "unmatched": []}

    @pytest.mark.unit
    def test_detect_language(self):
        classifier = TextLanguageClassifier()
        detected_language = classifier.detect_language("This is an english sentence.")
        assert detected_language == "en"

    @pytest.mark.unit
    def test_route_to_en_and_unmatched(self):
        classifier = TextLanguageClassifier()
        english_sentence = "This is an english sentence."
        german_setence = "Ein deutscher Satz ohne Verb."
        result = classifier.run(strings=[english_sentence, german_setence])
        assert result == {"en": [english_sentence], "unmatched": [german_setence]}

    @pytest.mark.unit
    def test_warning_if_no_language_detected(self, caplog):
        with caplog.at_level(logging.WARNING):
            classifier = TextLanguageClassifier()
            classifier.run(strings=["."])
            assert "Langdetect cannot detect the language of text: ." in caplog.text
