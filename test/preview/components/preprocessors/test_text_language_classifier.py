import pytest

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
        assert result == {"english": [], "unmatched": []}
