# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
import pytest
from _pytest.logging import LogCaptureFixture

from haystack import Document
from haystack.components.routers import TextLanguageRouter


class TestTextLanguageRouter:
    def test_non_string_input(self):
        with pytest.raises(TypeError, match="TextLanguageRouter expects a string as input."):
            classifier = TextLanguageRouter()
            classifier.run(text=Document(content="This is an english sentence."))

    def test_list_of_string(self):
        with pytest.raises(TypeError, match="TextLanguageRouter expects a string as input."):
            classifier = TextLanguageRouter()
            classifier.run(text=["This is an english sentence."])

    def test_empty_string(self):
        classifier = TextLanguageRouter()
        result = classifier.run(text="")
        assert result == {"unmatched": ""}

    def test_detect_language(self):
        classifier = TextLanguageRouter()
        detected_language = classifier._detect_language("This is an english sentence.")
        assert detected_language == "en"

    def test_route_to_en(self):
        classifier = TextLanguageRouter()
        english_sentence = "This is an english sentence."
        result = classifier.run(text=english_sentence)
        assert result == {"en": english_sentence}

    def test_route_to_unmatched(self):
        classifier = TextLanguageRouter()
        german_sentence = "Ein deutscher Satz ohne Verb."
        result = classifier.run(text=german_sentence)
        assert result == {"unmatched": german_sentence}

    def test_warning_if_no_language_detected(self, caplog: LogCaptureFixture):
        with caplog.at_level(logging.WARNING):
            classifier = TextLanguageRouter()
            classifier.run(text=".")
            assert "Langdetect cannot detect the language of text. Error: No features in text." in caplog.text

    def test_warning_if_no_language_detected_if_debug(self, caplog: LogCaptureFixture):
        with caplog.at_level(logging.DEBUG):
            classifier = TextLanguageRouter()
            classifier.run(text=".")
            assert "Langdetect cannot detect the language of text. Error: No features in text." in caplog.text
            assert "Langdetect cannot detect the language of text: ." in caplog.text
