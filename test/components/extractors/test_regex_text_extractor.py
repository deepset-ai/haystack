# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline
from haystack.components.extractors.regex_text_extractor import RegexTextExtractor
from haystack.dataclasses import ChatMessage


class TestRegexTextExtractor:
    def test_init_with_capture_group(self):
        pattern = r'<issue url="(.+?)">'
        extractor = RegexTextExtractor(regex_pattern=pattern)
        assert extractor.regex_pattern == pattern

    def test_init_without_capture_group(self):
        pattern = r"<issue>"
        extractor = RegexTextExtractor(regex_pattern=pattern)
        assert extractor.regex_pattern == pattern

    def test_extract_from_string_with_capture_group(self):
        pattern = r'<issue url="(.+?)">'
        extractor = RegexTextExtractor(regex_pattern=pattern)
        text = '<issue url="github.com/hahahaha">hahahah</issue>'
        result = extractor.run(text_or_messages=text)
        assert result == {"captured_text": "github.com/hahahaha"}

    def test_extract_from_string_without_capture_group(self):
        pattern = r"<issue>"
        extractor = RegexTextExtractor(regex_pattern=pattern)
        text = "This is an <issue> tag in the text"
        result = extractor.run(text_or_messages=text)
        assert result == {"captured_text": "<issue>"}

    def test_extract_from_string_no_match(self):
        pattern = r'<issue url="(.+?)">'
        extractor = RegexTextExtractor(regex_pattern=pattern)
        text = "This text has no matching pattern"
        result = extractor.run(text_or_messages=text)
        assert result == {}

    def test_extract_from_string_empty_input(self):
        pattern = r'<issue url="(.+?)">'
        extractor = RegexTextExtractor(regex_pattern=pattern)
        text = ""
        result = extractor.run(text_or_messages=text)
        assert result == {}

    def test_extract_from_chat_messages_single_message(self):
        pattern = r'<issue url="(.+?)">'
        extractor = RegexTextExtractor(regex_pattern=pattern)
        messages = [ChatMessage.from_user('<issue url="github.com/test">test issue</issue>')]
        result = extractor.run(text_or_messages=messages)
        assert result == {"captured_text": "github.com/test"}

    def test_extract_from_chat_messages_multiple_messages(self):
        pattern = r'<issue url="(.+?)">'
        extractor = RegexTextExtractor(regex_pattern=pattern)
        messages = [
            ChatMessage.from_user('First message with <issue url="first.com">first</issue>'),
            ChatMessage.from_user('Second message with <issue url="second.com">second</issue>'),
            ChatMessage.from_user('Last message with <issue url="last.com">last</issue>'),
        ]
        result = extractor.run(text_or_messages=messages)
        assert result == {"captured_text": "last.com"}

    def test_extract_from_chat_messages_no_match_in_last(self):
        pattern = r'<issue url="(.+?)">'
        extractor = RegexTextExtractor(regex_pattern=pattern)
        messages = [
            ChatMessage.from_user('First message with <issue url="first.com">first</issue>'),
            ChatMessage.from_user("Last message with no matching pattern"),
        ]
        result = extractor.run(text_or_messages=messages)
        assert result == {}

    def test_extract_from_chat_messages_empty_list(self):
        pattern = r'<issue url="(.+?)">'
        extractor = RegexTextExtractor(regex_pattern=pattern)
        messages = []
        result = extractor.run(text_or_messages=messages)
        assert result == {}

    def test_extract_from_chat_messages_invalid_type(self):
        pattern = r'<issue url="(.+?)">'
        extractor = RegexTextExtractor(regex_pattern=pattern)
        messages = ["not a ChatMessage object"]
        with pytest.raises(ValueError, match="Expected ChatMessage object, got <class 'str'>"):
            extractor.run(text_or_messages=messages)

    def test_multiple_capture_groups(self):
        pattern = r"(\w+)@(\w+)\.(\w+)"
        extractor = RegexTextExtractor(regex_pattern=pattern)
        text = "Contact us at user@example.com for support"
        result = extractor.run(text_or_messages=text)
        # return the first capture group (username)
        assert result == {"captured_text": "user"}

    def test_special_characters_in_pattern(self):
        """Test regex pattern with special characters."""
        pattern = r"\[(\w+)\]"
        extractor = RegexTextExtractor(regex_pattern=pattern)

        text = "This has [special] characters [in] brackets"
        result = extractor.run(text_or_messages=text)

        assert result == {"captured_text": "special"}

    def test_whitespace_handling(self):
        """Test regex pattern with whitespace handling."""
        pattern = r"\s+(\w+)\s+"
        extractor = RegexTextExtractor(regex_pattern=pattern)

        text = "word1   word2   word3"
        result = extractor.run(text_or_messages=text)

        assert result == {"captured_text": "word2"}

    def test_nested_capture_groups(self):
        """Test regex with nested capture groups."""
        pattern = r'<(\w+)\s+attr="([^"]+)">'
        extractor = RegexTextExtractor(regex_pattern=pattern)

        text = '<div attr="value">content</div>'
        result = extractor.run(text_or_messages=text)

        # Should return the first capture group (tag name)
        assert result == {"captured_text": "div"}

    def test_optional_capture_group(self):
        """Test regex with optional capture group."""
        pattern = r"(\w+)(?:@(\w+))?"
        extractor = RegexTextExtractor(regex_pattern=pattern)

        text = "username@domain"
        result = extractor.run(text_or_messages=text)

        assert result == {"captured_text": "username"}

    def test_optional_capture_group_no_match(self):
        """Test regex with optional capture group when optional part is missing."""
        pattern = r"(\w+)(?:@(\w+))?"
        extractor = RegexTextExtractor(regex_pattern=pattern)

        text = "username"
        result = extractor.run(text_or_messages=text)

        assert result == {"captured_text": "username"}

    def test_pipeline_integration(self):
        """Test component integration in a Haystack pipeline."""
        pattern = r'<issue url="(.+?)">'
        extractor = RegexTextExtractor(regex_pattern=pattern)

        pipe = Pipeline()
        pipe.add_component("extractor", extractor)

        text = '<issue url="github.com/pipeline-test">pipeline test</issue>'
        result = pipe.run(data={"extractor": {"text_or_messages": text}})

        assert result["extractor"] == {"captured_text": "github.com/pipeline-test"}

    def test_pipeline_integration_with_chat_messages(self):
        """Test component integration in pipeline with ChatMessages."""
        pattern = r'<issue url="(.+?)">'
        extractor = RegexTextExtractor(regex_pattern=pattern)

        pipe = Pipeline()
        pipe.add_component("extractor", extractor)

        messages = [ChatMessage.from_user('<issue url="github.com/chat-test">chat test</issue>')]
        result = pipe.run(data={"extractor": {"text_or_messages": messages}})

        assert result["extractor"] == {"captured_text": "github.com/chat-test"}

    def test_very_long_text(self):
        pattern = r"(\d+)"
        extractor = RegexTextExtractor(regex_pattern=pattern)
        long_text = "a" * 10000 + "123" + "b" * 10000
        result = extractor.run(text_or_messages=long_text)
        assert result == {"captured_text": "123"}

    def test_multiple_matches_first_is_captured(self):
        pattern = r"(\d+)"
        extractor = RegexTextExtractor(regex_pattern=pattern)
        text = "First: 123, Second: 456, Third: 789"
        result = extractor.run(text_or_messages=text)
        assert result == {"captured_text": "123"}
