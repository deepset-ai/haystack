# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re

from haystack import component, logging
from haystack.dataclasses import ChatMessage

logger = logging.getLogger(__name__)


def _backward_compatible(cls):
    """
    Decorator for backward compatibility with the removed `return_empty_on_no_match` parameter.

    Using __new__ or a metaclass does not work because of interference with the @component decorator.
    """
    original_init = cls.__init__

    msg = (
        "The `return_empty_on_no_match` init parameter has been removed and will be ignored. "
        "RegexTextExtractor now always returns `{'captured_text': ''}` when no match is found. "
        "Starting from Haystack 2.23.0, initializing the component with `return_empty_on_no_match` will raise an error."
    )

    def __init__(self, *args, **kwargs):
        if "return_empty_on_no_match" in kwargs:
            logger.warning(msg)
            kwargs.pop("return_empty_on_no_match")
        elif len(args) > 1:
            logger.warning(msg)
            args = args[:1]
        original_init(self, *args, **kwargs)

    cls.__init__ = __init__
    return cls


@_backward_compatible
@component
class RegexTextExtractor:
    """
    Extracts text from chat message or string input using a regex pattern.

    RegexTextExtractor parses input text or ChatMessages using a provided regular expression pattern.
    It can be configured to search through all messages or only the last message in a list of ChatMessages.

    ### Usage example

    ```python
    from haystack.components.extractors import RegexTextExtractor
    from haystack.dataclasses import ChatMessage

    # Using with a string
    parser = RegexTextExtractor(regex_pattern='<issue url=\"(.+)\">')
    result = parser.run(text_or_messages='<issue url="github.com/hahahaha">hahahah</issue>')
    # result: {"captured_text": "github.com/hahahaha"}

    # Using with ChatMessages
    messages = [ChatMessage.from_user('<issue url="github.com/hahahaha">hahahah</issue>')]
    result = parser.run(text_or_messages=messages)
    # result: {"captured_text": "github.com/hahahaha"}
    ```
    """

    def __init__(self, regex_pattern: str):
        """
        Creates an instance of the RegexTextExtractor component.

        :param regex_pattern:
            The regular expression pattern used to extract text.
            The pattern should include a capture group to extract the desired text.
            Example: `'<issue url="(.+)">'` captures `'github.com/hahahaha'` from `'<issue url="github.com/hahahaha">'`.
        """
        self.regex_pattern = regex_pattern

        # Check if the pattern has at least one capture group
        num_groups = re.compile(regex_pattern).groups
        if num_groups < 1:
            logger.warning(
                "The provided regex pattern {regex_pattern} doesn't contain any capture groups. "
                "The entire match will be returned instead.",
                regex_pattern=regex_pattern,
            )

    @component.output_types(captured_text=str)
    def run(self, text_or_messages: str | list[ChatMessage]) -> dict[str, str]:
        """
        Extracts text from input using the configured regex pattern.

        :param text_or_messages:
            Either a string or a list of ChatMessage objects to search through.

        :returns:
          - `{"captured_text": "matched text"}` if a match is found
          - `{"captured_text": ""}` if no match is found

        :raises:
            - ValueError: if receiving a list the last element is not a ChatMessage instance.
        """
        if isinstance(text_or_messages, str):
            return self._build_result(self._extract_from_text(text_or_messages))
        if not text_or_messages:
            logger.warning("Received empty list of messages")
            return {"captured_text": ""}
        return self._process_last_message(text_or_messages)

    def _build_result(self, result: str | list[str]) -> dict:
        """Helper method to build the return dictionary based on configuration."""
        if (isinstance(result, str) and result == "") or (isinstance(result, list) and not result):
            return {"captured_text": ""}
        return {"captured_text": result}

    def _process_last_message(self, messages: list[ChatMessage]) -> dict:
        """Process only the last message and build the result."""
        last_message = messages[-1]
        if not isinstance(last_message, ChatMessage):
            raise ValueError(f"Expected ChatMessage object, got {type(last_message)}")
        if last_message.text is None:
            logger.warning("Last message has no text content")
            return {"captured_text": ""}
        result = self._extract_from_text(last_message.text)
        return self._build_result(result)

    def _extract_from_text(self, text: str) -> str | list[str]:
        """
        Extract text using the regex pattern.

        :param text:
            The text to search through.

        :returns:
            The text captured by the first capturing group in the regex pattern.
            If the pattern has no capture groups, returns the entire match.
            If no match is found, returns an empty string.
        """
        match = re.search(self.regex_pattern, text)
        if not match:
            return ""
        if match.groups():
            return match.group(1)
        return match.group(0)
