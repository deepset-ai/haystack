# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
import string
from typing import Any

from haystack import component, default_to_dict


@component
class TextCleaner:
    """
    Cleans text strings.

    It can remove substrings matching a list of regular expressions, convert text to lowercase,
    remove punctuation, and remove numbers.
    Use it to clean up text data before evaluation.

    ### Usage example

    ```python
    from haystack.components.preprocessors import TextCleaner

    text_to_clean = "1Moonlight shimmered softly, 300 Wolves howled nearby, Night enveloped everything."

    cleaner = TextCleaner(convert_to_lowercase=True, remove_punctuation=False, remove_numbers=True)
    result = cleaner.run(texts=[text_to_clean])
    ```
    """

    def __init__(
        self,
        remove_regexps: list[str] | None = None,
        convert_to_lowercase: bool = False,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
    ) -> None:
        """
        Initializes the TextCleaner component.

        :param remove_regexps: A list of regex patterns to remove matching substrings from the text.
            The patterns are applied one after another, in list order, and match case-insensitively.
            To match a part of a pattern case-sensitively, use an inline flag group such as `(?-i:[A-Z]+)`.
        :param convert_to_lowercase: If `True`, converts all characters to lowercase.
        :param remove_punctuation: If `True`, removes punctuation from the text.
        :param remove_numbers: If `True`, removes numerical digits from the text.
        """
        self._remove_regexps = remove_regexps
        self._convert_to_lowercase = convert_to_lowercase
        self._remove_punctuation = remove_punctuation
        self._remove_numbers = remove_numbers

        # Compile each pattern on its own: joining them with "|" would renumber their groups (breaking
        # backreferences like `\1`), reject repeated group names, and reject inline flags in later patterns.
        self._regexes = [re.compile(regexp, flags=re.IGNORECASE) for regexp in remove_regexps or []]
        to_remove = ""
        if remove_punctuation:
            to_remove = string.punctuation
        if remove_numbers:
            to_remove += string.digits

        self._translator = str.maketrans("", "", to_remove) if to_remove else None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            remove_regexps=self._remove_regexps,
            convert_to_lowercase=self._convert_to_lowercase,
            remove_punctuation=self._remove_punctuation,
            remove_numbers=self._remove_numbers,
        )

    @component.output_types(texts=list[str])
    def run(self, texts: list[str]) -> dict[str, Any]:
        """
        Cleans up the given list of strings.

        :param texts: List of strings to clean.
        :returns: A dictionary with the following key:
            - `texts`:  the cleaned list of strings.
        :raises TypeError:
            If `texts` is not a list or any element is not a string.
        """
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise TypeError("`texts` must be a list of strings.")

        for regex in self._regexes:
            texts = [regex.sub("", text) for text in texts]

        if self._convert_to_lowercase:
            texts = [text.lower() for text in texts]

        if self._translator:
            texts = [text.translate(self._translator) for text in texts]

        return {"texts": texts}
