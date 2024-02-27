import re
import string
from typing import Any, Dict, List, Optional

from haystack import component


@component
class TextCleaner:
    """
    A preprocessor component to clean text data. It can remove substrings matching a list of regular expressions,
    convert text to lowercase, remove punctuation, and remove numbers.

    This is useful to cleanup text data before evaluation.
    """

    def __init__(
        self,
        remove_regexps: Optional[List[str]] = None,
        convert_to_lowercase: bool = False,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
    ):
        """
        :param remove_regexps: A list of regular expressions. If provided, it removes substrings
            matching these regular expressions from the text.
        :param convert_to_lowercase: If True, converts all characters to lowercase.
        :param remove_punctuation: If True, removes punctuation from the text.
        :param remove_numbers: If True, removes numerical digits from the text.
        """
        self._remove_regexps = remove_regexps
        self._convert_to_lowercase = convert_to_lowercase
        self._remove_punctuation = remove_punctuation
        self._remove_numbers = remove_numbers

        self._regex = None
        if remove_regexps:
            self._regex = re.compile("|".join(remove_regexps), flags=re.IGNORECASE)
        to_remove = ""
        if remove_punctuation:
            to_remove = string.punctuation
        if remove_numbers:
            to_remove += string.digits

        self._translator = str.maketrans("", "", to_remove) if to_remove else None

    @component.output_types(texts=List[str])
    def run(self, texts: List[str]) -> Dict[str, Any]:
        """
        Cleans up the given list of strings.

        :param texts: List of strings to clean.
        :returns: A dictionary with the following key:
            - `texts`:  the cleaned list of strings.
        """

        if self._regex:
            texts = [self._regex.sub("", text) for text in texts]

        if self._convert_to_lowercase:
            texts = [text.lower() for text in texts]

        if self._translator:
            texts = [text.translate(self._translator) for text in texts]

        return {"texts": texts}
