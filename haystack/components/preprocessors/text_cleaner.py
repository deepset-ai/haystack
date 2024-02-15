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
        Creates a new instance of TextCleaner.

        :param remove_regexps: A list of regular expressions. If provided, it removes substrings
            matching these regular expressions from the text. Defaults to None.
        :type remove_regexps: Optional[List[str]], optional
        :param convert_to_lowercase: If True, converts all characters to lowercase. Defaults to False.
        :type convert_to_lowercase: bool, optional
        :param remove_punctuation: If True, removes punctuation from the text. Defaults to False.
        :type remove_punctuation: bool, optional
        :param remove_numbers: If True, removes numerical digits from the text. Defaults to False.
        :type remove_numbers: bool, optional
        """
        self._remove_regexps = remove_regexps
        self._convert_to_lowercase = convert_to_lowercase
        self._remove_punctuation = remove_punctuation
        self._remove_numbers = remove_numbers

    @component.output_types(texts=List[str])
    def run(self, texts: List[str]) -> Dict[str, Any]:
        """
        Run the TextCleaner on the given list of strings.
        """
        if self._remove_regexps:
            combined_regex = "|".join(self._remove_regexps)
            texts = [re.sub(combined_regex, "", text, flags=re.IGNORECASE) for text in texts]

        if self._convert_to_lowercase:
            texts = [text.lower() for text in texts]

        if self._remove_punctuation:
            translator = str.maketrans("", "", string.punctuation)
            texts = [text.translate(translator) for text in texts]

        if self._remove_numbers:
            translator = str.maketrans("", "", string.digits)
            texts = [text.translate(translator) for text in texts]

        return {"texts": texts}
