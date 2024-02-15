import re
import string
from typing import List, Optional


def _preprocess_text(
    texts: List[str],
    regexes_to_ignore: Optional[List[str]] = None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_numbers: bool = False,
) -> List[str]:
    """
    Preprocess the outputs of the runnable to remove unwanted characters.

    :param regexes_to_ignore (list, optional): A list of regular expressions. If provided, it removes substrings
        matching these regular expressions from the text. Defaults to None.
    :param ignore_case (bool, optional): If True, converts all characters to lowercase. Defaults to False.
    :param ignore_punctuation (bool, optional): If True, removes punctuation from the text. Defaults to False.
    :param ignore_numbers (bool, optional): If True, removes numerical digits from the text. Defaults to False.
    :return: A list of preprocessed strings.
    """
    if regexes_to_ignore:
        combined_regex = "|".join(regexes_to_ignore)
        texts = [re.sub(combined_regex, "", text, flags=re.IGNORECASE) for text in texts]

    if ignore_case:
        texts = [text.lower() for text in texts]

    if ignore_punctuation:
        translator = str.maketrans("", "", string.punctuation)
        texts = [text.translate(translator) for text in texts]

    if ignore_numbers:
        translator = str.maketrans("", "", string.digits)
        texts = [text.translate(translator) for text in texts]

    return texts
