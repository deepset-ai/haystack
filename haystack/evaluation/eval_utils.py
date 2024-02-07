import re
import string
from typing import Any, Dict, List, Optional


def preprocess_text(
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


def get_answers_from_output(outputs: List[Dict[str, Any]], output_key: str, runnable_type: str) -> List[str]:
    """
    Extracts the answers from the output of a pipeline or component.

    :param outputs: The outputs of the runnable.
    :return: List of answers from the runnable output.
    """
    answers = []
    if runnable_type == "pipeline":
        # Iterate over output from each Pipeline run
        for output in outputs:
            # Iterate over output of component in each Pipeline run
            for component_output in output.values():
                # Only extract answers based on key
                for key in component_output.keys():
                    if output_key in key:
                        for generated_answer in component_output[output_key]:
                            if generated_answer.data:
                                answers.append(generated_answer.data)
    else:
        # Iterate over output from each Component run
        for output in outputs:
            # Only extract answers based on key
            for key in output.keys():
                if output_key in key:
                    for generated_answer in output[output_key]:
                        if generated_answer.data:
                            answers.append(generated_answer.data)
    return answers
