from typing import List, Optional, Tuple, Union, Dict, Any

import logging
import regex
from itertools import accumulate
from pathlib import Path
from pickle import UnpicklingError

import nltk
from transformers import PreTrainedTokenizer

from haystack.modeling.model.feature_extraction import FeatureExtractor


logger = logging.getLogger(__name__)


def split_by_regex(pattern: str, text: str) -> List[str]:
    """
    Splits a long text into chunks based on a regex match.

    The regex must have the separator in a match group (so in parenthesis)
    or it will be removed from the match.

    Good example: pattern=r"([0-9]*)"
    Bad example: pattern=r"[0-9]*"

    :param pattern: The regex to split the text on.
    :param text: The text to split.
    :return: The list of splits, along with the length of the separators matched.
    """
    raw_splits = regex.compile(pattern).split(text)
    return [string + separator for string, separator in zip(raw_splits[::2], raw_splits[1::2] + [""])]


def split_by_separators(separators: List[str], text: str) -> List[str]:
    """
    Splits a long text into chunks based on a several separators match.

    :param patterns: The substrings to split the text on.
    :param text: The text to split.
    :return: The list of splits
    """
    split_ids = set()
    for separator in separators:

        # Split the string on the separator and re-add the separator
        splits = [split + separator for split in text.split(separator)]
        splits[-1] = splits[-1][: -len(separator)]

        # Store all the locations where the string was split
        new_splitting_indices = accumulate([len(split) for split in splits])
        split_ids = split_ids.union(new_splitting_indices)

    # Split the string on all collected split points, merging together splits that contain only separators.
    texts = []
    last_split_id = 0
    for split_id in sorted(split_ids):
        split = text[last_split_id:split_id]
        if texts and split in separators + [""]:
            texts[-1] += split
        else:
            texts.append(split)
        last_split_id = split_id

    return texts


def split_by_separator(separator: str, text: str) -> List[str]:
    """
    Splits a long text into chunks based on a separator.

    :param separator: The separator to split the text on. If None, splits by whitespace (see .split() documentation)
    :param text: The text to split.
    :return: The list of splits
    """
    units = []
    raw_units = text.split(separator)

    # Manage separators at the start of the string:
    # Collect them all and attach them to the first unit.
    prefix = ""
    for prefix_position, unit in enumerate(raw_units):
        if not unit:
            prefix += separator
        else:
            units.append(prefix + unit + separator)
            break

    # Re-attach a separator to each unit
    for unit in raw_units[prefix_position + 1 :]:
        if units and not unit:
            units[-1] += separator
        else:
            units.append(unit + separator)

    # Separators at the end of the string:
    # Last one should always be removed, because
    # if it was present in the original string it will be added twice
    units[-1] = units[-1][: -len(separator)]
    return units


def split_by_sentence_tokenizer(text: str, tokenizer) -> List[str]:
    """
    Splits a given text with an NLTK sentence tokenizer, preserving all whitespace.

    :param text: The text to tokenize.
    :return: The tokenized text as a list of strings.
    """
    token_spans = tokenizer.span_tokenize(text)
    return split_on_spans(text=text, spans=token_spans)


def split_by_transformers_tokenizer(text: str, tokenizer: Union[FeatureExtractor, PreTrainedTokenizer]) -> List[str]:
    """
    Splits a given text with a tokenizer, preserving all whitespace.

    :param text: The text to tokenize.
    :return: The tokenized text as a list of strings.
    """
    token_batch = tokenizer(text=text)
    token_positions = [pos for pos, i in enumerate(token_batch.sequence_ids()) if i == 0]
    token_chars = [token_batch.token_to_chars(i) for i in token_positions]
    token_spans = [(chars.start, chars.end) for chars in token_chars]
    return split_on_spans(text=text, spans=token_spans)


def split_on_spans(text: str, spans: List[Tuple[int, int]]) -> List[str]:
    """
    Splits a given text on the arbitrary spans given.

    :param text: The text to tokenize
    :param spans: The spans to split on.
    :return: The tokenized text as a list of strings
    """
    units = []
    prev_token_start = 0
    for token_start, _ in spans:

        if prev_token_start != token_start:
            units.append(text[prev_token_start:token_start])
            prev_token_start = token_start

    if prev_token_start != len(text):
        units.append(text[prev_token_start:])

    if not units:
        return [text]
    return units


def load_sentence_tokenizer(language: Optional[str] = None, tokenizer_model_folder: Optional[Path] = None):
    """
    Attempt to load the sentence tokenizer with sensible fallbacks.

    Tried to load from self.tokenizer_model_folder first, then falls back to 'tokenizers/punkt' and eventually
    falls back to the default English tokenizer.
    """
    # Try loading from the specified path
    if tokenizer_model_folder and language:
        tokenizer_model_path = Path(tokenizer_model_folder) / f"{language}.pickle"
        try:
            return nltk.data.load(f"file:{str(tokenizer_model_path)}", format="pickle")
        except LookupError as e:
            logger.exception(
                "Couldn't load sentence tokenizer from %s "
                "Check that the path is correct and that Haystack has permission to access it.",
                tokenizer_model_path,
            )
        except (UnpicklingError, ValueError) as e:
            logger.exception(
                "Couldn't find custom sentence tokenizer model for %s in %s. ", language, tokenizer_model_folder
            )
        logger.warning("Trying to find a tokenizer for %s under the default path (tokenizers/punkt/).", language)

    # Try loading from the default path
    if language:
        try:
            return nltk.data.load(f"tokenizers/punkt/{language}.pickle")
        except LookupError as e:
            logger.exception(
                "Couldn't load sentence tokenizer from the default tokenizer path (tokenizers/punkt/). "
                'Make sure NLTK is properly installed, or try running `nltk.download("punkt")` from '
                "any Python shell."
            )
        except (UnpicklingError, ValueError) as e:
            logger.exception(
                "Couldn't find custom sentence tokenizer model for %s in the default tokenizer path (tokenizers/punkt/)"
                'Make sure NLTK is properly installed, or try running `nltk.download("punkt")` from '
                "any Python shell.",
                language,
            )
        logger.warning(
            "Using an English tokenizer as fallback. You may train your own model and use the 'tokenizer_model_folder' parameter."
        )

    # Fallback to English from the default path
    return nltk.data.load("tokenizers/punkt/english.pickle")
