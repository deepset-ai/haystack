from typing import List, Optional, Tuple, Union, Dict, Any

import logging
import regex
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
    texts = [text]
    for separator in separators:
        split_text = []
        for text in texts:
            splits = split_by_separator(separator=separator, text=text)
            split_text += splits
        texts = split_text
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
    if raw_units[0] == "":
        units = [separator]
    for unit in raw_units:
        if unit:
            units.append(unit)
        else:
            units[-1] += separator
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


def make_merge_groups(
    contents: List[Tuple[str, int]], window_size: int, window_overlap: int, max_tokens: int, max_chars: int
):
    """
    Creates the groups of documents that need to be merged, respecting all boundaries.

    :param contents: a tuple with the document content and how many tokens it contains.
        Can be created with `validate_boundaries()`
    :param window_size: The number of documents to include in each merged batch. For example,
        if set to 2, the documents are merged in pairs. When set to 0, merges all documents
        into one single document.
    :param window_overlap: Applies a sliding window approach over the documents groups.
        For example, if `window_size=3` and `window_overlap=2`, the resulting documents come
        from the merge of the following groups: `[doc1, doc2, doc3]`, `[doc2, doc3, doc4]`, ...
    :param max_tokens: the maximum number of tokens allowed
    :param max_chars: the maximum number of chars allowed
    :returns: a list of lists, each sublist containing the indices of the documents that should be merged together.
        The length of this list is equal to the number of documents that will be produced by the merger.
    """
    groups = []
    if max_tokens:
        group = []
        content_len = 0
        tokens_count = 0
        doc_index = 0
        while doc_index < len(contents):

            if max_chars and content_len + len(contents[doc_index][0]) > max_chars:
                # Rare and odd case: log loud
                logger.warning(
                    "One document reached `max_chars` (%s) before either `max_tokens` (%s) or `window_size` (%s). "
                    "The last unit is moved to the next document to keep the chars count below the threshold."
                    "Consider raising `max_chars` and double-check how the input coming from earlier nodes looks like.\n"
                    "Enable DEBUG logs to see the document.",
                    max_chars,
                    max_tokens,
                    window_size,
                )
                logger.debug(
                    "********* Original document content: *********\n%s\n****************************",
                    contents[doc_index][0],
                )
                groups.append(group)
                group = []
                tokens_count = 0
                content_len = 0
                if window_overlap:
                    doc_index -= window_overlap

            if max_tokens and tokens_count + contents[doc_index][1] > max_tokens:
                # More plausible case: log, but not too loud
                logger.info(
                    "One document reached `max_tokens` (%s) before `window_size` (%s). "
                    "The last unit is moved to the next document to keep the token count below the threshold.\n"
                    "Enable DEBUG logs to see the document.",
                    max_tokens,
                    window_size,
                )
                logger.debug(
                    "********* Original document content: *********\n%s\n****************************",
                    contents[doc_index][0],
                )
                groups.append(group)
                group = []
                tokens_count = 0
                content_len = 0
                if window_overlap:
                    doc_index -= window_overlap

            if window_size and len(group) >= window_size:
                # Fully normal: debug log only
                logger.debug(
                    "One document reached `window_size` (%s) before `max_tokens` (%s). ", max_tokens, window_size
                )
                groups.append(group)
                group = []
                tokens_count = 0
                content_len = 0
                if window_overlap:
                    doc_index -= window_overlap

            # Still accumulating
            group.append(doc_index)
            tokens_count += contents[doc_index][1]
            content_len += len(contents[doc_index][0])
            doc_index += 1

        # Last group after the loop
        if group:
            group.append(doc_index)
        return groups

    # Shortcuts for when max_tokens is not used
    elif window_size:
        return [
            list(range(pos, min(pos + window_size, len(contents))))
            for pos in range(0, max(1, len(contents) - window_overlap), window_size - window_overlap)
        ]
    else:
        return [list(range(len(contents)))]


def validate_unit_boundaries(
    contents: List[str], max_chars: int, max_tokens: int, tokens: Optional[List[str]] = None
) -> List[Tuple[str, int]]:
    """
    Makes sure all boundaries (max_tokens if given, max_char if given) are respected. Splits the strings if necessary.

    :param contents: the content of all documents to merge, as a string
    :param tokens: a single list with all the tokens contained in all documents to merge.
        So, if `contents = ["hello how are you", "I'm fine thanks"]`,
        tokens should contain something similar to `["hello", "how", "are", "you", "I'm", "fine", "thanks"]`
    :param max_tokens: the maximum amount of tokens allowed in a doc
    :param max_chars: the maximum number of chars allowed in a doc
    :return: a tuple (content, n_of tokens) if max_token is set, else (content, 0)
    """
    valid_contents = []

    # Count tokens and chars, split if necessary
    if max_tokens:
        if not tokens:
            raise ValueError("if max_tokens is set, you must pass the tokenized text to `tokens`.")

        for content in contents:
            tokens_length = 0
            for tokens_count, token in enumerate(tokens):
                tokens_length += len(token)

                # If we reached the doc length, record how many tokens it contained and pass on the next doc
                if tokens_length >= len(content):
                    valid_contents.append((content, tokens_count))
                    break

                # This doc has more than max_tokens: save the head as a separate document and continue
                if tokens_count >= max_tokens:
                    logger.info(
                        "Found unit of text with a token count higher than the maximum allowed. "
                        "The unit is going to be cut at %s tokens, and the remaining %s chars will go to one (or more) new documents. "
                        "Set the maximum amout of tokens allowed through the 'max_tokens' parameter."
                        "Keep in mind that very long Documents can severely impact the performance of Readers.\n"
                        "Enable DEBUG level logs to see the content of the unit that is being split",
                        max_tokens,
                        len(content) - tokens_length,
                    )
                    logger.debug(
                        "********* Original document content: *********\n%s\n****************************", content
                    )
                    valid_contents.append((content[:tokens_length], tokens_count))
                    content = content[:tokens_length]
                    tokens_count = 0

                # This doc has more than max_chars: save the head as a separate document and continue
                if max_chars and tokens_length >= max_chars:
                    logger.warning(
                        "Found unit of text with a character count higher than the maximum allowed. "
                        "The unit is going to be cut at %s chars, so %s chars are being moved to one (or more) new documents. "
                        "Set the maximum amout of characters allowed through the 'max_chars' parameter. "
                        "Keep in mind that very long Documents can severely impact the performance of Readers.\n"
                        "Enable DEBUG level logs to see the content of the unit that is being split",
                        max_chars,
                        len(content) - max_chars,
                    )
                    logger.debug(
                        "********* Original document content: *********\n%s\n****************************", content
                    )
                    valid_contents.append((content[:max_chars], tokens_count))
                    content = content[:max_chars]
                    tokens_count = 0

    # Validate only the chars, split if necessary
    else:
        for content in contents:
            if max_chars and len(content) >= max_chars:
                logger.error(
                    "Found unit of text with a character count higher than the maximum allowed. "
                    "The unit is going to be cut at %s chars, so %s chars are being moved to one (or more) new documents. "
                    "Set the maximum amout of characters allowed through the 'max_chars' parameter. "
                    "Keep in mind that very long Documents can severely impact the performance of Readers.",
                    max_chars,
                    len(content) - max_chars,
                )
                valid_contents += [
                    (content[max_chars * i : max_chars * (i + 1)], 0) for i in range(int(len(content) / max_chars) + 1)
                ]
            else:
                valid_contents.append((content, 0))

    return valid_contents


def merge_headlines(
    sources: List[Tuple[str, List[Dict[str, Any]]]], separator: str
) -> List[Dict[str, Union[str, int]]]:
    """
    Merges the headlines dictionary with the new position of each headline into the merged document.
    Assumes the documents are in the same order as when they were merged.

    :param sources: tuple (source document content, source document headlines).
    :param separator: the string used to join the document's content
    :return: a dictionary that can be assigned to the merged document's headlines key.
    """
    aligned_headlines = []
    position_in_merged_document = 0
    for content, headlines in sources:
        for headline in headlines:
            headline["start_idx"] += position_in_merged_document
            aligned_headlines.append(headline)
        position_in_merged_document += len(content) + len(separator)
    return aligned_headlines


def common_values(list_of_dicts: List[Dict[str, Any]], exclude: List[str]) -> Dict[str, Any]:
    """
    Retains all keys shared across all the documents being merged.

    Such keys are checked recursively, see tests.

    :param list_of_dicts: dicts to merge
    :param exclude: keys to drop regardless of their content
    :return: the merged dictionary
    """
    shortest_dict = min(list_of_dicts, key=len)
    merge_dictionary = {}
    for key, value in shortest_dict.items():

        # if not all dicts have this key, skip
        if not key in exclude and all(key in dict.keys() for dict in list_of_dicts):

            # if the value is a dictionary, merge recursively
            if isinstance(value, dict):
                list_of_subdicts = [dictionary[key] for dictionary in list_of_dicts]
                merge_dictionary[key] = common_values(list_of_subdicts, exclude=[])

            # If the value is not a dictionary, keep only if the values is the same for all
            elif all(value == dict[key] for dict in list_of_dicts):
                merge_dictionary[key] = value

    return merge_dictionary
