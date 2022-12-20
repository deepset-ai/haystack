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
            groups.append(group)
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
    contents: List[str], max_chars: int, max_tokens: int = 0, tokens: Optional[List[str]] = None
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
    if not contents:
        if tokens:
            raise ValueError(
                "An error occurred during tokenization, and tokens were generated even if there are no documents. "
                "This is a bug with tokenization. If you provided your own tokenizer function, double check "
                "that it is not losing or adding chars during the splitting. "
                "The function should pass this assert for every possible input text:\n"
                "   `assert len(input) == len(''.join(output))`\n"
                "If you used a tokenized provided by Haystack, please report the bug to the maintainers. "
            )
        return []

    valid_contents = []

    if max_tokens:
        if not tokens:
            raise ValueError("if max_tokens is set, you must pass the tokenized text to `tokens`.")

        # Group the tokens by document
        tokens_groups = []
        tokens_length = 0
        tokens_count = 0
        tokens_group = []
        content_id = 0
        for token in tokens:
            if tokens_length < len(contents[content_id]):
                tokens_group.append(token)
                tokens_count += 1
                tokens_length += len(token)

            if tokens_length == len(contents[content_id]):
                tokens_groups.append((contents[content_id], tokens_group, tokens_length, tokens_count))
                tokens_length = 0
                tokens_count = 0
                content_id += 1

            elif tokens_length > len(contents[content_id]):
                raise ValueError(
                    "It seems like the tokens you provided don't match with your documents content. "
                    "This is likely to be a bug with tokenization. "
                    "If you provided your own tokenizer function, double check "
                    "that it is not losing or adding chars during the splitting of tokens. "
                    "The following function should pass this assert for every possible input text:\n"
                    "   assert len(input) == len(''.join(output))\n"
                    "If you used a tokenizer provided by Haystack, please report the bug to the maintainers."
                )

        # Validate the groups on max_chars
        chars_validated_contents = []
        for content, tokens, tokens_length, tokens_count in tokens_groups:
            if max_chars and tokens_length <= max_chars:
                chars_validated_contents.append((content, tokens, tokens_length, tokens_count))
            else:
                while tokens_length > max_chars:
                    logger.warning(
                        "Found unit of text with a character count higher than the maximum allowed. "
                        "The unit is going to be cut at %s chars, so %s chars are being moved to one (or more) new documents. "
                        "Set the maximum amount of characters allowed through the 'max_chars' parameter. "
                        "Keep in mind that very long Documents can severely impact the performance of Readers.\n"
                        "Enable debug level logs to see the content of the unit that is being split",
                        max_chars,
                        len(content) - max_chars,
                    )
                    logger.debug("Original document content:\n%s\n", content)

                    # Hard-split and re-count the tokens
                    while len(content) > max_chars:
                        split_tokens_length = 0
                        for token_count, token in enumerate(tokens):
                            split_tokens_length += len(token)
                            if split_tokens_length >= max_chars:
                                break

                    tokens_length -= split_tokens_length
                    broken_token_split_position = max_chars - split_tokens_length
                    broken_token_head = tokens[token_count][:broken_token_split_position]

                    valid_contents.append(
                        (content[:max_chars], tokens[:token_count] + [broken_token_head], max_chars, token_count)
                    )
                    content = content[max_chars:]

                    # If we're splitting over a token: count it in the new document too.
                    broken_token_tail = tokens[token_count][broken_token_split_position:]
                    tokens = [broken_token_tail] + tokens[token_count:]

                valid_contents.append((content, tokens, sum(len(token) for token in tokens), token_count))

        # Validate the groups on max_tokens
        valid_contents = []
        for content, tokens, tokens_length, tokens_count in chars_validated_contents:
            if max_tokens and tokens_count <= max_tokens:
                valid_contents.append((content, tokens_count))
            else:
                while tokens_count > max_tokens:
                    logger.info(
                        "Found unit of text with a token count higher than the maximum allowed. "
                        "The unit is going to be cut at %s tokens, and the remaining %s chars will go to one (or more) new documents. "
                        "Set the maximum amount of tokens allowed through the 'max_tokens' parameter."
                        "Keep in mind that very long Documents can severely impact the performance of Readers.\n"
                        "Enable debug level logs to see the content of the unit that is being split",
                        max_tokens,
                        len(content) - tokens_length,
                    )
                    logger.debug("Original document content:\n%s\n", content)

                    max_tokens_length = sum(len(token) for token in tokens[:max_tokens])
                    valid_contents.append((content[:max_tokens_length], max_tokens))
                    content = content[max_tokens_length:]
                    tokens = tokens[max_tokens:]
                    tokens_count -= max_tokens
                    tokens_length -= max_tokens_length

                max_tokens_length = sum(len(token) for token in tokens)
                valid_contents.append((content, len(tokens)))

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
    if not sources:
        return []

    if len(sources) == 1:
        return sources[0][1]

    aligned_headlines = []
    position_in_merged_document = 0
    for content, headlines in sources:
        for headline in headlines:
            headline["start_idx"] += position_in_merged_document
            aligned_headlines.append(headline)
        position_in_merged_document += len(content) + len(separator)
    return aligned_headlines


def common_values(list_of_dicts: List[Dict[str, Any]], exclude: Optional[List[str]] = []) -> Dict[str, Any]:
    """
    Retains all keys shared across all the documents being merged.

    Such keys are checked recursively, see tests.

    :param list_of_dicts: dicts to merge
    :param exclude: keys to drop regardless of their content
    :return: the merged dictionary
    """
    if not list_of_dicts:
        return {}

    if exclude is None:
        exclude = []

    shortest_dict = min(list_of_dicts, key=len)
    merge_dictionary = {}
    for key, value in shortest_dict.items():

        # if not all dicts have this key, skip
        if not key in exclude and all(key in dict.keys() for dict in list_of_dicts):

            # if the value is a dictionary, merge recursively
            if isinstance(value, dict):
                list_of_subdicts = [dictionary[key] for dictionary in list_of_dicts]
                merge_dictionary[key] = common_values(list_of_subdicts)

            # If the value is not a dictionary, keep only if the values is the same for all
            elif all(value == dict[key] for dict in list_of_dicts):
                merge_dictionary[key] = value

    return merge_dictionary
