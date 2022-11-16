from typing import Any, Union, Optional, List, Dict

import logging
import numpy as np
from haystack.modeling.visual import SAMPLE


logger = logging.getLogger(__name__)


class Sample:
    """A single training/test sample. This should contain the input and the label. Is initialized with
    the human readable clear_text. Over the course of data preprocessing, this object is populated
    with tokenized and featurized versions of the data."""

    def __init__(
        self,
        id: str,
        clear_text: dict,
        tokenized: Optional[dict] = None,
        features: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        :param id: The unique id of the sample
        :param clear_text: A dictionary containing various human readable fields (e.g. text, label).
        :param tokenized: A dictionary containing the tokenized version of clear text plus helpful meta data: offsets (start position of each token in the original text) and start_of_word (boolean if a token is the first one of a word).
        :param features: A dictionary containing features in a vectorized format needed by the model to process this sample.
        """
        self.id = id
        self.clear_text = clear_text
        self.features = features
        self.tokenized = tokenized

    def __str__(self):

        if self.clear_text:
            clear_text_str = "\n \t".join([k + ": " + str(v) for k, v in self.clear_text.items()])
            if len(clear_text_str) > 3000:
                clear_text_str = (
                    clear_text_str[:3_000] + f"\nTHE REST IS TOO LONG TO DISPLAY. "
                    f"Remaining chars :{len(clear_text_str)-3_000}"
                )
        else:
            clear_text_str = "None"

        if self.features:
            if isinstance(self.features, list):
                features = self.features[0]
            else:
                features = self.features
            feature_str = "\n \t".join([k + ": " + str(v) for k, v in features.items()])
        else:
            feature_str = "None"

        if self.tokenized:
            tokenized_str = "\n \t".join([k + ": " + str(v) for k, v in self.tokenized.items()])
            if len(tokenized_str) > 3000:
                tokenized_str = (
                    tokenized_str[:3_000] + f"\nTHE REST IS TOO LONG TO DISPLAY. "
                    f"Remaining chars: {len(tokenized_str)-3_000}"
                )
        else:
            tokenized_str = "None"
        s = (
            f"\n{SAMPLE}\n"
            f"ID: {self.id}\n"
            f"Clear Text: \n \t{clear_text_str}\n"
            f"Tokenized: \n \t{tokenized_str}\n"
            f"Features: \n \t{feature_str}\n"
            "_____________________________________________________"
        )
        return s


class SampleBasket:
    """An object that contains one source text and the one or more samples that will be processed. This
    is needed for tasks like question answering where the source text can generate multiple input - label
    pairs."""

    def __init__(
        self,
        id_internal: Optional[Union[int, str]],
        raw: dict,
        id_external: Optional[str] = None,
        samples: Optional[List[Sample]] = None,
    ):
        """
        :param id_internal: A unique identifying id. Used for identification within Haystack.
        :param external_id: Used for identification outside of Haystack. E.g. if another framework wants to pass along its own id with the results.
        :param raw: Contains the various data needed to form a sample. It is ideally in human readable form.
        :param samples: An optional list of Samples used to populate the basket at initialization.
        """
        self.id_internal = id_internal
        self.id_external = id_external
        self.raw = raw
        self.samples = samples


def process_answers(answers, doc_offsets, passage_start_c, passage_start_t):
    """TODO Write Comment"""
    answers_clear = []
    answers_tokenized = []
    for answer in answers:
        # This section calculates start and end relative to document
        answer_text = answer["text"]
        answer_len_c = len(answer_text)
        if "offset" in answer:
            answer_start_c = answer["offset"]
        else:
            answer_start_c = answer["answer_start"]
        answer_end_c = answer_start_c + answer_len_c - 1
        answer_start_t = offset_to_token_idx_vecorized(doc_offsets, answer_start_c)
        answer_end_t = offset_to_token_idx_vecorized(doc_offsets, answer_end_c)

        # TODO: Perform check that answer can be recovered from document?
        # This section converts start and end so that they are relative to the passage
        # TODO: Is this actually necessary on character level?
        answer_start_c -= passage_start_c
        answer_end_c -= passage_start_c
        answer_start_t -= passage_start_t
        answer_end_t -= passage_start_t

        curr_answer_clear = {"text": answer_text, "start_c": answer_start_c, "end_c": answer_end_c}
        curr_answer_tokenized = {
            "start_t": answer_start_t,
            "end_t": answer_end_t,
            "answer_type": answer.get("answer_type", "span"),
        }

        answers_clear.append(curr_answer_clear)
        answers_tokenized.append(curr_answer_tokenized)
    return answers_clear, answers_tokenized


def get_passage_offsets(doc_offsets, doc_stride, passage_len_t, doc_text):
    """
    Get spans (start and end offsets) for passages by applying a sliding window function.
    The sliding window moves in steps of doc_stride.
    Returns a list of dictionaries which each describe the start, end and id of a passage
    that is formed when chunking a document using a sliding window approach."""

    passage_spans = []
    passage_id = 0
    doc_len_t = len(doc_offsets)
    while True:
        passage_start_t = passage_id * doc_stride
        passage_end_t = passage_start_t + passage_len_t
        passage_start_c = doc_offsets[passage_start_t]

        # If passage_end_t points to the last token in the passage, define passage_end_c as the length of the document
        if passage_end_t >= doc_len_t - 1:
            passage_end_c = len(doc_text)

        # Get document text up to the first token that is outside the passage. Strip of whitespace.
        # Use the length of this text as the passage_end_c
        else:
            end_ch_idx = doc_offsets[passage_end_t + 1]
            raw_passage_text = doc_text[:end_ch_idx]
            passage_end_c = len(raw_passage_text.strip())

        passage_span = {
            "passage_start_t": passage_start_t,
            "passage_end_t": passage_end_t,
            "passage_start_c": passage_start_c,
            "passage_end_c": passage_end_c,
            "passage_id": passage_id,
        }
        passage_spans.append(passage_span)
        passage_id += 1
        # If the end idx is greater than or equal to the length of the passage
        if passage_end_t >= doc_len_t:
            break
    return passage_spans


def offset_to_token_idx(token_offsets, ch_idx) -> Optional[int]:
    """Returns the idx of the token at the given character idx"""
    n_tokens = len(token_offsets)
    for i in range(n_tokens):
        if (i + 1 == n_tokens) or (token_offsets[i] <= ch_idx < token_offsets[i + 1]):
            return i
    return None


def offset_to_token_idx_vecorized(token_offsets, ch_idx):
    """Returns the idx of the token at the given character idx"""
    # case ch_idx is at end of tokens
    if ch_idx >= np.max(token_offsets):
        # idx must be including
        idx = np.argmax(token_offsets)
    # looking for the first occurence of token_offsets larger than ch_idx and taking one position to the left.
    # This is needed to overcome n special_tokens at start of sequence
    # and failsafe matching (the character start might not always coincide with a token offset, e.g. when starting at whitespace)
    else:
        idx = np.argmax(token_offsets > ch_idx) - 1
    return idx
