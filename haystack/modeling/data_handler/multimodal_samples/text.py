from typing import Any, Optional, List, Dict

import logging

import torch
import numpy as np
from transformers import AutoTokenizer

from haystack.modeling.data_handler.multimodal_samples.base import Sample, SampleBasket
from haystack.modeling.model.feature_extraction import FeatureExtractor
from haystack.errors import ModelingError


logger = logging.getLogger(__name__)


DEFAULT_EXTRACTION_PARAMS = {
    "max_length": 256,
    "add_special_tokens": True,
    "truncation": True,
    "truncation_strategy": "longest_first",
    "padding": "max_length",
    "return_token_type_ids": True,
}


class TextSample(Sample):
    def __init__(
        self,
        id: str,
        data: Dict[str, Any],
        feature_extractor: FeatureExtractor,
        extraction_params: Optional[Dict[str, Any]] = None,
    ):
        """
        A single training/test sample. This should contain the input and the label. It should contain also the original
        human readable data source (the sentence).
        Over the course of data preprocessing, this object will be populated with processed versions of the data.

        :param id: The unique id of the sample
        :param data: A dictionary containing various human readable fields (e.g. text, label).
        :param feature_extractor: the tokenizer to use to tokenize this text.
        :param extraction_params: the parameters to provide to the tokenizer's `encode_plus()` method. See DEFAULT_EXTRACTION_PARAMS
            for the default values of this field.
            The incoming dictionary will be merged with priority to user-defined values, so if `extraction_params={'max_length': 128}`,
            `encode_plus()` will receive also `truncation=True` and all the default parameters along with `max_lenght=128`.
        """
        super().__init__(id=id, data=data)
        self.tokenized: Dict[str, Any] = {}
        self.features: Dict[str, Any] = {}

        # extract features
        self.features = self.get_features(
            data=[self.data["text"]], feature_extractor=feature_extractor, extraction_params=extraction_params
        )

        # tokenize text
        self.tokenized = feature_extractor.convert_ids_to_tokens(self.features["input_ids"])
        if not self.tokenized:
            raise ModelingError(
                f"The text could not be tokenized, likely because it contains characters that the tokenizer does not recognize."
            )

    @staticmethod
    def get_features(
        data: List[Any], feature_extractor: FeatureExtractor, extraction_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract the features from a text snippet using the given Tokenizer.
        The resulting features are contained in a dictionary like {'input_ids': <tensor>, 'token_type_ids': <tensor>, etc...}

        :param data: The text to extract features from.
        :param feature_extractor: the tokenizer to use to tokenize this text.
        :param extraction_params: the parameters to provide to the tokenizer's `encode_plus()` method. Defaults to the
            following values: {
                "max_length": 64,
                "add_special_tokens": True,
                "truncation": True,
                "truncation_strategy": "longest_first",
                "padding": "max_length",
                "return_token_type_ids": True
            }
            The incoming dictionary will be merged with priority to user-defined values, so if `extraction_params={'max_length': 128}`,
            `encode_plus()` will receive also `truncation=True` and all the default parameters along with `max_lenght=128`.
        """
        params = DEFAULT_EXTRACTION_PARAMS | (extraction_params or {})
        features = feature_extractor(text=data, **params)
        return _safe_tensor_conversion(features)

    def __str__(self):
        data_str = "\n \t".join([f"{k}: {v[:200]+'...' if len(v)> 200 else v}" for k, v in self.data.items()])
        feature_str = "\n \t".join([f"{k}: {v}" for k, v in self.features.items()])
        tokenized_str = "\n \t".join([f"{k}: {v[:200]+'...' if len(v)> 200 else v}" for k, v in self.tokenized.items()])

        return (
            "-----------------------------------------------------"
            " 🌻 Text Sample:"
            f"ID: {self.id}\n"
            f"Data: \n \t{data_str}\n"
            f"Tokenized: \n \t{tokenized_str}\n"
            f"Features: \n \t{feature_str}\n"
            "-----------------------------------------------------"
        )


def _safe_tensor_conversion(features: Dict[str, Any]):
    """
    Converts all features into tensors if all input values are 2D integer vectors.
    """
    for tensor_name, tensors in features.items():
        sample = tensors[0][0]

        # Check that the cast to long is safe
        if not np.issubdtype(type(sample), np.integer):
            raise ModelingError(
                f"Feature '{tensor_name}' (sample value: {sample}, type: {type(sample)})"
                " can't be converted safely into a torch.long type."
            )
        # Cast all data to long
        tensors = torch.as_tensor(np.array(tensors), dtype=torch.long)
        features[tensor_name] = tensors
    return features


class TextSampleBasket(SampleBasket):
    def __init__(self, id: str, raw: Dict[str, Any], samples: Optional[List[Sample]] = None):
        """
        An object that contains one source text and the one or more samples that will be processed. This
        is needed for tasks like question answering where the source text can generate multiple
        input - label pairs.

        :param id: A unique identifying id.
        :param raw: Contains the various data needed to form a sample. It is ideally in human readable form.
        :param samples: An optional list of Samples used to populate the basket at initialization.
        """
        self.id = id
        self.raw = raw
        self.samples = samples


# def process_answers(answers, doc_offsets, passage_start_c, passage_start_t):
#     """TODO Write Comment"""
#     answers_clear = []
#     answers_tokenized = []
#     for answer in answers:
#         # This section calculates start and end relative to document
#         answer_text = answer["text"]
#         answer_len_c = len(answer_text)
#         if "offset" in answer:
#             answer_start_c = answer["offset"]
#         else:
#             answer_start_c = answer["answer_start"]
#         answer_end_c = answer_start_c + answer_len_c - 1
#         answer_start_t = offset_to_token_idx_vecorized(doc_offsets, answer_start_c)
#         answer_end_t = offset_to_token_idx_vecorized(doc_offsets, answer_end_c)

#         # TODO: Perform check that answer can be recovered from document?
#         # This section converts start and end so that they are relative to the passage
#         # TODO: Is this actually necessary on character level?
#         answer_start_c -= passage_start_c
#         answer_end_c -= passage_start_c
#         answer_start_t -= passage_start_t
#         answer_end_t -= passage_start_t

#         curr_answer_clear = {"text": answer_text, "start_c": answer_start_c, "end_c": answer_end_c}
#         curr_answer_tokenized = {
#             "start_t": answer_start_t,
#             "end_t": answer_end_t,
#             "answer_type": answer.get("answer_type", "span"),
#         }

#         answers_clear.append(curr_answer_clear)
#         answers_tokenized.append(curr_answer_tokenized)
#     return answers_clear, answers_tokenized


# def get_passage_offsets(doc_offsets, doc_stride, passage_len_t, doc_text):
#     """
#     Get spans (start and end offsets) for passages by applying a sliding window function.
#     The sliding window moves in steps of doc_stride.
#     Returns a list of dictionaries which each describe the start, end and id of a passage
#     that is formed when chunking a document using a sliding window approach."""

#     passage_spans = []
#     passage_id = 0
#     doc_len_t = len(doc_offsets)
#     while True:
#         passage_start_t = passage_id * doc_stride
#         passage_end_t = passage_start_t + passage_len_t
#         passage_start_c = doc_offsets[passage_start_t]

#         # If passage_end_t points to the last token in the passage, define passage_end_c as the length of the document
#         if passage_end_t >= doc_len_t - 1:
#             passage_end_c = len(doc_text)

#         # Get document text up to the first token that is outside the passage. Strip of whitespace.
#         # Use the length of this text as the passage_end_c
#         else:
#             end_ch_idx = doc_offsets[passage_end_t + 1]
#             raw_passage_text = doc_text[:end_ch_idx]
#             passage_end_c = len(raw_passage_text.strip())

#         passage_span = {
#             "passage_start_t": passage_start_t,
#             "passage_end_t": passage_end_t,
#             "passage_start_c": passage_start_c,
#             "passage_end_c": passage_end_c,
#             "passage_id": passage_id,
#         }
#         passage_spans.append(passage_span)
#         passage_id += 1
#         # If the end idx is greater than or equal to the length of the passage
#         if passage_end_t >= doc_len_t:
#             break
#     return passage_spans


# def offset_to_token_idx(token_offsets, ch_idx) -> Optional[int]:
#     """Returns the idx of the token at the given character idx"""
#     n_tokens = len(token_offsets)
#     for i in range(n_tokens):
#         if (i + 1 == n_tokens) or (token_offsets[i] <= ch_idx < token_offsets[i + 1]):
#             return i
#     return None


# def offset_to_token_idx_vecorized(token_offsets, ch_idx):
#     """Returns the idx of the token at the given character idx"""
#     # case ch_idx is at end of tokens
#     if ch_idx >= np.max(token_offsets):
#         # idx must be including
#         idx = np.argmax(token_offsets)
#     # looking for the first occurence of token_offsets larger than ch_idx and taking one position to the left.
#     # This is needed to overcome n special_tokens at start of sequence
#     # and failsafe matching (the character start might not always coincide with a token offset, e.g. when starting at whitespace)
#     else:
#         idx = np.argmax(token_offsets > ch_idx) - 1
#     return idx
