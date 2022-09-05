from typing import Any, Optional, List, Dict

import logging

import torch
import numpy as np

from haystack.modeling.data_handler.multimodal_samples.base import Sample
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
