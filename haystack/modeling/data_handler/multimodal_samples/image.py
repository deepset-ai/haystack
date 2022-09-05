from typing import Any, Optional, List, Dict

import logging
import numpy as np
from transformers import AutoTokenizer

from haystack.modeling.data_handler.multimodal_samples.base import Sample
from haystack.modeling.model.feature_extraction import FeatureExtractor


logger = logging.getLogger(__name__)


DEFAULT_EXTRACTION_PARAMS = {"return_tensors": "pt"}


class ImageSample(Sample):
    def __init__(
        self, id: str, data: Any, feature_extractor: AutoTokenizer, extraction_params: Optional[Dict[str, Any]] = None
    ):
        """
        A single training/test sample. This should contain the input and the label. It should contain also the original
        human readable data source (the sentence).
        Over the course of data preprocessing, this object will be populated with processed versions of the data.

        :param id: The unique id of the sample
        :param data: A dictionary containing various human readable fields (e.g. text, label).
        :param feature_extractor: the tokenizer to use to tokenize this text.
        :param extraction_params: the parameters to provide to the tokenizer's `__call__()` method. See DEFAULT_EXTRACTION_PARAMS
            for the default values of this field.
            The incoming dictionary will be merged with priority to user-defined values, so if `extraction_params={'max_length': 128}`,
            `__call__()` will receive also `truncation=True` and all the default parameters along with `max_lenght=128`.
        """
        super().__init__(id=id, data=data)

        self.features = self.get_features(
            data=[self.data], feature_extractor=feature_extractor, extraction_params=extraction_params
        )

    @staticmethod
    def get_features(
        data: List[Any], feature_extractor: FeatureExtractor, extraction_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract the features from a list of image objects using the given Feature Extractor.
        The resulting features are contained in a dictionary like {'input_ids': <tensor>, 'token_type_ids': <tensor>, etc...}

        :param data: The image to extract features from.
        :param feature_extractor: the feature extractor to use to extract the features
        :param extraction_params: the parameters to provide to the feature extractor's `__call__()` method. Defaults to the
            following values: {
                "return_tensors": "pt",
            }
            The incoming dictionary will be merged with priority to user-defined values, so if
            `extraction_params={'segmentation_maps': <PIL object>}`, `__call__()` will receive also `return_tensors="pt"`
            and all the default parameters along with `segmentation_maps=< PIL object>`.
        """
        params = DEFAULT_EXTRACTION_PARAMS | (extraction_params or {})
        return feature_extractor(images=data, **params)

    def __str__(self):
        return (
            "-----------------------------------------------------"
            " 🌻 Image Sample:"
            f"ID: {self.id}\n"
            # TODO find better data to show here
            # f"Data: \n \t{data_str}\n"
            # f"Features: \n \t{feature_str}\n"
            "-----------------------------------------------------"
        )
