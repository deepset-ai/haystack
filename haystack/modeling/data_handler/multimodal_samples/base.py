from typing import Any, Union, Optional, List, Dict

import logging
from abc import ABC, abstractmethod

import numpy as np

from haystack.modeling.model.feature_extraction import FeatureExtractor


logger = logging.getLogger(__name__)


class Sample(ABC):
    def __init__(self, id: str, data: Dict[str, Any]):
        """
        A single training/test sample. This should contain the input and the label. It should contain also the original
        human readable data source (like the sentence, or the path to the original image).
        Over the course of data preprocessing, this object will be populated with processed versions of the data.

        :param id: The id of the sample
        :param data: A dictionary containing various human readable fields (e.g. text, label).
        """
        self.id = id
        self.data = data

    @classmethod
    @abstractmethod
    def get_features(
        data: List[Any], feature_extractor: FeatureExtractor, extraction_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        pass
