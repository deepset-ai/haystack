from typing import Any, List

import logging
from abc import ABC, abstractmethod

import torch


logger = logging.getLogger(__name__)


class HaystackModel(ABC):
    """
    Interface on top of HaystackTransformer and HaystackSentenceTransformer
    """

    @abstractmethod
    def encode(self, data: List[Any], **kwargs) -> torch.Tensor:
        """
        The output dimension of this language model
        """
        raise NotImplementedError("Abstract method, use a subclass.")
