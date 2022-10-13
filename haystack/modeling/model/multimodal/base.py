from typing import Any, List, Union, Optional

import logging
from pathlib import Path
from abc import ABC, abstractmethod

import torch


logger = logging.getLogger(__name__)


class HaystackModel(ABC):
    """
    The interface on top of HaystackTransformer and HaystackSentenceTransformer.
    """

    def __init__(
        self, pretrained_model_name_or_path: Union[str, Path], model_type: Optional[str], content_type: str
    ):  # replace the type of content_type with ContentTypes starting Python3.8
        """
        :param pretrained_model_name_or_path: The name of the model to load
        :param model_type: the value of `model_type` from the model's `Config` class.
        :param content_type: The type of data (such as "text", "image" and so on) the model should process.
            See the values of `haystack.schema.ContentTypes`.
        """
        logger.info(
            f" 🤖 Loading '{pretrained_model_name_or_path}' "
            f"({self.__class__.__name__} of type '{model_type if model_type else '<unknown>'}' "
            f"for {content_type} data)"
        )
        self.model_name_or_path = pretrained_model_name_or_path
        self.model_type = model_type
        self.content_type = content_type

    @abstractmethod
    def encode(self, data: List[Any], **kwargs) -> torch.Tensor:
        """
        Run the model on the input data to obtain output vectors.
        """
        raise NotImplementedError("Abstract method, use a subclass.")

    @abstractmethod
    def to(self, devices: Optional[List[torch.device]]) -> None:
        """
        Send the model to the specified PyTorch device(s)
        """
        raise NotImplementedError("Abstract method, use a subclass.")

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """
        The output embedding size.
        """
        raise NotImplementedError("Abstract method, use a subclass.")
