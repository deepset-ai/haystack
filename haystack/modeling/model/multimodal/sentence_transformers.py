from typing import Optional, Dict, Any, List, Union

import logging
from pathlib import Path

import torch
from torch import nn
from sentence_transformers import SentenceTransformer

from haystack.modeling.model.multimodal.base import HaystackModel
from haystack.schema import ContentTypes


logger = logging.getLogger(__name__)


class HaystackSentenceTransformerModel(HaystackModel):
    """
    Parent class for `sentence-transformers` models.

    These models read raw data (text, image), internally extract features, and return vectors that capture the meaning
    of the original data.

    Models inheriting from `HaystackSentenceTransformerModel` are designed to be used in parallel one with the other
    in multimodal retrieval settings, for example image retrieval from a text query, mixed table/text retrieval, and so on.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        model_type: str,
        content_type: ContentTypes,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param pretrained_model_name_or_path: The name of the model to load.
        :param model_type: the value of `model_type` from the model's `Config` class
        :param content_type: The type of data (such as "text", "image", and so on) the model should process.
            See the values of `haystack.schema.ContentTypes`.
        :param model_kwargs: A dictionary of parameters to pass to the model's initialization.
            (revision, use_auth_key, and so on)
            Haystack applies some default parameters to some models. You can override them by specifying the
            desired value in this parameter. See `DEFAULT_MODEL_PARAMS`.
        """
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_type=model_type,
            content_type=content_type,
        )
        try:
            self.model = SentenceTransformer(pretrained_model_name_or_path, **(model_kwargs or {}))
        except Exception:
            logger.exception(
                "Models of type '%s' like %s "
                "are only supported through sentence-transformers. Make sure this "
                "model is compatible with sentence-transformers or use an alternative, compatible "
                "implementation of this model.",
                model_type,
                pretrained_model_name_or_path,
            )

    @property
    def embedding_dim(self) -> int:
        """
        Finds out the output embedding dim by running the model on a minimal amount of mock data.
        """
        emedding_dim = self.model.get_sentence_embedding_dimension()
        if not emedding_dim:
            logger.warning(
                "Can't find the output embedding dimensions for '%s'. Some checks will not run as intended.",
                self.model_name_or_path,
            )
        return emedding_dim

    def to(self, devices: Optional[List[torch.device]]) -> None:
        """
        Send the model to the specified PyTorch devices.
        """
        if devices:
            if len(devices) > 1:
                self.model = nn.DataParallel(self.model, device_ids=devices)
            else:
                self.model.to(devices[0])

    def encode(self, data: List[Any], **kwargs) -> torch.Tensor:
        """
        Generate the tensors representing the input data.

        Validates the inputs according to what the subclass declared in the `expected_inputs` property.
        Then passes the vectors to the `_forward()` method and returns its output untouched.
        """
        return self.model.encode(data, convert_to_tensor=True, **kwargs)
