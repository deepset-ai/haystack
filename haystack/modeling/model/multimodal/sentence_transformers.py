from typing import Optional, Dict, Any, List, union

import logging
from pathlib import Path

import torch
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
    in multimodal retrieval settings, for example image retrieval from a text query, mixed table/text retrieval, etc.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        model_type: str,
        content_type: ContentTypes,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param pretrained_model_name_or_path: name of the model to load
        :param model_type: the value of model_type from the model's Config
        :param content_type: the type of data (text, image, ...) the model is supposed to process.
            See the values of `haystack.schema.ContentTypes`.
        :param model_kwargs: dictionary of parameters to pass to the model's initialization
            (revision, use_auth_key, etc...)
            Haystack applies some default parameters to some models. They can be overridden by users by specifying the
            desired value in this parameter. See `DEFAULT_MODEL_PARAMS`.
        """
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_type=model_type,
            content_type=content_type,
        )
        try:
            self.model = SentenceTransformer(pretrained_model_name_or_path, **(model_kwargs or {}))
        except Exception as e:
            logger.exception(
                f"Models of type '{model_type}' like {pretrained_model_name_or_path} "
                "are only supported through sentence-transformers. Please make sure this "
                "model is compatible with sentence-transformers or use an alternative, compatible "
                "implementation of this model."
            )

    def encode(self, data: List[Any], **kwargs) -> torch.Tensor:
        """
        Generate the tensors representing the input data.

        Validates the inputs according to what the subclass declared in the `expected_inputs` property.
        Then passes the vectors to the `_forward()` method and returns its output untouched.
        """
        return self.model.encode(data, **kwargs)
