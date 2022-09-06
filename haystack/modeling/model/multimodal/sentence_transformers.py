from typing import Optional, Dict, Any, List

import logging

import torch
from sentence_transformers import SentenceTransformer

from haystack.modeling.model.multimodal.base import HaystackModel
from haystack.modeling.utils import silence_transformers_logs
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

    @silence_transformers_logs
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        model_type: str,
        content_type: ContentTypes,
        model_kwargs: Optional[Dict[str, Any]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
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
        :param feature_extractor_kwargs: FIXME unused
        """
        logger.info(
            f" 🤖 Loading '{pretrained_model_name_or_path}' "
            f"(Sentence Transformers {model_type if model_type else ''} model for {content_type} data)"
        )
        super().__init__()
        self.model_type = model_type
        self.content_type = content_type
        self.model = SentenceTransformer(pretrained_model_name_or_path, **(model_kwargs or {}))

    def encode(self, data: List[Any], **kwargs) -> torch.Tensor:
        """
        Generate the tensors representing the input data.

        Validates the inputs according to what the subclass declared in the `expected_inputs` property.
        Then passes the vectors to the `_forward()` method and returns its output untouched.
        """
        return self.model.encode(data, **kwargs)
