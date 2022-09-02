from typing import Tuple, Set, Optional, Union, Dict, Any, Type, Literal

import logging
from pathlib import Path
from functools import wraps
from abc import ABC, abstractmethod

import torch
from torch import nn
import transformers
from transformers import AutoConfig, PreTrainedModel
from sentence_transformers import SentenceTransformer

from haystack.errors import ModelingError
import haystack.modeling.model._clip_adapter as clip_adapters


logger = logging.getLogger(__name__)


def silence_transformers_logs(from_pretrained_func):
    """
    A wrapper that raises the log level of Transformers to
    ERROR to hide some unnecessary warnings.
    """

    @wraps(from_pretrained_func)
    def quiet_from_pretrained_func(cls, *args, **kwargs):

        # Raise the log level of Transformers
        t_logger = logging.getLogger("transformers")
        original_log_level = t_logger.level
        t_logger.setLevel(logging.ERROR)

        result = from_pretrained_func(cls, *args, **kwargs)

        # Restore the log level
        t_logger.setLevel(original_log_level)

        return result

    return quiet_from_pretrained_func


class MultiModalModel(nn.Module, ABC):
    """
    Parent class for models that can embed different data types into **COMPARABLE** semantic vector spaces.

    These models read feature vectors (generated by a feature extractor) and return vectors that capture
    the meaning of the original data.

    Models inheriting from MultiModalModel are designed to be used in parallel one with the other
    in multimodal retrieval settings, for example image retrieval from a text query, combined table and
    text retrieval, etc... They must therefore embed their source data into comparable vector spaces.
    """

    @silence_transformers_logs
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        model_type: str,
        content_type: Optional[str] = "text",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param pretrained_model_name_or_path: name of the model to load
        :param model_type: the value of model_type from the model's Config
        :param content_type: the type of data (text, image, ...) the model is supposed to process.
        :param model_kwargs: dictionary of parameters to pass to the model's initialization (revision, use_auth_key, etc...)
            Haystack applies some default parameters to some models. They can be overridden by users by specifying the
            desired value in this parameter. See `HUGGINGFACE_DEFAULT_MODEL_PARAMS`.
        """
        logger.info(
            f" 🤖 Loading model '{pretrained_model_name_or_path}' ({model_type if model_type else ''} for {content_type} data)"
        )
        super().__init__()
        self.model_type = model_type
        self.content_type = content_type

        model_params = HUGGINGFACE_DEFAULT_MODEL_PARAMS.get(model_type, {}) | (model_kwargs or {})
        model_class: PreTrainedModel = getattr(transformers, model_type, None) or getattr(
            clip_adapters, model_type, None
        )
        self.model = model_class.from_pretrained(str(pretrained_model_name_or_path), **(model_params or {}))

        self.model.eval()  # Put model in evaluation/inference mode (in contrast with training mode)

    @property
    @abstractmethod
    def output_dims():
        """
        The output dimension of this language model
        """
        pass

    @property
    @abstractmethod
    def expected_inputs(self) -> Tuple[Set[str], Set[str]]:
        """
        Returns a tuple, (List[mandatory arg names], List[optional arg names])
        """
        pass

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Performs a forward pass of the LM model.

        Validates the inputs according to what the subclass declared in the `expected_inputs` property,
        then hands over the params to the actual model. It passes the vectors as they are to the model
        and returns the pooler output of the model's forward pass (assuming the model has
        a pooler and populates the `pooler_output` attribute of its output).
        """
        mandatory_args, optional_args = self.expected_inputs
        all_args = mandatory_args | optional_args
        given_args = set(kwargs.keys())
        if not (given_args >= mandatory_args and given_args <= all_args):
            raise ModelingError(
                "The input parameters do not match the model's expectations.\n"
                f"Input names: {', '.join(sorted(kwargs.keys()))}\n"
                f"Expected: {', '.join(sorted(all_args))} (where {', '.join(sorted(mandatory_args))} are mandatory)"
            )
        with torch.no_grad():
            output = self._forward(**kwargs)
        return output

    def _forward(self, **kwargs) -> torch.Tensor:
        """
        Hook for subclasses to run their own code before or after the inference.

        The default implementation simply returns the pooler output of the model's forward pass.
        """
        output = self.model(**kwargs)
        return output.pooler_output


class TextModel(MultiModalModel):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        model_type: str,
        content_type: Optional[Literal["text"]] = "text",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if content_type != "text":
            raise ModelingError(f"{pretrained_model_name_or_path} can't handle data of type {content_type}")

        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_type=model_type,
            content_type="text",
            model_kwargs=model_kwargs,
        )

    @property
    def expected_inputs(self) -> Tuple[Set[str], Set[str]]:
        return {"input_ids", "token_type_ids", "attention_mask"}, set()

    @property
    def output_dims(self) -> int:
        return self.dim  # "hidden_size", "d_model",


class ImageModel(MultiModalModel):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        model_type: str,
        content_type: Optional[Literal["image"]] = "image",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if content_type != "image":
            raise ModelingError(f"{pretrained_model_name_or_path} can't handle data of type {content_type}")

        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_type=model_type,
            content_type="image",
            model_kwargs=model_kwargs,
        )

    @property
    def expected_inputs(self) -> Tuple[Set[str], Set[str]]:
        return {"pixel_values"}, {"bool_masked_pos", "head_mask"}

    @property
    def output_dims(self) -> int:
        return self.window_size


class CLIPModel(MultiModalModel):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        model_type: str,
        content_type: Optional[Union[Literal["text"], Literal["image"]]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Supports the specific initialization path of CLIP models.

        :param pretrained_model_name_or_path: name of the model to load
        :param model_type: the value of model_type from the model's Config
        :param content_type: the type of data (text, image, ...) the model is supposed to process.
        :param model_kwargs: dictionary of parameters to pass to the model's initialization (revision, use_auth_key, etc...)
        """
        model_kwargs = model_kwargs or {}
        self.projection_dim = model_kwargs.pop("projection_dims", 512)
        self.logit_scale_init_value = model_kwargs.pop("logit_scale_init_value", 2.6592)

        if content_type == "text":
            model_type = "CLIPModelAdapterText"
            self._expected_inputs = {"input_ids", "token_type_ids", "attention_mask"}, set()
            self._forward = self._text_forward

        elif content_type == "image":
            model_type = "CLIPModelAdapterVision"
            self._expected_inputs = {"pixel_values"}, set()
            self._forward = self._image_forward

        else:
            raise ModelingError(f"{pretrained_model_name_or_path} can't handle data of type {content_type}")

        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_type=model_type,
            content_type=content_type,
            model_kwargs=model_kwargs,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * self.logit_scale_init_value)

    @property
    def expected_inputs(self) -> Tuple[Set[str], Set[str]]:
        return self._expected_inputs

    @property
    def output_dims(self) -> int:
        return self.dim  # "hidden_size", "d_model",

    def _text_forward(self, **kwargs) -> torch.Tensor:
        """
        Forward adaptation for CLIP on text input.

        Renames one input tensor to match the ones coming out of the feature-extractor (TODO check why this is needed)
        and performs a projection and normalization on the resulting output.
        """
        kwargs["position_ids"] = kwargs.pop("token_type_ids")  # Rename position_ids into token_type_ids
        output = self.model(**kwargs)
        embeds = output[0]
        return embeds

    def _image_forward(self, **kwargs) -> torch.Tensor:
        """
        Forward adaptation for CLIP on image input.

        Performs a projection and normalization on the resulting output.
        """
        output = self.model(**kwargs)
        embeds = output[0]
        return embeds


#: Match the name of the HuggingFace Model class to the corresponding Haystack wrapper
HUGGINGFACE_TO_HAYSTACK: Dict[str, Type[MultiModalModel]] = {"AutoModel": TextModel, "CLIP": CLIPModel}

#: HF Capitalization pairs. Contains alternative capitalizations.
HUGGINGFACE_CAPITALIZE = {"clip": "CLIP"}

#: Default parameters to be given at init time to some specific models
HUGGINGFACE_DEFAULT_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {}


def get_mm_language_model(
    pretrained_model_name_or_path: Union[Path, str],
    content_type: Optional[str] = None,
    autoconfig_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> MultiModalModel:
    """
    Load a pretrained language model by specifying its name and downloading the model.

    See all supported model variations at: https://huggingface.co/models.
    The appropriate language model class is inferred automatically from model configuration.

    :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
    :param content_type: the type (text, image, ...) of content the model is supposed to handle.
        Some models, like CLIP, need a different initialization depending on such type.
    :param autoconfig_kwargs: Additional keyword arguments to pass to AutoConfig, like the revision or the auth key.
    :param model_kwargs: Additional keyword arguments to pass to the language model constructor.
        Haystack applies some default parameters to some models. They can be overridden by users by specifying the
        desired value in this parameter. See `HUGGINGFACE_DEFAULT_MODEL_PARAMS`.
    """

    if not pretrained_model_name_or_path or not isinstance(pretrained_model_name_or_path, (str, Path)):
        raise ValueError(f"{pretrained_model_name_or_path} is not a valid pretrained_model_name_or_path parameter")

    model_name = str(pretrained_model_name_or_path)

    # Use AutoConfig to understand the model class
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name, **(autoconfig_kwargs or {}))
    if not config.model_type:
        logger.error(
            f"Model type not understood for '{model_name}'. Please provide the name of a model that can be "
            f"downloaded from the Model Hub.\nUsing the AutoModel class for '{pretrained_model_name_or_path}'. "
            "This can cause crashes!"
        )
        model_type = HUGGINGFACE_CAPITALIZE.get(config.model_type.lower(), "AutoModel")

    # Find the HF class corresponding to this model type
    language_model_class = HUGGINGFACE_TO_HAYSTACK.get(model_type)
    if not language_model_class:
        raise ValueError(
            f"The type of the given model (name/path: {pretrained_model_name_or_path}, detected type: {model_type}) "
            "is not supported by Haystack or was not correctly identified. Please use supported models only. "
            f"Supported model types: {', '.join(HUGGINGFACE_TO_HAYSTACK.keys())}"
        )

    # Instantiate the model's wrapper
    language_model = language_model_class(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        model_type=model_type,
        content_type=content_type,
        model_kwargs=model_kwargs,
    )
    return language_model


def get_sentence_tranformers_model(
    pretrained_model_name_or_path: Union[Path, str],
    content_type: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[MultiModalModel, SentenceTransformer]:

    st_model = str(pretrained_model_name_or_path).replace("sentence-transformers/", "")
    logger.info(
        f" 🤖 Loading model '{pretrained_model_name_or_path}' " f"(SentenceTransformer model for {content_type} data)"
    )
    return SentenceTransformer(st_model, **model_kwargs)
