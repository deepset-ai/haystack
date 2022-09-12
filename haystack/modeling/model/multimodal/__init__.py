from typing import Optional, Union, Dict, Any, Type, List

import logging
from pathlib import Path
from transformers import AutoConfig
import torch
from torch.nn import DataParallel

from haystack.modeling.model.multimodal.base import HaystackModel
from haystack.modeling.model.multimodal.transformers import HaystackTextTransformerModel
from haystack.modeling.model.multimodal.sentence_transformers import HaystackSentenceTransformerModel
from haystack.schema import ContentTypes


logger = logging.getLogger(__name__)


#: Match the name of the HuggingFace Model class to the corresponding Haystack wrapper
HUGGINGFACE_TO_HAYSTACK: Dict[str, Type[HaystackModel]] = {
    "AutoModel": HaystackTextTransformerModel,
    "Albert": HaystackTextTransformerModel,
    "Bert": HaystackTextTransformerModel,
    "BigBird": HaystackTextTransformerModel,
    "Camembert": HaystackTextTransformerModel,
    "Codebert": HaystackTextTransformerModel,
    "DebertaV2": HaystackTextTransformerModel,
    "DistilBert": HaystackTextTransformerModel,
    "Electra": HaystackTextTransformerModel,
    "GloVe": HaystackTextTransformerModel,
    "MiniLM": HaystackTextTransformerModel,
    "Roberta": HaystackTextTransformerModel,
    "Umberto": HaystackTextTransformerModel,
    "Word2Vec": HaystackTextTransformerModel,
    "XLMRoberta": HaystackTextTransformerModel,
    "XLNet": HaystackTextTransformerModel,
    # These models are supported only through sentence-tranformers
    "CLIP": HaystackSentenceTransformerModel,
    "MPNet": HaystackSentenceTransformerModel,
    # Later
    # "DPRContextEncoder": HaystackSentenceTransformerModel,
    # "DPRQuestionEncoder": HaystackSentenceTransformerModel,
}


#: HF Capitalization pairs. Contains alternative capitalizations.
HUGGINGFACE_CAPITALIZE = {
    "big-bird": "BigBird",
    "deberta-v2": "DebertaV2",
    "xlm-roberta": "XLMRoberta",
    **{k.lower(): k for k in HUGGINGFACE_TO_HAYSTACK.keys()},
}


#: Default parameters to be given at init time to some specific models
DEFAULT_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {}


def get_model(
    pretrained_model_name_or_path: Union[Path, str],
    content_type: Optional[ContentTypes] = None,
    devices: Optional[List[torch.device]] = None,
    autoconfig_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
    pooler_kwargs: Optional[Dict[str, Any]] = None,
) -> HaystackModel:
    """
    Load a pretrained language model by specifying its name and either downloading the model from HuggingFace Hub
    (if it's given a name) or loading it from disk (if it's given a path).

    See all supported model variations at: https://huggingface.co/models.
    The appropriate language model class is inferred automatically from the model's configuration and/or its name.

    :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
    :param content_type: the type (text, image, ...) of content the model is supposed to handle.
    :param autoconfig_kwargs: Additional keyword arguments to pass to AutoConfig, like the revision or the auth key.
    :param model_kwargs: Additional keyword arguments to pass to the language model constructor.
        Haystack applies some default parameters to some models. They can be overridden by users by specifying the
        desired value in this parameter. See `DEFAULT_MODEL_PARAMS`.
    :param feature_extractor_kwargs: dictionary of parameters to pass to the feature extractor's initialization (revision, use_auth_key, etc...)
        Haystack applies some default parameters to some models. They can be overridden by users by specifying the
        desired value in this parameter. See `DEFAULT_MODEL_PARAMS`.
    :param pooler_kwargs: dictionary of parameters to pass to the pooler's initialization (summary_last_dropout, summary_activation, etc...)
        Haystack applies some default parameters to some models. They can be overridden by users by specifying the
        desired value in this parameter. See `POOLER_PARAMETERS`.
    """
    if not pretrained_model_name_or_path or not isinstance(pretrained_model_name_or_path, (str, Path)):
        raise ValueError(
            f"{pretrained_model_name_or_path} is not a valid 'pretrained_model_name_or_path' value. "
            "Please provide a string or a Path object."
        )

    model_name = str(pretrained_model_name_or_path)
    model_type = ""

    # Prepare the kwargs the model wrapper expects (see each wrapper's init for details)
    wrapper_kwarg_groups = {}
    wrapper_kwarg_groups["model_kwargs"] = {**DEFAULT_MODEL_PARAMS.get(model_type, {}), **(model_kwargs or {})}

    if model_name.startswith("sentence-transformers/"):
        # SentenceTransformers are much faster, so use them whenever possible
        # FIXME find a way to distinguish them better!
        model_wrapper_class = HaystackSentenceTransformerModel
        try:
            # Use AutoConfig to log some more info about the model class
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name, **(autoconfig_kwargs or {}))
            model_type = config.model_type
        except Exception as e:
            logger.debug(f"Can't find model type for {pretrained_model_name_or_path}: {e}")

        if feature_extractor_kwargs is not None:
            logger.warning(
                "Can't forward feature_extractor_kwargs to a SentenceTransformers model. "
                "These kwargs are being dropped. "
                f"Content of feature_extractor_kwargs: {feature_extractor_kwargs}"
            )

    else:
        # Use AutoConfig to understand the model class
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name, **(autoconfig_kwargs or {}))
        if not config.model_type:
            logger.error(
                f"Model type not understood for '{model_name}'. Please provide the name of a model that can be "
                f"downloaded from the Model Hub.\nUsing the AutoModel class for '{pretrained_model_name_or_path}'. "
                "This can cause crashes!"
            )
            model_type = "AutoModel"
        else:
            model_type = HUGGINGFACE_CAPITALIZE.get(config.model_type.lower(), "AutoModel")

        # Find the HF class corresponding to this model type
        model_wrapper_class = HUGGINGFACE_TO_HAYSTACK.get(model_type)
        if not model_wrapper_class:
            raise ValueError(
                f"The type of the given model (name/path: {pretrained_model_name_or_path}, detected type: {model_type}) "
                "is not supported by Haystack or was not correctly identified. Please use supported models only. "
                f"Supported model types: {', '.join(HUGGINGFACE_TO_HAYSTACK.keys())}"
            )
        wrapper_kwarg_groups["feature_extractor_kwargs"] = feature_extractor_kwargs
        wrapper_kwarg_groups["pooler_kwargs"] = pooler_kwargs

    # Instantiate the model's wrapper
    model_wrapper = model_wrapper_class(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        model_type=model_type,
        content_type=content_type,
        **wrapper_kwarg_groups,
    )

    if devices:
        if len(devices) > 1:
            model_wrapper.model = DataParallel(model_wrapper.model, device_ids=devices)
        else:
            model_wrapper.model.to(devices[0])

    return model_wrapper
