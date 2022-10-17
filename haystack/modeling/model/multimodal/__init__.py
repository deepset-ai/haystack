from typing import Optional, Union, Dict, Any, Type, List

import logging
from pathlib import Path
from transformers import AutoConfig
import torch
from huggingface_hub import hf_hub_download

from haystack.modeling.model.multimodal.base import HaystackModel
from haystack.modeling.model.multimodal.sentence_transformers import HaystackSentenceTransformerModel


logger = logging.getLogger(__name__)


#: Match the name of the HuggingFace Model class to the corresponding Haystack wrapper
HUGGINGFACE_TO_HAYSTACK: Dict[str, Type[HaystackModel]] = {
    "CLIP": HaystackSentenceTransformerModel,
    "MPNet": HaystackSentenceTransformerModel,
    "DPRContextEncoder": HaystackSentenceTransformerModel,
    "DPRQuestionEncoder": HaystackSentenceTransformerModel,
}


#: HF Capitalization pairs. Contains alternative capitalizations.
HUGGINGFACE_CAPITALIZE = {
    "big-bird": "BigBird",
    "deberta-v2": "DebertaV2",
    "xlm-roberta": "XLMRoberta",
    **{k.lower(): k for k in HUGGINGFACE_TO_HAYSTACK.keys()},
}


def get_model(
    pretrained_model_name_or_path: Union[Path, str],
    content_type: str,  # change to ContentTypes starting Python3.8
    devices: Optional[List[torch.device]] = None,
    autoconfig_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
    pooler_kwargs: Optional[Dict[str, Any]] = None,
) -> HaystackModel:
    """
    Load a pretrained language model by specifying its name and either downloading the model from the Hugging Face hub
    (if it's given a model identifier from Hugging Face Hub) or loading it from disk (if it's given a local path).

    For all supported model variations, see [Models](https://huggingface.co/models).
    The appropriate language model class is inferred automatically from the model's configuration or its name.

    :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
    :param content_type: The type (text, image, ...) of content the model should handle.
    :param autoconfig_kwargs: Additional keyword arguments to pass to AutoConfig, like the revision or the auth key.
    :param model_kwargs: Additional keyword arguments to pass to the language model constructor.
        Haystack applies some default parameters to some models. You can override them by specifying the
        desired value in this parameter. See `DEFAULT_MODEL_PARAMS`.
    :param feature_extractor_kwargs: A dictionary of parameters to pass to the feature extractor's initialization (revision, use_auth_key, etc...)
        Haystack applies some default parameters to some models. You can override them by specifying the
        desired value in this parameter. See `DEFAULT_MODEL_PARAMS`.
    :param pooler_kwargs: A dictionary of parameters to pass to the pooler's initialization (summary_last_dropout, summary_activation, etc...)
        Haystack applies some default parameters to some models. You can override them by specifying the
        desired value in this parameter. See `POOLER_PARAMETERS`.
    """
    autoconfig_kwargs = autoconfig_kwargs or {}
    model_kwargs = model_kwargs or {}
    feature_extractor_kwargs = feature_extractor_kwargs or {}
    pooler_kwargs = pooler_kwargs or {}

    if not pretrained_model_name_or_path or not isinstance(pretrained_model_name_or_path, (str, Path)):
        raise ValueError(
            f"{pretrained_model_name_or_path} is not a valid 'pretrained_model_name_or_path' value. "
            "Please provide a string or a Path object."
        )
    model_name = str(pretrained_model_name_or_path)
    model_type: Optional[str] = ""
    model_wrapper_class: Type[HaystackModel]

    # Prepare the kwargs the model wrapper expects (see each wrapper's init for details)
    wrapper_kwarg_groups = {}
    wrapper_kwarg_groups["model_kwargs"] = model_kwargs

    # SentenceTransformers are much faster, so use them whenever possible
    if _is_sentence_transformers_model(
        pretrained_model_name_or_path, use_auth_token=autoconfig_kwargs.get("use_auth_token", False)
    ):
        model_wrapper_class = HaystackSentenceTransformerModel
        try:
            # Use AutoConfig to log some more info about the model class
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name, **autoconfig_kwargs)
            model_type = config.model_type
        except Exception as e:
            logger.debug(f"Can't find model type for {pretrained_model_name_or_path}: {e}")

        if feature_extractor_kwargs is not None:
            logger.debug(
                "Can't forward feature_extractor_kwargs to a SentenceTransformers model. "
                "These kwargs are being dropped. "
                f"Content of feature_extractor_kwargs: {feature_extractor_kwargs}"
            )

    else:
        # Use AutoConfig to understand the model class
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name, **autoconfig_kwargs)
        if not config.model_type:
            logger.error(
                f"Model type not understood for '{pretrained_model_name_or_path}'. Please provide the name of "
                "a model that can be downloaded from the Model Hub.\nUsing the AutoModel class. "
                "THIS CAN CAUSE CRASHES and won't work for models that are not working with text."
            )
            model_type = None
        else:
            try:
                model_type = HUGGINGFACE_CAPITALIZE[config.model_type.lower()]
            except KeyError as e:
                logger.error(
                    f"Haystack doesn't support model '{pretrained_model_name_or_path}' (type '{config.model_type.lower()}') "
                    "We'll use the AutoModel class for it. "
                    "THIS CAN CAUSE CRASHES and won't work for models that are not working with text. "
                    f"Supported model types: {', '.join(HUGGINGFACE_CAPITALIZE.keys())}"
                )
                model_type = None

        # Find the HF class corresponding to this model type
        try:
            model_wrapper_class = HUGGINGFACE_TO_HAYSTACK[model_type or "AutoModel"]
        except KeyError as e:
            raise ValueError(
                f"The type of the given model (name/path: {pretrained_model_name_or_path}, detected type: {model_type or config.model_type}) "
                "is not supported by Haystack or was not correctly identified. Please use supported models only. "
                f"Supported model types: {', '.join(HUGGINGFACE_TO_HAYSTACK.keys())}"
            ) from e

        if feature_extractor_kwargs:
            wrapper_kwarg_groups["feature_extractor_kwargs"] = feature_extractor_kwargs
        if pooler_kwargs:
            wrapper_kwarg_groups["pooler_kwargs"] = pooler_kwargs

    # Instantiate the model's wrapper
    model_wrapper = model_wrapper_class(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        model_type=model_type,
        content_type=content_type,
        **wrapper_kwarg_groups,
    )
    model_wrapper.to(devices)

    return model_wrapper


def _is_sentence_transformers_model(pretrained_model_name_or_path: Union[Path, str], use_auth_token: Union[bool, str]):

    # Check if sentence transformers config file is in local path
    if Path(pretrained_model_name_or_path).exists():
        if (Path(pretrained_model_name_or_path) / "config_sentence_transformers.json").exists():
            return True

    # Check if sentence transformers config file is in model hub
    try:
        hf_hub_download(
            repo_id=str(pretrained_model_name_or_path),
            filename="config_sentence_transformers.json",
            use_auth_token=use_auth_token,
        )
        return True

    except Exception as e:
        logger.debug("%s not found in model hub: an error occurred. Error: %s", pretrained_model_name_or_path, e)

        # Pattern matching the name as a last resort
        if str(pretrained_model_name_or_path).startswith("sentence-transformers"):
            logger.debug(
                "The model name starts with 'sentence-transformers': assuming this is a Sentence Transformers model."
            )
            return True

        logger.debug(
            "The model name doesn't start with 'sentence-transformers': assuming this is NOT a Sentence Transformers model."
        )
        return False
