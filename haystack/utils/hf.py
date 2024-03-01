import copy
import inspect
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import requests

from haystack import logging
from haystack.dataclasses import StreamingChunk
from haystack.lazy_imports import LazyImport
from haystack.utils.auth import Secret
from haystack.utils.device import ComponentDevice

with LazyImport(message="Run 'pip install transformers[torch]'") as torch_import:
    import torch

with LazyImport(message="Run 'pip install transformers'") as transformers_import:
    from huggingface_hub import HfApi, InferenceClient
    from huggingface_hub.utils import RepositoryNotFoundError

logger = logging.getLogger(__name__)


class HFModelType(Enum):
    EMBEDDING = 1
    GENERATION = 2


def serialize_hf_model_kwargs(kwargs: Dict[str, Any]):
    """
    Recursively serialize HuggingFace specific model keyword arguments
    in-place to make them JSON serializable.

    :param kwargs: The keyword arguments to serialize
    """
    torch_import.check()

    for k, v in kwargs.items():
        # torch.dtype
        if isinstance(v, torch.dtype):
            kwargs[k] = str(v)

        if isinstance(v, dict):
            serialize_hf_model_kwargs(v)


def deserialize_hf_model_kwargs(kwargs: Dict[str, Any]):
    """
    Recursively deserialize HuggingFace specific model keyword arguments
    in-place to make them JSON serializable.

    :param kwargs: The keyword arguments to deserialize
    """
    torch_import.check()

    for k, v in kwargs.items():
        # torch.dtype
        if isinstance(v, str) and v.startswith("torch."):
            dtype_str = v.split(".")[1]
            dtype = getattr(torch, dtype_str, None)
            if dtype is not None and isinstance(dtype, torch.dtype):
                kwargs[k] = dtype

        if isinstance(v, dict):
            deserialize_hf_model_kwargs(v)


def resolve_hf_device_map(device: Optional[ComponentDevice], model_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Update `model_kwargs` to include the keyword argument `device_map` based on `device` if `device_map` is not
    already present in `model_kwargs`. This method is useful you want to force loading a transformers model when using
    `AutoModel.from_pretrained` to use `device_map`.

    We handle the edge case where `device` and `device_map` is specified by ignoring the `device` parameter and printing
    a warning.

    :param device: The device on which the model is loaded. If `None`, the default device is automatically
        selected.
    :param model_kwargs: Additional HF keyword arguments passed to `AutoModel.from_pretrained`.
        For details on what kwargs you can pass, see the model's documentation.
    """
    model_kwargs = copy.copy(model_kwargs) or {}
    if model_kwargs.get("device_map"):
        if device is not None:
            logger.warning(
                "The parameters `device` and `device_map` from `model_kwargs` are both provided. "
                "Ignoring `device` and using `device_map`."
            )
        # Resolve device if device_map is provided in model_kwargs
        device_map = model_kwargs["device_map"]
    else:
        device_map = ComponentDevice.resolve_device(device).to_hf()

    # Set up device_map which allows quantized loading and multi device inference
    # requires accelerate which is always installed when using `pip install transformers[torch]`
    model_kwargs["device_map"] = device_map

    return model_kwargs


def list_inference_deployed_models(headers: Optional[Dict] = None) -> List[str]:
    """
    List all currently deployed models on HF TGI free tier

    :param headers: Optional dictionary of headers to include in the request
    :return: list of all currently deployed models
    :raises Exception: If the request to the TGI API fails

    """
    resp = requests.get(
        "https://api-inference.huggingface.co/framework/text-generation-inference", headers=headers, timeout=10
    )

    payload = resp.json()
    if resp.status_code != 200:
        message = payload.get("error", "Unknown TGI error")
        error_type = payload.get("error_type", "Unknown TGI error type")
        raise Exception(f"Failed to fetch TGI deployed models: {message}. Error type: {error_type}")
    return [model["model_id"] for model in payload]


def check_valid_model(model_id: str, model_type: HFModelType, token: Optional[Secret]) -> None:
    """
    Check if the provided model ID corresponds to a valid model on HuggingFace Hub.
    Also check if the model is an embedding or generation model.

    :param model_id: A string representing the HuggingFace model ID.
    :param model_type: the model type, HFModelType.EMBEDDING or HFModelType.GENERATION
    :param token: The optional authentication token.
    :raises ValueError: If the model is not found or is not a embedding model.
    """
    transformers_import.check()

    api = HfApi()
    try:
        model_info = api.model_info(model_id, token=token.resolve_value() if token else None)
    except RepositoryNotFoundError as e:
        raise ValueError(
            f"Model {model_id} not found on HuggingFace Hub. Please provide a valid HuggingFace model_id."
        ) from e

    if model_type == HFModelType.EMBEDDING:
        allowed_model = model_info.pipeline_tag in ["sentence-similarity", "feature-extraction"]
        error_msg = f"Model {model_id} is not a embedding model. Please provide a embedding model."
    elif model_type == HFModelType.GENERATION:
        allowed_model = model_info.pipeline_tag in ["text-generation", "text2text-generation"]
        error_msg = f"Model {model_id} is not a text generation model. Please provide a text generation model."

    if not allowed_model:
        raise ValueError(error_msg)


def check_generation_params(kwargs: Optional[Dict[str, Any]], additional_accepted_params: Optional[List[str]] = None):
    """
    Check the provided generation parameters for validity.

    :param kwargs: A dictionary containing the generation parameters.
    :param additional_accepted_params: An optional list of strings representing additional accepted parameters.
    :raises ValueError: If any unknown text generation parameters are provided.
    """
    transformers_import.check()

    if kwargs:
        accepted_params = {
            param
            for param in inspect.signature(InferenceClient.text_generation).parameters.keys()
            if param not in ["self", "prompt"]
        }
        if additional_accepted_params:
            accepted_params.update(additional_accepted_params)
        unknown_params = set(kwargs.keys()) - accepted_params
        if unknown_params:
            raise ValueError(
                f"Unknown text generation parameters: {unknown_params}. The valid parameters are: {accepted_params}."
            )


with LazyImport(message="Run 'pip install transformers[torch]'") as torch_and_transformers_import:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, StoppingCriteria, TextStreamer

    transformers_import.check()
    torch_import.check()

    class StopWordsCriteria(StoppingCriteria):
        """
        Stops text generation in HuggingFace generators if any one of the stop words is generated.

        Note: When a stop word is encountered, the generation of new text is stopped.
        However, if the stop word is in the prompt itself, it can stop generating new text
        prematurely after the first token. This is particularly important for LLMs designed
        for dialogue generation. For these models, like for example mosaicml/mpt-7b-chat,
        the output includes both the new text and the original prompt. Therefore, it's important
        to make sure your prompt has no stop words.
        """

        def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            stop_words: List[str],
            device: Union[str, torch.device] = "cpu",
        ):
            super().__init__()
            # check if tokenizer is a valid tokenizer
            if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                raise ValueError(
                    f"Invalid tokenizer provided for StopWordsCriteria - {tokenizer}. "
                    f"Please provide a valid tokenizer from the HuggingFace Transformers library."
                )
            if not tokenizer.pad_token:
                if tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            encoded_stop_words = tokenizer(stop_words, add_special_tokens=False, padding=True, return_tensors="pt")
            self.stop_ids = encoded_stop_words.input_ids.to(device)

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in self.stop_ids:
                found_stop_word = self.is_stop_word_found(input_ids, stop_id)
                if found_stop_word:
                    return True
            return False

        def is_stop_word_found(self, generated_text_ids: torch.Tensor, stop_id: torch.Tensor) -> bool:
            generated_text_ids = generated_text_ids[-1]
            len_generated_text_ids = generated_text_ids.size(0)
            len_stop_id = stop_id.size(0)
            result = all(generated_text_ids[len_generated_text_ids - len_stop_id :].eq(stop_id))
            return result

    class HFTokenStreamingHandler(TextStreamer):
        """
        Streaming handler for HuggingFaceLocalGenerator and HuggingFaceLocalChatGenerator.

        Note: This is a helper class for HuggingFaceLocalGenerator & HuggingFaceLocalChatGenerator enabling streaming
        of generated text via Haystack Callable[StreamingChunk, None] callbacks.

        Do not use this class directly.
        """

        def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            stream_handler: Callable[[StreamingChunk], None],
            stop_words: Optional[List[str]] = None,
        ):
            super().__init__(tokenizer=tokenizer, skip_prompt=True)  # type: ignore
            self.token_handler = stream_handler
            self.stop_words = stop_words or []

        def on_finalized_text(self, word: str, stream_end: bool = False):
            word_to_send = word + "\n" if stream_end else word
            if word_to_send.strip() not in self.stop_words:
                self.token_handler(StreamingChunk(content=word_to_send))
