import inspect
from typing import Any, Dict, List, Optional

from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install transformers'") as transformers_import:
    from huggingface_hub import InferenceClient, HfApi
    from huggingface_hub.utils import RepositoryNotFoundError


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


def check_valid_model(model_id: str, token: Optional[str]) -> None:
    """
    Check if the provided model ID corresponds to a valid model on HuggingFace Hub.
    Also check if the model is a text generation model.

    :param model_id: A string representing the HuggingFace model ID.
    :param token: An optional string representing the authentication token.
    :raises ValueError: If the model is not found or is not a text generation model.
    """
    transformers_import.check()

    api = HfApi()
    try:
        model_info = api.model_info(model_id, token=token)
    except RepositoryNotFoundError as e:
        raise ValueError(
            f"Model {model_id} not found on HuggingFace Hub. Please provide a valid HuggingFace model_id."
        ) from e

    allowed_model = model_info.pipeline_tag in ["text-generation", "text2text-generation"]
    if not allowed_model:
        raise ValueError(f"Model {model_id} is not a text generation model. Please provide a text generation model.")
