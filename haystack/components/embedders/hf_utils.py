from typing import Optional

from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install transformers'") as transformers_import:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError


def check_valid_model(model_id: str, token: Optional[str]) -> None:
    """
    Check if the provided model ID corresponds to a valid model on HuggingFace Hub.
    Also check if the model is a embedding model.

    :param model_id: A string representing the HuggingFace model ID.
    :param token: An optional string representing the authentication token.
    :raises ValueError: If the model is not found or is not a embedding model.
    """
    transformers_import.check()

    api = HfApi()
    try:
        model_info = api.model_info(model_id, token=token)
    except RepositoryNotFoundError as e:
        raise ValueError(
            f"Model {model_id} not found on HuggingFace Hub. Please provide a valid HuggingFace model_id."
        ) from e

    allowed_model = model_info.pipeline_tag in ["sentence-similarity", "feature-extraction"]
    if not allowed_model:
        raise ValueError(f"Model {model_id} is not a embedding model. Please provide a embedding model.")
