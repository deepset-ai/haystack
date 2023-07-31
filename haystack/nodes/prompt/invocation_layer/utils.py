from typing import Optional, Union

from huggingface_hub import model_info


def has_azure_parameters(**kwargs) -> bool:
    azure_params = ["azure_base_url", "azure_deployment_name"]
    return any(kwargs.get(param) for param in azure_params)


def get_task(model: str, use_auth_token: Optional[Union[str, bool]] = None, timeout: float = 3.0) -> Optional[str]:
    """
    Retrieve the task (pipeline tag) associated with a given model.

    Simplified version of transformers.pipelines.get_task with support for timeouts
    """
    try:
        return model_info(model, token=use_auth_token, timeout=timeout).pipeline_tag
    except Exception as e:
        raise RuntimeError(f"The task for {model} couldn't be found!") from e
