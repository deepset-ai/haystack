from typing import List, Optional, Dict, Any, overload

import logging
from haystack.lazy_imports import LazyImport

from haystack.preview import component, default_to_dict

with LazyImport(message="Run 'pip install gradientai'") as gradientai_import:
    from gradientai import Gradient

logger = logging.getLogger(__name__)


@component
class GradientGenerator:
    """
    LLM Generator interfacing [Gradient AI](https://gradient.ai/).

    Queries the LLM using Gradient AI's SDK ('gradientai' package).
    See [Gradient AI API](https://docs.gradient.ai/docs/sdk-quickstart) for more details.
    """

    @overload
    def __init__(
        self,
        *,
        access_token: Optional[str] = None,
        base_model_slug: str,
        host: Optional[str] = None,
        max_generated_token_count: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        *,
        access_token: Optional[str] = None,
        host: Optional[str] = None,
        max_generated_token_count: Optional[int] = None,
        model_adapter_id: str,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        ...

    def __init__(
        self,
        *,
        access_token: Optional[str] = None,
        base_model_slug: Optional[str] = None,
        host: Optional[str] = None,
        max_generated_token_count: Optional[int] = None,
        model_adapter_id: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        """
        Create a GradientGenerator component.

        :param access_token: The Gradient access token. If not provided it's read from the environment
                             variable GRADIENT_ACCESS_TOKEN.
        :param base_model_slug: The base model slug to use.
        :param host: The Gradient host. By default it uses https://api.gradient.ai/.
        :param max_generated_token_count: The maximum number of tokens to generate.
        :param model_adapter_id: The model adapter ID to use.
        :param temperature: The temperature to use.
        :param top_k: The top k to use.
        :param top_p: The top p to use.
        :param workspace_id: The Gradient workspace ID. If not provided it's read from the environment
                             variable GRADIENT_WORKSPACE_ID.
        """
        self._access_token = access_token
        self._base_model_slug = base_model_slug
        self._host = host
        self._max_generated_token_count = max_generated_token_count
        self._model_adapter_id = model_adapter_id
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._workspace_id = workspace_id

        if (base_model_slug is None and model_adapter_id is None) or (
            isinstance(base_model_slug, str) and isinstance(model_adapter_id, str)
        ):
            raise ValueError("expected be provided exactly one of base_model_slug or model_adapter_id")

        self._gradient = Gradient(access_token=access_token, host=host, workspace_id=workspace_id)
        if isinstance(base_model_slug, str):
            self._model = self._gradient.get_base_model(base_model_slug=base_model_slug)
        if isinstance(model_adapter_id, str):
            self._model = self._gradient.get_model_adapter(model_adapter_id=model_adapter_id)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            base_model_slug=self._base_model_slug,
            host=self._host,
            max_generated_token_count=self._max_generated_token_count,
            model_adapter_id=self._model_adapter_id,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
            workspace_id=self._workspace_id,
        )

    @component.output_types(replies=List[str])
    def run(self, prompt: str):
        """
        Queries the LLM with the prompt to produce replies.

        :param prompt: The prompt to be sent to the generative model.
        """
        resp = self._model.complete(
            query=prompt,
            max_generated_token_count=self._max_generated_token_count,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
        )
        return {"replies": [resp.generated_output]}
