from typing import List, Optional, Dict, Any, overload

import logging
from haystack.lazy_imports import LazyImport

from haystack.preview import component, default_to_dict

with LazyImport(message="Run 'pip install gradientai'") as gradientai_import:
    from gradientai import Gradient

logger = logging.getLogger(__name__)


@component
class GradientGenerator:
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
        return default_to_dict(
            self,
            access_token=self._access_token,
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
        resp = self._model.complete(
            query=prompt,
            max_generated_token_count=self._max_generated_token_count,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
        )
        return {"replies": [resp.generated_output]}
