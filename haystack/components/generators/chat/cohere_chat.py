import dataclasses
import logging
import os
from typing import Optional, List, Callable, Dict, Any

from haystack import component, default_from_dict, default_to_dict
from haystack.components.generators.utils import serialize_callback_handler, deserialize_callback_handler
from haystack.dataclasses import StreamingChunk, ChatMessage
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install cohere'") as cohere_import:
    import cohere
logger = logging.getLogger(__name__)


class CohereChatGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: str = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        cohere_import.check()

        if not api_key:
            api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "CohereChatGenerator needs an API key to run. Either provide it as init parameter or set the env var COHERE_API_KEY."
            )

        if not api_base_url:
            api_base_url = cohere.COHERE_API_URL

        self.api_key = api_key
        self.model_name = model_name
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.generation_kwargs = generation_kwargs
        self.model_parameters = kwargs
        self.client = cohere.Client(api_key=self.api_key, api_url=self.api_base_url)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        :return: The serialized component as a dictionary.
        """
        callback_name = serialize_callback_handler(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model_name=self.model_name,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohereChatGenerator":
        """
        Deserialize this component from a dictionary.
        :param data: The dictionary representation of this component.
        :return: The deserialized component instance.
        """
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callback_handler(serialized_callback_handler)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        ...

    def _check_finish_reason(self, message: ChatMessage) -> None:
        """
        Check the `finish_reason` returned with the OpenAI completions.
        If the `finish_reason` is `length` or `content_filter`, log a warning.
        :param message: The message returned by the LLM.
        """
        ...
