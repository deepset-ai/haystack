import logging
import sys
from typing import Any, Callable, Dict, List, Optional

from haystack.lazy_imports import LazyImport
from haystack.preview import DeserializationError, component, default_from_dict, default_to_dict

with LazyImport(message="Run 'pip install cohere'") as cohere_import:
    from cohere import Client

logger = logging.getLogger(__name__)


API_BASE_URL = "https://api.cohere.ai"


def default_streaming_callback(chunk):
    """
    Default callback function for streaming responses from Cohere API.
    Prints the tokens of the first completion to stdout as soon as they are received and returns the chunk unchanged.
    """
    print(chunk.text, flush=True, end="")


@component
class CohereGenerator:
    """Cohere Generator compatible with Cohere generate endpoint"""

    def __init__(
        self,
        api_key: str,
        model: str = "command",
        streaming_callback: Optional[Callable] = None,
        api_base_url: str = API_BASE_URL,
        **kwargs,
    ):
        """
        Args:
            api_key (str): The API key for the Cohere API.
            model_name (str): The name of the model to use.
            streaming_callback (Callable, optional): A callback function to be called with the streaming response. Defaults to None.
            api_base_url (str, optional): The base URL of the Cohere API. Defaults to "https://api.cohere.ai".
        """
        self.api_key = api_key
        self.model = model
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.model_parameters = kwargs
        self.client = Client(api_key=self.api_key, api_url=self.api_base_url)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        if self.streaming_callback:
            module = self.streaming_callback.__module__
            if module == "builtins":
                callback_name = self.streaming_callback.__name__
            else:
                callback_name = f"{module}.{self.streaming_callback.__name__}"
        else:
            callback_name = None

        return default_to_dict(
            self,
            api_key=self.api_key,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            **self.model_parameters,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohereGenerator":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        streaming_callback = None
        if "streaming_callback" in init_params and init_params["streaming_callback"]:
            parts = init_params["streaming_callback"].split(".")
            module_name = ".".join(parts[:-1])
            function_name = parts[-1]
            module = sys.modules.get(module_name, None)
            if not module:
                raise DeserializationError(f"Could not locate the module of the streaming callback: {module_name}")
            streaming_callback = getattr(module, function_name, None)
            if not streaming_callback:
                raise DeserializationError(f"Could not locate the streaming callback: {function_name}")
            data["init_parameters"]["streaming_callback"] = streaming_callback
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(self, prompt: str):
        """
        Queries the LLM with the prompts to produce replies.
        :param prompt: The prompts to be sent to the generative model.
        """
        response = self.client.generate(
            model=self.model, prompt=prompt, stream=self.streaming_callback is not None, **self.model_parameters
        )
        replies: List[str]
        metadata: List[Dict[str, Any]]
        if self.streaming_callback:
            metadata_dict: Dict[str, Any] = {}
            for chunk in response:
                self.streaming_callback(chunk)
                metadata_dict["index"] = chunk.index
            replies = response.texts
            metadata_dict["finish_reason"] = response.finish_reason
            metadata = [metadata_dict]
            self._check_truncated_answers(metadata)
            return {"replies": replies, "metadata": metadata}
        metadata = [{"finish_reason": response[0].finish_reason}]
        replies = [response[0].text]
        self._check_truncated_answers(metadata)
        return {"replies": replies, "metadata": metadata}

    def _check_truncated_answers(self, metadata: List[Dict[str, Any]]):
        """
        Check the `finish_reason` returned with the Cohere response.
        If the `finish_reason` is `MAX_TOKEN`, log a warning to the user.
        """
        if metadata[0]["finish_reason"] == "MAX_TOKENS":
            logger.warning(
                "Responses have been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions."
            )
