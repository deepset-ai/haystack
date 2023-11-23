import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from haystack.preview.lazy_imports import LazyImport
from haystack.preview import DeserializationError, component, default_from_dict, default_to_dict

with LazyImport(message="Run 'pip install cohere'") as cohere_import:
    from cohere import Client, COHERE_API_URL

logger = logging.getLogger(__name__)


@component
class CohereGenerator:
    """LLM Generator compatible with Cohere's generate endpoint.

    Queries the LLM using Cohere's API. Invocations are made using 'cohere' package.
    See [Cohere API](https://docs.cohere.com/reference/generate) for more details.

    Example usage:

    ```python
    from haystack.preview.generators import CohereGenerator
    generator = CohereGenerator(api_key="test-api-key")
    generator.run(prompt="What's the capital of France?")
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "command",
        streaming_callback: Optional[Callable] = None,
        api_base_url: str = COHERE_API_URL,
        **kwargs,
    ):
        """
         Instantiates a `CohereGenerator` component.
        :param api_key: The API key for the Cohere API. If not set, it will be read from the COHERE_API_KEY env var.
        :param model_name: The name of the model to use. Available models are: [command, command-light, command-nightly, command-nightly-light]. Defaults to "command".
        :param streaming_callback: A callback function to be called with the streaming response. Defaults to None.
        :param api_base_url: The base URL of the Cohere API. Defaults to "https://api.cohere.ai".
        :param kwargs: Additional model parameters. These will be used during generation. Refer to https://docs.cohere.com/reference/generate for more details.
          Some of the parameters are:
          - 'max_tokens': The maximum number of tokens to be generated. Defaults to 1024.
          - 'truncate': One of NONE|START|END to specify how the API will handle inputs longer than the maximum token length. Defaults to END.
          - 'temperature': A non-negative float that tunes the degree of randomness in generation. Lower temperatures mean less random generations.
          - 'preset': Identifier of a custom preset. A preset is a combination of parameters, such as prompt, temperature etc. You can create presets in the playground.
          - 'end_sequences': The generated text will be cut at the beginning of the earliest occurrence of an end sequence. The sequence will be excluded from the text.
          - 'stop_sequences': The generated text will be cut at the end of the earliest occurrence of a stop sequence. The sequence will be included the text.
          - 'k': Defaults to 0, min value of 0.01, max value of 0.99.
          - 'p': Ensures that only the most likely tokens, with total probability mass of `p`, are considered for generation at each step. If both `k` and `p` are enabled, `p` acts after `k`.
          - 'frequency_penalty': Used to reduce repetitiveness of generated tokens. The higher the value, the stronger a penalty is applied to previously present tokens,
                                 proportional to how many times they have already appeared in the prompt or prior generation.'
          - 'presence_penalty': Defaults to 0.0, min value of 0.0, max value of 1.0. Can be used to reduce repetitiveness of generated tokens.
                                Similar to `frequency_penalty`, except that this penalty is applied equally to all tokens that have already appeared, regardless of their exact frequencies.
          - 'return_likelihoods': One of GENERATION|ALL|NONE to specify how and if the token likelihoods are returned with the response. Defaults to NONE.
          - 'logit_bias': Used to prevent the model from generating unwanted tokens or to incentivize it to include desired tokens.
                          The format is {token_id: bias} where bias is a float between -10 and 10.

        """
        if not api_key:
            api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "CohereGenerator needs an API key to run. Either provide it as init parameter or set the env var COHERE_API_KEY."
            )

        self.api_key = api_key
        self.model_name = model_name
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
            model_name=self.model_name,
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
        :param prompt: The prompt to be sent to the generative model.
        """
        response = self.client.generate(
            model=self.model_name, prompt=prompt, stream=self.streaming_callback is not None, **self.model_parameters
        )
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

        metadata = [{"finish_reason": resp.finish_reason} for resp in response]
        replies = [resp.text for resp in response]
        self._check_truncated_answers(metadata)
        return {"replies": replies, "metadata": metadata}

    def _check_truncated_answers(self, metadata: List[Dict[str, Any]]):
        """
        Check the `finish_reason` returned with the Cohere response.
        If the `finish_reason` is `MAX_TOKEN`, log a warning to the user.
        :param metadata: The metadata returned by the Cohere API.
        """
        if metadata[0]["finish_reason"] == "MAX_TOKENS":
            logger.warning(
                "Responses have been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions."
            )
