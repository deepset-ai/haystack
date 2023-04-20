from typing import Dict, Optional, Any, Tuple, List

import os
import json
import logging

import requests
from tenacity import retry, wait_exponential, retry_if_not_result
from tiktoken import get_encoding
from tiktoken.model import MODEL_TO_ENCODING
from generalimport import is_imported

from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.preview.nodes.prompt.providers.base import prompt_model_provider


logger = logging.getLogger(__name__)


OPENAI_TIMEOUT = float(os.environ.get("HAYSTACK_OPENAI_TIMEOUT_SEC", 30))


DEFAULT_MODEL_PARAMS = {
    "temperature": 0.7,
    "top_p": 1,
    "n": 1,
    "echo": False,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "best_of": 1,
    "logit_bias": {},
}


@prompt_model_provider
class GPT3Provider:
    """
    OUsed for OpenAI's GPT-3 InstructGPT models. Invocations are made using REST API.
    See [OpenAI GPT-3](https://platform.openai.com/docs/models/gpt-3) for more details.
    Supports Azure GPT3 models as well.
    """

    def __init__(
        self,
        model_name_or_path: str = "text-davinci-003",
        api_key: Optional[str] = None,
        max_length: Optional[int] = 100,
        azure_base_url: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: str = "2022-12-01",
        default_model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of OpenAIInvocationLayer for OpenAI's GPT-3 InstructGPT models.

        :param model_name_or_path: The name or path of the underlying model.
        :param api_key: The OpenAI API key. If empty, Haystack also check if an environment variable called
            `OPENAI_API_KEY` is set and read it from there.
        :param max_length: The maximum length of the output text.
        :param azure_base_url: [Optional for Azure OpenAI] the URL of the Azure OpenAI endpoint.
        :param azure_deployment: [Optional for Azure OpenAI] the name of the deployment.
        :param api_version: [Optional for Azure OpenAI] the API version to use.
        :param default_model_params: Additional parameters to pass to the underlying model by default.
            Relevant parameters include:
                - `suffix`
                - `temperature`
                - `top_p`
                - `presence_penalty`
                - `frequency_penalty`
                - `best_of`
                - `n`
                - `max_tokens`
                - `logit_bias`
                - `stop`
                - `echo`
                - `logprobs`
            Note that `stream` will always be False as this class does not support streaming yet.
            For more details about these parameters, see OpenAI
            [documentation](https://platform.openai.com/docs/api-reference/completions/create).
        """
        self.api_key = self._check_api_key(api_key)
        self.model_name_or_path = model_name_or_path
        self.tokenizer, self.max_tokens_limit = self._configure_tokenizer(model=model_name_or_path)
        self.url, self.headers = self._compile_url_and_headers(
            azure_base_url=azure_base_url, azure_deployment=azure_deployment, api_version=api_version
        )
        # 16 is the default length for answers from OpenAI shown in the docs
        # here, https://platform.openai.com/docs/api-reference/completions/create.
        # max_length must be set otherwise OpenAIInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_length or 16
        self.default_model_params = {**DEFAULT_MODEL_PARAMS, **(default_model_params or {})}
        self.default_model_params["max_tokens"] = self.max_length
        self.default_model_params["stream"] = False  # No support for streaming

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Returns True if the given model name (with the given arguments) is supported by this provider.

        :param model_name_or_path: the model identifier.
        :param **kwargs: any other argument needed to load this model.
        :returns: True if the model is compatible with this provider, False otherwise.
        """
        if not is_imported("requests") or not is_imported("tiktoken"):
            logger.debug("Either 'requests' or 'tiktoken' could not be imported. OpenAI GPT-3 models can't be invoked.")
            return False
        return any(m for m in ["ada", "babbage", "curie", "davinci"] if m in model_name_or_path)

    @retry(retry=retry_if_not_result(bool), wait=wait_exponential(min=1, max=10))
    def invoke(self, prompt: str, model_params: Optional[Dict[str, Any]] = None):
        """
        Sends a prompt the model.

        :param prompt: the prompt to send to the model
        :param model_params: any other parameter needed to invoke this model.
        :return: The responses from the model.
        """
        prompt = self.ensure_token_limit(prompt=prompt)

        all_params = self.default_model_params
        if model_params:
            all_params.update(self._translate_model_parameters(model_params))
        payload = {"model": self.model_name_or_path, "prompt": prompt, **all_params}

        response = requests.post(url=self.url, headers=self.headers, data=json.dumps(payload), timeout=OPENAI_TIMEOUT)
        if response.status_code != 200:
            openai_error: OpenAIError
            if response.status_code == 429:
                openai_error = OpenAIRateLimitError(f"API rate limit exceeded: {response.text}")
            else:
                openai_error = OpenAIError(
                    f"OpenAI returned an error.\n"
                    f"Status code: {response.status_code}\n"
                    f"Response body: {response.text}",
                    status_code=response.status_code,
                )
            raise openai_error

        self._check_truncated_answers(result=response.json())
        return [ans["text"] for ans in response["choices"]]

    def ensure_token_limit(self, prompt: str) -> str:
        """
        Ensure that the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.

        :param prompt: Prompt text to be sent to the generative model.
        :returns: the same prompt, cut down to the maximum length if it was too long. The tail of the prompt is cut.
        """
        n_prompt_tokens = len(self.tokenizer.encode(prompt))
        n_answer_tokens = self.max_length
        if n_prompt_tokens + n_answer_tokens <= self.max_tokens_limit:
            return prompt

        logger.error(
            "The prompt has been truncated from %s tokens to %s tokens such that the prompt length and "
            "answer length (%s tokens) fits within the max token limit (%s tokens). "
            "Reduce the length of the prompt to prevent it from being cut off.",
            n_prompt_tokens,
            self.max_tokens_limit - n_answer_tokens,
            n_answer_tokens,
            self.max_tokens_limit,
        )
        tokenized_payload = self.tokenizer.encode(prompt)
        decoded_string = self.tokenizer.decode(tokenized_payload[: self.max_tokens_limit - n_answer_tokens])
        return decoded_string

    def _check_api_key(self, api_key: Optional[str]) -> str:
        """
        Tries to load the API key and lightly validates it.

        :param api_key: the API key given in `__init__`, if any.
        :raises ValueError: if no valid key could be found.
        :returns: the api key to use.
        """
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", None)
        if not isinstance(api_key, str) or len(api_key) == 0:
            raise ValueError(
                "No valid OpenAI API key found. You can provide one by either setting an environment variable "
                "called `OPENAI_API_KEY`, or by passing one to the constructor of this class. Visit "
                "https://openai.com/api/ to get a key if you don't have one."
            )
        return api_key

    def _compile_url_and_headers(
        self, azure_base_url: Optional[str], azure_deployment: Optional[str], api_version: Optional[str]
    ) -> Tuple[str, str]:
        """
        Compiles the URL and headers to use for the API calls
        """
        if azure_base_url and azure_deployment:
            return (
                f"{azure_base_url}/openai/deployments/{azure_deployment}/completions?api-version={api_version}",
                {"api-key": self.api_key, "Content-Type": "application/json"},
            )
        if azure_base_url or azure_deployment:
            raise ValueError("To use Azure OpenAI you must provide both 'azure_base_url' and 'azure_deployment'.")
        return (
            "https://api.openai.com/v1/completions",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )

    def _configure_tokenizer(self, model: str) -> Tuple[int, Any]:
        """
        Returns the proper tokenizer and maximum token count for the given model name. Heuristic based.
        """
        if "davinci" in model:
            return (get_encoding(MODEL_TO_ENCODING.get(model, "p50k_base")), 4000)
        else:
            return (get_encoding("gpt2"), 2048)

    def _translate_model_parameters(self, model_params: Dict[str, Any]):
        """
        Some parameter names might need to be converted to be understood by OpenAI models.
        For example, we use 'stop_words' but OpenAI uses 'stop'
        """
        if "stop_words" in model_params.keys() and not "stop" in model_params.keys():
            model_params["stop"] = model_params.pop("stop_words")
        if "top_k" in model_params:
            top_k = model_params.pop("top_k")
            model_params["n"] = top_k
            model_params["best_of"] = top_k
        return model_params

    def _check_truncated_answers(self, result: Dict[str, Any]) -> None:
        """
        Check the `finish_reason` the answers returned by OpenAI completions endpoint. If the `finish_reason` is
        `length`, log a warning to the user.
        """
        truncated_completions = sum(1 for ans in result["choices"] if ans["finish_reason"] == "length")
        if truncated_completions > 0:
            logger.warning(
                "%s completions have been truncated before reaching a natural stopping point. "
                "Increase the `max_tokens` parameter to allow for longer completions.",
                truncated_completions,
            )
