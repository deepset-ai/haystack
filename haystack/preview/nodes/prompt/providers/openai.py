from typing import Dict, Optional, Any, Union, Tuple

import os
import json
import logging

from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.preview.nodes.prompt.providers.base import prompt_model_provider


logger = logging.getLogger(__name__)


TIKTOKEN_IMPORTED = False
try:
    import tiktoken
    from tiktoken.model import MODEL_TO_ENCODING

    TIKTOKEN_IMPORTED = True
except ImportError as exc:
    logger.debug("tiktoken could not be imported. OpenAI and Azure models won't work.")


REQUESTS_IMPORTED = False
try:
    import requests

    REQUESTS_IMPORTED = True
except ImportError as exc2:
    logger.debug("requests could not be imported. OpenAI and Azure models won't work.")


try:
    from tenacity import retry, wait_exponential, retry_if_not_result
except ImportError as exc3:
    logger.debug("tenacity could not be imported. Requests to OpenAI won't be attempted again if they fail.")

    # No-op retry decorator in case 'tenacity' is not installed.
    def retry(func):
        return func


OPENAI_TIMEOUT = float(os.environ.get("HAYSTACK_OPENAI_TIMEOUT_SEC", 30))


@prompt_model_provider
class OpenAIProvider:
    """
    OUsed for OpenAI's GPT-3 InstructGPT models. Invocations are made using REST API.
    See [OpenAI GPT-3](https://platform.openai.com/docs/models/gpt-3) for more details.
    """

    def __init__(
        self,
        model_name_or_path: str = "text-davinci-003",
        api_key: Optional[str] = None,
        max_length: Optional[int] = 100,
        default_model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of OpenAIInvocationLayer for OpenAI's GPT-3 InstructGPT models.

        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum length of the output text.
        :param api_key: The OpenAI API key. If empty, Haystack also check if an environment variable called
            `OPENAI_API_KEY` is set and read it from there.
        :param default_model_params: Additional parameters to pass to the underlying model. Relevant parameters include:
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
        For more details about these parameters, see OpenAI
        [documentation](https://platform.openai.com/docs/api-reference/completions/create).
        """
        self.api_key = self._check_api_key(api_key)
        self.model_name_or_path = model_name_or_path
        self.default_model_params = {
            "max_tokens": max_length,
            "temperature": 0.7,
            "top_p": 1,
            "n": 1,
            "stream": False,  # no support for streaming
            "echo": False,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "best_of": 1,
            "logit_bias": {},
            **default_model_params,
        }
        self.url = "https://api.openai.com/v1/completions"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        if "davinci" in model_name_or_path:
            self.max_tokens_limit = 4000
            self._tokenizer = tiktoken.get_encoding(MODEL_TO_ENCODING.get(model_name_or_path, "p50k_base"))
        else:
            self.max_tokens_limit = 2048
            self._tokenizer = tiktoken.get_encoding(MODEL_TO_ENCODING.get("gpt2", "p50k_base"))

        # 16 is the default length for answers from OpenAI shown in the docs
        # here, https://platform.openai.com/docs/api-reference/completions/create.
        # max_length must be set otherwise OpenAIInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_length or 16

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        valid_model = any(m for m in ["ada", "babbage", "curie", "davinci"] if m in model_name_or_path)
        return valid_model and kwargs.get("azure_base_url") is None

    @retry(retry=retry_if_not_result(bool), wait=wait_exponential(min=1, max=10))
    def invoke(self, prompt: str, model_params: Optional[Dict[str, Any]] = None):
        """
        Invokes a prompt on the model. It takes in a prompt and returns a list of responses.

        :return: The responses from the model.
        """
        if not REQUESTS_IMPORTED:
            raise ImportError(
                "requests could not be imported. OpenAI and Azure models won't work. "
                "Run 'pip install requests' in your virtualenv to fix the issue."
            )

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
        res = response.json()
        self._check_truncated_answers(result=res, expected_answers_count=all_params["n"])

        responses = [ans["text"] for ans in res["choices"]]
        return responses

    def _check_api_key(self, api_key) -> None:
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

    def _check_truncated_answers(self, result: Dict, expected_answers_count: int) -> None:
        """
        Check the `finish_reason` the answers returned by OpenAI completions endpoint. If the `finish_reason` is
        `length`, log a warning to the user.

        :param result: The result returned from the OpenAI API.
        :param payload: The payload sent to the OpenAI API.
        """
        number_of_truncated_completions = sum(1 for ans in result["choices"] if ans["finish_reason"] == "length")
        if number_of_truncated_completions > 0:
            logger.warning(
                "%s out of the %s completions have been truncated before reaching a natural stopping point. "
                "Increase the `max_tokens` parameter to allow for longer completions.",
                number_of_truncated_completions,
                expected_answers_count,
            )

    def _ensure_token_limit(self, prompt: str) -> str:
        """Ensure that the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.

        :param prompt: Prompt text to be sent to the generative model.
        """
        n_prompt_tokens = self.tokenizer.encode(prompt)
        n_answer_tokens = self.max_length
        if (n_prompt_tokens + n_answer_tokens) <= self.max_tokens_limit:
            return prompt

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens such that the prompt length and "
            "answer length (%s tokens) fits within the max token limit (%s tokens). "
            "Reduce the length of the prompt to prevent it from being cut off.",
            n_prompt_tokens,
            self.max_tokens_limit - n_answer_tokens,
            n_answer_tokens,
            self.max_tokens_limit,
        )

        tokenized_payload = self._tokenizer.encode(prompt)
        decoded_string = self._tokenizer.decode(tokenized_payload[: self.max_tokens_limit - n_answer_tokens])

        return decoded_string


class AzureOpenAIProvider(OpenAIProvider):
    """
    Used to invoke the OpenAI API on Azure. It is essentially the same as the OpenAIProvider
    with additional two parameters: azure_base_url and azure_deployment_name. The azure_base_url is the URL of the Azure
    OpenAI endpoint and the azure_deployment_name is the name of the deployment.
    """

    def __init__(
        self,
        azure_base_url: str,
        azure_deployment_name: str,
        api_key: str,
        api_version: str = "2022-12-01",
        model_name_or_path: str = "text-davinci-003",
        max_length: Optional[int] = 100,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            api_key=api_key, model_name_or_path=model_name_or_path, max_length=max_length, model_kwargs=model_kwargs
        )
        self.azure_base_url = azure_base_url
        self.azure_deployment_name = azure_deployment_name
        self.api_version = api_version

    @property
    def url(self) -> str:
        return f"{self.azure_base_url}/openai/deployments/{self.azure_deployment_name}/completions?api-version={self.api_version}"

    @property
    def headers(self) -> Dict[str, str]:
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Ensures Azure OpenAI Invocation Layer is selected when azure_base_url and azure_deployment_name are provided in
        addition to a list of supported models.
        """
        valid_model = any(m for m in ["ada", "babbage", "davinci", "curie"] if m in model_name_or_path)
        return (
            valid_model and kwargs.get("azure_base_url") is not None and kwargs.get("azure_deployment_name") is not None
        )
