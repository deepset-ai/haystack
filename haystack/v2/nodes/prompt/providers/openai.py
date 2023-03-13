from typing import Dict, Optional, Any, Union, Tuple

import json
import logging

from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.v2.nodes.prompt.providers import prompt_model_provider


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


@prompt_model_provider
class OpenAIInvocationLayer:
    """
    PromptModelInvocationLayer implementation for OpenAI's GPT-3 InstructGPT models. Invocations are made using REST API.
    See [OpenAI GPT-3](https://platform.openai.com/docs/models/gpt-3) for more details.
    """

    def __init__(
        self,
        api_key: str,
        model_name_or_path: str = "text-davinci-003",
        max_length: Optional[int] = 100,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
         Creates an instance of OpenAIInvocationLayer for OpenAI's GPT-3 InstructGPT models.

        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum length of the output text.
        :param api_key: The OpenAI API key.
        :param model_kwargs: Additional keyword arguments passed to the underlying model. The list of OpenAI-relevant
        kwargs includes: `suffix`, `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `best_of`, `n`, `max_tokens`,
        `logit_bias`, `stop`, `echo`, and `logprobs`. For more details about these kwargs, see OpenAI
        [documentation](https://platform.openai.com/docs/api-reference/completions/create).
        """
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.default_model_kwargs = {
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
            **model_kwargs,
        }
        if not isinstance(api_key, str) or len(api_key) == 0:
            raise ValueError(f"api_key {api_key} must be a valid OpenAI key. Visit https://openai.com/api/ to get one.")
        self.api_key = api_key

        # 16 is the default length for answers from OpenAI shown in the docs
        # here, https://platform.openai.com/docs/api-reference/completions/create.
        # max_length must be set otherwise OpenAIInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_length or 16

        tokenizer_name, max_tokens_limit = self._openai_text_completion_tokenization_details(
            model_name=self.model_name_or_path
        )
        self.max_tokens_limit = max_tokens_limit
        self._tokenizer = tiktoken.get_encoding(tokenizer_name=tokenizer_name)

    @property
    def url(self) -> str:
        return "https://api.openai.com/v1/completions"

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        valid_model = any(m for m in ["ada", "babbage", "davinci", "curie"] if m in model_name_or_path)
        return valid_model and kwargs.get("azure_base_url") is None

    def invoke(self, prompt: str, model_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invokes a prompt on the model. It takes in a prompt and returns a list of responses using a REST invocation.

        :return: The responses from the model.
        """
        kwargs = self.default_model_kwargs
        if model_kwargs:
            # we use keyword stop_words but OpenAI uses stop
            if "stop_words" in model_kwargs.keys() and not "stop" in model_kwargs.keys():
                model_kwargs["stop"] = model_kwargs.pop("stop_words")
            if "top_k" in model_kwargs:
                top_k = model_kwargs.pop("top_k")
                model_kwargs["n"] = top_k
                model_kwargs["best_of"] = top_k
        kwargs.update(model_kwargs)
        payload = {"model": self.model_name_or_path, "prompt": prompt, **kwargs}
        res = self.openai_request(url=self.url, headers=self.headers, payload=payload)
        self._check_openai_text_completion_answers(result=res, payload=payload)
        responses = [ans["text"].strip() for ans in res["choices"]]
        return responses

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

    def _openai_text_completion_tokenization_details(self, model_name: str):
        """Return the tokenizer name and max tokens limit for a given OpenAI `model_name`.

        :param model_name: Name of the OpenAI model.
        """
        tokenizer_name = "gpt2"
        if "davinci" in model_name:
            max_tokens_limit = 4000
            tokenizer_name = MODEL_TO_ENCODING.get(model_name, "p50k_base")
        else:
            max_tokens_limiapi_keyt = 2048
        return tokenizer_name, max_tokens_limit

    @retry(retry=retry_if_not_result(bool), wait=wait_exponential(min=1, max=10))
    def openai_request(
        self, url: str, headers: Dict, payload: Dict, timeout: Union[float, Tuple[float, float]] = OPENAI_TIMEOUT
    ):
        """Make a request to the OpenAI API given a `url`, `headers`, `payload`, and `timeout`.

        :param url: The URL of the OpenAI API.
        :param headers: Dictionary of HTTP Headers to send with the :class:`Request`.
        :param payload: The payload to send with the request.
        :param timeout: The timeout length of the request. The default is 30s.
        """
        if not REQUESTS_IMPORTED:
            raise ImportError(
                "requests could not be imported. OpenAI and Azure models won't work. "
                "Run 'pip install requests' in your virtualenv to fix the issue."
            )

        response = requests.request("POST", url, headers=headers, data=json.dumps(payload), timeout=timeout)
        res = json.loads(response.text)

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

        return res

    def _check_openai_text_completion_answers(self, result: Dict, payload: Dict) -> None:
        """
        Check the `finish_reason` the answers returned by OpenAI completions endpoint. If the `finish_reason` is `length`,
        log a warning to the user.

        :param result: The result returned from the OpenAI API.
        :param payload: The payload sent to the OpenAI API.
        """
        number_of_truncated_completions = sum(1 for ans in result["choices"] if ans["finish_reason"] == "length")
        if number_of_truncated_completions > 0:
            logger.warning(
                "%s out of the %s completions have been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions.",
                number_of_truncated_completions,
                payload["n"],
            )


class AzureOpenAIInvocationLayer(OpenAIInvocationLayer):
    """
    Azure OpenAI Invocation Layer

    This layer is used to invoke the OpenAI API on Azure. It is essentially the same as the OpenAIInvocationLayer
    with additional two parameters: azure_base_url and azure_deployment_name. The azure_base_url is the URL of the Azure OpenAI
    endpoint and the azure_deployment_name is the name of the deployment.
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
