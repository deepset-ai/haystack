import json
import os
from typing import Optional, Dict, Union, List, Any
import logging

import requests

from haystack.environment import HAYSTACK_REMOTE_API_TIMEOUT_SEC, HAYSTACK_REMOTE_API_MAX_RETRIES
from haystack.errors import CohereInferenceLimitError, CohereUnauthorizedError, CohereError
from haystack.nodes.prompt.invocation_layer import (
    PromptModelInvocationLayer,
    TokenStreamingHandler,
    DefaultTokenStreamingHandler,
)
from haystack.nodes.prompt.invocation_layer.handlers import DefaultPromptHandler
from haystack.utils.requests import request_with_retry

logger = logging.getLogger(__name__)
TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))


class CohereInvocationLayer(PromptModelInvocationLayer):
    """
    PromptModelInvocationLayer implementation for Cohere's command models. Invocations are made using REST API.
    """

    def __init__(self, api_key: str, model_name_or_path: str, max_length: Optional[int] = 100, **kwargs):
        """
         Creates an instance of CohereInvocationLayer for the specified Cohere model

        :param api_key: Cohere API key
        :param model_name_or_path: Cohere model name
        :param max_length: The maximum length of the output text.
        """
        super().__init__(model_name_or_path)
        valid_api_key = isinstance(api_key, str) and api_key
        if not valid_api_key:
            raise ValueError(
                f"api_key {api_key} must be a valid Cohere token. "
                f"Your token is available in your Cohere settings page."
            )
        valid_model_name_or_path = isinstance(model_name_or_path, str) and model_name_or_path
        if not valid_model_name_or_path:
            raise ValueError(f"model_name_or_path {model_name_or_path} must be a valid Cohere model name")
        self.api_key = api_key
        self.max_length = max_length

        # See https://docs.cohere.com/reference/generate
        # for a list of supported parameters
        self.model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "end_sequences",
                "frequency_penalty",
                "k",
                "logit_bias",
                "max_tokens",
                "model",
                "num_generations",
                "p",
                "presence_penalty",
                "return_likelihoods",
                "stream",
                "stream_handler",
                "temperature",
                "truncate",
            ]
            if key in kwargs
        }
        # cohere uses BPE tokenizer
        # the tokenization lengths are very close to gpt2, in our experiments the differences were minimal
        # See model info at https://docs.cohere.com/docs/models
        model_max_length = 4096 if "command" in model_name_or_path else 2048
        self.prompt_handler = DefaultPromptHandler(
            model_name_or_path="gpt2", model_max_length=model_max_length, max_length=self.max_length or 100
        )

    @property
    def url(self) -> str:
        return "https://api.cohere.ai/v1/generate"

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Request-Source": "python-sdk",
        }

    def invoke(self, *args, **kwargs):
        """
        Invokes a prompt on the model. It takes in a prompt and returns a list of responses using a REST invocation.
        :return: The responses are being returned.
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                f"No prompt provided. Model {self.model_name_or_path} requires prompt."
                f"Make sure to provide prompt in kwargs."
            )
        stop_words = kwargs.pop("stop_words", None)
        kwargs_with_defaults = self.model_input_kwargs
        kwargs_with_defaults.update(kwargs)

        # either stream is True (will use default handler) or stream_handler is provided
        stream = (
            kwargs_with_defaults.get("stream", False) or kwargs_with_defaults.get("stream_handler", None) is not None
        )

        # see https://docs.cohere.com/reference/generate
        params = {
            "end_sequences": kwargs_with_defaults.get("end_sequences", stop_words),
            "frequency_penalty": kwargs_with_defaults.get("frequency_penalty", None),
            "k": kwargs_with_defaults.get("k", None),
            "max_tokens": kwargs_with_defaults.get("max_tokens", self.max_length),
            "model": kwargs_with_defaults.get("model", self.model_name_or_path),
            "num_generations": kwargs_with_defaults.get("num_generations", None),
            "p": kwargs_with_defaults.get("p", None),
            "presence_penalty": kwargs_with_defaults.get("presence_penalty", None),
            "prompt": prompt,
            "return_likelihoods": kwargs_with_defaults.get("return_likelihoods", None),
            "stream": stream,
            "temperature": kwargs_with_defaults.get("temperature", None),
            "truncate": kwargs_with_defaults.get("truncate", None),
        }
        response = self._post(params, stream=stream)
        if not stream:
            output = json.loads(response.text)
            generated_texts = [o["text"] for o in output["generations"] if "text" in o]
        else:
            handler: TokenStreamingHandler = kwargs_with_defaults.pop("stream_handler", DefaultTokenStreamingHandler())
            generated_texts = self._process_streaming_response(response=response, stream_handler=handler)
        return generated_texts

    def _process_streaming_response(self, response, stream_handler: TokenStreamingHandler):
        # sseclient doesn't work with Cohere streaming API
        # let's do it manually
        tokens = []
        for line in response.iter_lines():
            if line:
                streaming_item = json.loads(line)
                text = streaming_item.get("text")
                if text:
                    tokens.append(stream_handler(text))
        return ["".join(tokens)]  # return a list of strings just like non-streaming

    def _post(
        self,
        data: Dict[str, Any],
        stream: bool = False,
        attempts: int = RETRIES,
        status_codes_to_retry: Optional[List[int]] = None,
        timeout: float = TIMEOUT,
        **kwargs,
    ) -> requests.Response:
        """
        Post data to the HF inference model. It takes in a prompt and returns a list of responses using a REST
        invocation.
        :param data: The data to be sent to the model.
        :param stream: Whether to stream the response.
        :param attempts: The number of attempts to make.
        :param status_codes_to_retry: The status codes to retry on.
        :param timeout: The timeout for the request.
        :return: The response from the model as a requests.Response object.
        """
        response: requests.Response
        if status_codes_to_retry is None:
            status_codes_to_retry = [429]
        try:
            response = request_with_retry(
                method="POST",
                status_codes_to_retry=status_codes_to_retry,
                attempts=attempts,
                url=self.url,
                headers=self.headers,
                json=data,
                timeout=timeout,
                stream=stream,
            )
        except requests.HTTPError as err:
            res = err.response
            if res.status_code == 429:
                raise CohereInferenceLimitError(f"API rate limit exceeded: {res.text}")
            if res.status_code == 401:
                raise CohereUnauthorizedError(f"API key is invalid: {res.text}")

            raise CohereError(
                f"Cohere model returned an error.\nStatus code: {res.status_code}\nResponse body: {res.text}",
                status_code=res.status_code,
            )
        return response

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        # the prompt for this model will be of the type str
        resize_info = self.prompt_handler(prompt)  # type: ignore
        if resize_info["prompt_length"] != resize_info["new_prompt_length"]:
            logger.warning(
                "The prompt has been truncated from %s tokens to %s tokens so that the prompt length and "
                "answer length (%s tokens) fit within the max token limit (%s tokens). "
                "Shorten the prompt to prevent it from being cut off",
                resize_info["prompt_length"],
                max(0, resize_info["model_max_length"] - resize_info["max_length"]),  # type: ignore
                resize_info["max_length"],
                resize_info["model_max_length"],
            )
        return prompt

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Ensures CohereInvocationLayer is selected only when Cohere models are specified in
        the model name.
        """
        is_inference_api = "api_key" in kwargs
        return (
            model_name_or_path is not None
            and is_inference_api
            and any(token == model_name_or_path for token in ["command", "command-light", "base", "base-light"])
        )
