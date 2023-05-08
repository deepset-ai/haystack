import json
import os
from typing import Optional, Dict, Union, List, Any, Callable
import logging

import requests
from transformers.pipelines import get_task

from haystack.environment import HAYSTACK_REMOTE_API_TIMEOUT_SEC, HAYSTACK_REMOTE_API_MAX_RETRIES
from haystack.errors import (
    HuggingFaceInferenceLimitError,
    HuggingFaceInferenceUnauthorizedError,
    HuggingFaceInferenceError,
)
from haystack.nodes.prompt.invocation_layer import (
    PromptModelInvocationLayer,
    TokenStreamingHandler,
    DefaultTokenStreamingHandler,
)
from haystack.utils.requests import request_with_retry

logger = logging.getLogger(__name__)
HF_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
HF_RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))


class HFInferenceEndpointInvocationLayer(PromptModelInvocationLayer):
    """
    A PromptModelInvocationLayer that invokes Hugging Face remote Inference Endpoint and API Inference to prompt the model.
    For more details see Hugging Face Inference API [documentation](https://huggingface.co/docs/api-inference/index)
    and Hugging Face Inference Endpoints [documentation](https://huggingface.co/inference-endpoints)

    The Inference API is free to use, and rate limited. If you need an inference solution for production, you can use
    Inference Endpoints service.

    See documentation for more details: https://huggingface.co/docs/inference-endpoints

    """

    def __init__(self, api_key: str, model_name_or_path: str, max_length: Optional[int] = 100, **kwargs):
        """
         Creates an instance of HFInferenceEndpointInvocationLayer
        :param model_name_or_path: can be either:
        a) Hugging Face Inference model name (i.e. google/flan-t5-xxl)
        b) Hugging Face Inference Endpoint URL (i.e. e.g. https://<your-unique-deployment-id>.us-east-1.aws.endpoints.huggingface.cloud)
        :param max_length: The maximum length of the output text.
        :param api_key: The Hugging Face API token. You’ll need to provide your user token which can
        be found in your Hugging Face account [settings](https://huggingface.co/settings/tokens)
        """
        super().__init__(model_name_or_path)
        self.prompt_preprocessors: Dict[str, Callable] = {}
        valid_api_key = isinstance(api_key, str) and api_key
        if not valid_api_key:
            raise ValueError(
                f"api_key {api_key} must be a valid Hugging Face token. "
                f"Your token is available in your Hugging Face settings page."
            )
        valid_model_name_or_path = isinstance(model_name_or_path, str) and model_name_or_path
        if not valid_model_name_or_path:
            raise ValueError(
                f"model_name_or_path {model_name_or_path} must be a valid Hugging Face inference endpoint URL."
            )
        self.api_key = api_key
        self.max_length = max_length

        # See https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
        # for a list of supported parameters
        self.model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "top_p",
                "top_k",
                "truncate",
                "typical_p",
                "best_of",
                "watermark",
                "details",
                "temperature",
                "repetition_penalty",
                "max_new_tokens",
                "max_time",
                "return_full_text",
                "num_return_sequences",
                "do_sample",
                "seed",
                "stream",
                "stream_handler",
            ]
            if key in kwargs
        }
        self.prompt_preprocessors["oasst"] = lambda prompt: f"<|prompter|>{prompt}<|endoftext|><|assistant|>"

    def preprocess_prompt(self, prompt: str):
        for key, prompt_preprocessor in self.prompt_preprocessors.items():
            if key in self.model_name_or_path:
                return prompt_preprocessor(prompt)
        return prompt

    @property
    def url(self) -> str:
        if HFInferenceEndpointInvocationLayer.is_inference_endpoint(self.model_name_or_path):
            # Inference Endpoint URL
            # i.e. https://o3x2xh3o4m47mxny.us-east-1.aws.endpoints.huggingface.cloud
            url = self.model_name_or_path

        else:
            url = f"https://api-inference.huggingface.co/models/{self.model_name_or_path}"
        return url

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

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
        prompt = self.preprocess_prompt(prompt)
        stop_words = kwargs.pop("stop_words", None) or []
        kwargs_with_defaults = self.model_input_kwargs
        if kwargs:
            if "max_new_tokens" not in kwargs_with_defaults:
                kwargs_with_defaults["max_new_tokens"] = self.max_length
            kwargs_with_defaults.update(kwargs)

        # either stream is True (will use default handler) or stream_handler is provided
        stream = (
            kwargs_with_defaults.get("stream", False) or kwargs_with_defaults.get("stream_handler", None) is not None
        )

        kwargs_with_defaults.update(kwargs)
        # see https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
        params = {
            "top_p": kwargs_with_defaults.get("top_p", None),
            "top_k": kwargs_with_defaults.get("top_k", None),
            "truncate": kwargs_with_defaults.get("truncate", None),
            "typical_p": kwargs_with_defaults.get("typical_p", None),
            "best_of": kwargs_with_defaults.get("best_of", None),
            "watermark": kwargs_with_defaults.get("watermark", False),
            "stop": kwargs_with_defaults.get("stop", stop_words),
            "details": kwargs_with_defaults.get("details", True),
            "temperature": kwargs_with_defaults.get("temperature", None),
            "repetition_penalty": kwargs_with_defaults.get("repetition_penalty", None),
            "max_new_tokens": kwargs_with_defaults.get("max_new_tokens", self.max_length),
            "max_time": kwargs_with_defaults.get("max_time", None),
            "return_full_text": kwargs_with_defaults.get("return_full_text", False),
            "num_return_sequences": kwargs_with_defaults.get("num_return_sequences", None),
            "do_sample": kwargs_with_defaults.get("do_sample", False),
        }
        response = self._post(data={"inputs": prompt, "parameters": params, "stream": stream}, stream=stream)
        if not stream:
            output = json.loads(response.text)
            generated_texts = [o["generated_text"] for o in output if "generated_text" in o]
        else:
            handler: TokenStreamingHandler = kwargs_with_defaults.pop("stream_handler", DefaultTokenStreamingHandler())
            generated_texts = self._process_streaming_response(
                response=response, stream_handler=handler, stop_words=stop_words
            )
        return generated_texts

    def _process_streaming_response(self, response, stream_handler: TokenStreamingHandler, stop_words=None):
        tokens: List[str] = []
        for byte_payload in response.iter_lines():
            # Skip line
            if byte_payload == b"\n":
                continue

            payload = byte_payload.decode("utf-8")

            # Event data
            if payload.startswith("data:"):
                # Decode payload
                json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                # JSON event payload looks like this, see docs for details:
                # {'details': None, 'generated_text': None,
                # 'token': {'id': 4943, 'logprob': -1.0859375, 'special': False, 'text': 'Here'}}

                token: str = self._extract_token(json_payload)
                if token is not None and token.strip() not in stop_words:
                    tokens.append(stream_handler(token, event_data=json_payload))
        return ["".join(tokens)]  # return a list of strings just like non-streaming

    def _extract_token(self, event_data: Dict[str, Any]):
        # extract token from event data and only consider non-special tokens
        return event_data["token"]["text"] if not event_data["token"]["special"] else None

    def _post(
        self,
        data: Dict[str, Any],
        stream: bool = False,
        attempts: int = HF_RETRIES,
        status_codes: Optional[List[int]] = None,
        timeout: float = HF_TIMEOUT,
        **kwargs,
    ) -> requests.Response:
        """
        Post data to the HF inference model. It takes in a prompt and returns a list of responses using a REST invocation.
        :param data: The data to be sent to the model.
        :param attempts: The number of attempts to make.
        :param status_codes: The status codes to retry on.
        :param timeout: The timeout for the request.
        :return: The responses are being returned.
        """
        response: requests.Response
        if status_codes is None:
            status_codes = [429]
        try:
            response = request_with_retry(
                method="POST",
                status_codes=status_codes,
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
                raise HuggingFaceInferenceLimitError(f"API rate limit exceeded: {res.text}")
            if res.status_code == 401:
                raise HuggingFaceInferenceUnauthorizedError(f"API key is invalid: {res.text}")

            raise HuggingFaceInferenceError(
                f"HuggingFace Inference returned an error.\nStatus code: {res.status_code}\nResponse body: {res.text}",
                status_code=res.status_code,
            )
        return response

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        # TODO: new implementation incoming for all layers, let's omit this for now
        return prompt

    @staticmethod
    def is_inference_endpoint(model_name_or_path: str) -> bool:
        return model_name_or_path is not None and all(
            token in model_name_or_path for token in ["https://", "endpoints"]
        )

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        if cls.is_inference_endpoint(model_name_or_path):
            return True
        else:
            # Check if the model is an HF inference API
            task_name: Optional[str] = None
            is_inference_api = False
            try:
                task_name = get_task(model_name_or_path, use_auth_token=kwargs.get("use_auth_token", None))
                is_inference_api = "api_key" in kwargs
            except RuntimeError:
                # This will fail for all non-HF models
                return False

            return is_inference_api and task_name in ["text2text-generation", "text-generation"]
