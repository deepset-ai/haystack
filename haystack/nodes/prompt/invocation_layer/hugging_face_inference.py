import json
import os
from typing import Optional, Dict, Union, List
import logging

from transformers.pipelines import get_task

from haystack.environment import HAYSTACK_REMOTE_API_TIMEOUT_SEC
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer
from haystack.utils.requests import request_with_retry

logger = logging.getLogger(__name__)
HF_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 20))


class HFInferenceEndpointInvocationLayer(PromptModelInvocationLayer):
    """
    A PromptModelInvocationLayer that invokes Hugging Face remote Inference Endpoint and API Inference to prompt the model.
    For more details see Hugging Face Inference API [documentation](https://huggingface.co/docs/api-inference/index)
    and Hugging Face Inference Endpoints [documentation](https://huggingface.co/inference-endpoints)

    The Inference API is free to use, and rate limited. If you need an inference solution for production, you can use
    Inference Endpoints service.

    See documentation for more details: https://huggingface.co/docs/inference-endpoints

    """

    def __init__(self, api_key: str, model_name_or_path: str, max_length: Optional[int] = 500, **kwargs):
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
        self.max_length = max_length or 16

        # See https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
        # for a list of supported parameters
        self.model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "top_k",
                "top_p",
                "temperature",
                "repetition_penalty",
                "max_new_tokens",
                "max_time",
                "return_full_text",
                "num_return_sequences",
                "do_sample",
            ]
            if key in kwargs
        }

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
        stop_words = kwargs.pop("stop_words", None)

        kwargs_with_defaults = self.model_input_kwargs
        if "max_new_tokens" not in kwargs_with_defaults:
            kwargs_with_defaults["max_new_tokens"] = self.max_length
        if kwargs:
            if "top_k" in kwargs:
                top_k = kwargs.pop("top_k")
                kwargs["num_return_sequences"] = top_k
            kwargs_with_defaults.update(kwargs)
        # see https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
        accepted_params = [
            "top_p",
            "top_k",
            "temperature",
            "repetition_penalty",
            "max_new_tokens",
            "max_time",
            "return_full_text",
            "num_return_sequences",
            "do_sample",
        ]
        params = {key: kwargs_with_defaults.get(key) for key in accepted_params if key in kwargs_with_defaults}
        json_data = {"inputs": prompt, "parameters": params}
        response = request_with_retry(
            method="POST", url=self.url, headers=self.headers, json=json_data, timeout=HF_TIMEOUT
        )
        output = json.loads(response.text)
        generated_texts = [o["generated_text"] for o in output if "generated_text" in o]
        if stop_words:
            for idx, _ in enumerate(generated_texts):
                earliest_stop_word_idx = len(generated_texts[idx])
                for stop_word in stop_words:
                    stop_word_idx = generated_texts[idx].find(stop_word)
                    if stop_word_idx != -1:
                        earliest_stop_word_idx = min(earliest_stop_word_idx, stop_word_idx)
                generated_texts[idx] = generated_texts[idx][:earliest_stop_word_idx]

        return generated_texts

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
