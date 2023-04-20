from typing import Optional, Dict, Union, List, Any

import os
import json
import logging
from copy import deepcopy

import requests
from tenacity import retry, wait_exponential, retry_if_not_result
from transformers.pipelines import get_task

from haystack.errors import (
    HuggingFaceInferenceLimitError,
    HuggingFaceInferenceUnauthorizedError,
    HuggingFaceInferenceError,
)
from haystack.preview.nodes.prompt.providers.base import prompt_model_provider


logger = logging.getLogger(__name__)


HF_TIMEOUT = float(os.environ.get("HAYSTACK_HF_INFERENCE_API_TIMEOUT_SEC", 30))


DEFAULT_MODEL_PARAMS = {}


@prompt_model_provider
class HFRemoteProvider:
    """
    A PromptModelInvocationLayer that invokes Hugging Face remote Inference Endpoint and API Inference to prompt
    the model. For more details see Hugging Face Inference API
    [documentation](https://huggingface.co/docs/api-inference/index) and Hugging Face Inference Endpoints
    [documentation](https://huggingface.co/inference-endpoints)

    The Inference API is free to use, and rate limited. If you need an inference solution for production, you can
    use Inference Endpoints service.

    See documentation for more details: https://huggingface.co/docs/inference-endpoints
    """

    def __init__(
        self,
        model_name_or_path: str,
        api_key: Optional[str] = None,
        max_length: Optional[int] = 100,
        default_model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates a model provider for remote Hugging Face models using Inference API or Inference Endpoints.

        :param model_name_or_path: can be either:
            a) Hugging Face Inference model name (i.e. google/flan-t5-xxl)
            b) Hugging Face Inference Endpoint URL
               (i.e. e.g. https://<your-unique-deployment-id>.us-east-1.aws.endpoints.huggingface.cloud)
        :param max_length: The maximum length of the output text.
        :param api_key: The Hugging Face API token. You’ll need to provide your user token which can be found in
            your Hugging Face account [settings](https://huggingface.co/settings/tokens). If empty, Haystack also checks
            if an environment variable called `HF_INFERENCE_API_KEY` is set and reads the key from there.
        :param default_model_params: Additional parameters to pass to the underlying model by default.
            Relevant parameters include:
                - `top_k`
                - `top_p`
                - `temperature`
                - `repetition_penalty`
                - `max_new_tokens`
                - `max_time`
                - `return_full_text`
                - `num_return_sequences`
                - `do_sample`
                - `use_auth_token`
            For more details about these parameters, see HuggingFace
            [documentation](https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task).
        """
        if _is_inference_endpoint(model_name_or_path):
            # Inference Endpoint URL: i.e. https://o3x2xh3o4m47mxny.us-east-1.aws.endpoints.huggingface.cloud
            self.url = model_name_or_path
        else:
            # Inference API
            self.url = f"https://api-inference.huggingface.co/models/{model_name_or_path}"

        self.headers = {"Authorization": f"Bearer {self._check_api_key(api_key)}", "Content-Type": "application/json"}

        default_model_params = default_model_params or {}
        if "max_new_tokens" not in default_model_params:
            default_model_params["max_new_tokens"] = max_length

        self.default_model_params = {**DEFAULT_MODEL_PARAMS, **(default_model_params or {})}

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Returns True if the given model name (with the given arguments) is supported by this provider.

        :param model_name_or_path: the model identifier.
        :param **kwargs: any other argument needed to load this model.
        :returns: True if the model is compatible with this provider, False otherwise.
        """
        if _is_inference_endpoint(model_name_or_path):
            return True

        try:
            task_name = get_task(model_name_or_path, use_auth_token=kwargs.get("use_auth_token", None))
            return task_name in ["text2text-generation", "text-generation"]

        except RuntimeError as exc:
            logger.debug("'%s' doesn't seem to be a HF model: %s", model_name_or_path, str(exc))
        return False

    @retry(retry=retry_if_not_result(bool), wait=wait_exponential(min=1, max=10))
    def invoke(self, prompt: str, model_params: Optional[Dict[str, Any]] = None):
        """
        Sends a prompt the model. See https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
        for the parameters that this model can take.

        :param prompt: the prompt to send to the model
        :param model_params: any other parameter needed to invoke this model.
        :return: The responses from the model.
        """
        stop_words = model_params.pop("stop_words", None)

        params = deepcopy(self.default_model_params)
        if "top_k" in model_params:
            top_k = model_params.pop("top_k")
            model_params["num_return_sequences"] = top_k
        params.update(model_params)

        try:
            payload = {"inputs": prompt, "parameters": params}
            response = requests.post(url=self.url, headers=self.headers, data=json.dumps(payload), timeout=HF_TIMEOUT)
            output = json.loads(response.text)
            generated_texts = [o["generated_text"] for o in output if "generated_text" in o]

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

        if stop_words:
            generated_texts = [text.removesuffix(stop_word) for stop_word in stop_words for text in generated_texts]
        return generated_texts

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        # TODO
        return prompt

    def _check_api_key(self, api_key: Optional[str]) -> str:
        """
        Tries to load the API key and lightly validates it.

        :param api_key: the API key given in `__init__`, if any.
        :raises ValueError: if no valid key could be found.
        :returns: the api key to use.
        """
        if not api_key:
            api_key = os.environ.get("HF_INFERENCE_API_KEY", None)
        # if not isinstance(api_key, str):
        #     raise ValueError(
        #         "No valid HuggingFace Inference API key found. You can provide one by either setting an environment "
        #         "variable called `HF_INFERENCE_API_KEY`, or by passing one to the constructor of this class."
        #     )
        return api_key


def _is_inference_endpoint(model_name_or_path: str):
    """
    HF Inference Endpoints look like https://o3x2xh3o4m47mxny.us-east-1.aws.endpoints.huggingface.cloud
    """
    return all(token in model_name_or_path for token in ["https://", "huggingface"])
